# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from itertools import permutations

"""
T: number of frames
C: number of speakers (classes)
D: dimension of embedding (for deep clustering loss)
B: mini-batch size
"""


def pad_labels(ts, out_size):
    # https://github.com/hitachi-speech/EEND/blob/95216f0025dde8f0358d71412be046a36a030b33/eend/chainer_backend/models.py#L230
    # pad label's speaker-dim to be model's n_speakers
    for i, t in enumerate(ts):
        if t.shape[1] < out_size:
            # padding
            ts[i] = F.pad(
                t,
                [(0, 0), (0, out_size - t.shape[1])],
                mode='constant',
                value=0.,
            )
        elif t.shape[1] > out_size:
            # truncate
            raise ValueError
    return ts


def pad_results(ys, out_size):
    # https://github.com/hitachi-speech/EEND/blob/95216f0025dde8f0358d71412be046a36a030b33/eend/chainer_backend/models.py#L248
    # pad label's speaker-dim to be model's n_speakers
    ys_padded = []
    for i, y in enumerate(ys):
        if y.shape[1] < out_size:
            # padding
            ys_padded.append(torch.cat([y, torch.zeros((y.shape[0], out_size - y.shape[1]), dtype=y.dtype)], dim=1))
        elif y.shape[1] > out_size:
            # truncate
            raise ValueError
        else:
            ys_padded.append(y)
    return ys_padded


def pit_loss(pred, label, label_delay=0):
    """
    Permutation-invariant training (PIT) cross entropy loss function.

    Args:
      pred:  (T,C)-shaped pre-activation values
      label: (T,C)-shaped labels in {0,1}
      label_delay: if label_delay == 5:
            pred: 0 1 2 3 4 | 5 6 ... 99 100 |
          label: x x x x x | 0 1 ... 94  95 | 96 97 98 99 100
          calculated area: | <------------> |

    Returns:
      min_loss: (1,)-shape mean cross entropy
      label_perms[min_index]: permutated labels
    """
    # label permutations along the speaker axis
    label_perms = [label[..., list(p)] for p
                    in permutations(range(label.shape[-1]))]
    losses = torch.stack(
        [F.binary_cross_entropy_with_logits(
            pred[label_delay:, ...],
            l[:len(l) - label_delay, ...]) for l in label_perms])
    min_loss = losses.min() * (len(label) - label_delay)
    min_index = losses.argmin().detach()
    
    return min_loss, label_perms[min_index]


def batch_pit_loss(ys, ts, label_delay=0):
    """
    PIT loss over mini-batch.

    Args:
      ys: B-length list of predictions
      ts: B-length list of labels

    Returns:
      loss: (1,)-shape mean cross entropy over mini-batch
      labels: B-length list of permuted labels
    """
    loss_w_labels = [pit_loss(y, t, label_delay)
                     for (y, t) in zip(ys, ts)]
    losses, labels = zip(*loss_w_labels)
    loss = torch.stack(losses).sum()
    n_frames = np.sum([t.shape[0] for t in ts])
    loss = loss / n_frames
    return loss, labels


def batch_pit_n_speaker_loss(ys, ts, n_speakers_list):
    """
    PIT loss over mini-batch.
    Args:
      ys: B-length list of predictions (pre-activations)
      ts: B-length list of labels
      n_speakers_list: list of n_speakers in batch
    Returns:
      loss: (1,)-shape mean cross entropy over mini-batch
      labels: B-length list of permuted labels
    """
    max_n_speakers = ts[0].shape[1]
    # (B, T, C)
    ys = nn.utils.rnn.pad_sequence(ys, padding_value=-1)

    losses = []
    for shift in range(max_n_speakers):
        # rolled along with speaker-axis
        ts_roll = [torch.roll(t, -shift, dims=1) for t in ts]
        ts_roll = nn.utils.rnn.pad_sequence(ts_roll, padding_value=-1)
        # loss: (B, T, C)
        loss = F.binary_cross_entropy_with_logits(ys, ts_roll, reduction='none')
        # sum over time: (B, C)
        loss = torch.sum(loss, dim=1)
        losses.append(loss)
    # losses: (B, C, C)
    losses = torch.stack(losses, dim=2)
    # losses[b, i, j] is a loss between
    # `i`-th speaker in y and `(i+j)%C`-th speaker in t

    perms = torch.IntTensor(list(permutations(range(max_n_speakers))))
    # y_ind: [0,1,2,3]
    y_ind = torch.arange(max_n_speakers, dtype=torch.int32)
    #  perms  -> relation to t_inds      -> t_inds
    # 0,1,2,3 -> 0+j=0,1+j=1,2+j=2,3+j=3 -> 0,0,0,0
    # 0,1,3,2 -> 0+j=0,1+j=1,2+j=3,3+j=2 -> 0,0,1,3
    t_inds = torch.remainder(perms - y_ind, max_n_speakers)

    losses_perm = []
    for t_ind in t_inds:
        losses_perm.append(
            torch.mean(losses[:, y_ind, t_ind], dim=1))
    # losses_perm: (B, Perm)
    losses_perm = torch.stack(losses_perm, dim=1)

    # masks: (B, Perms)
    def select_perm_indices(num, max_num):
        perms = list(permutations(range(max_num)))
        sub_perms = list(permutations(range(num)))
        return [
            [x[:num] for x in perms].index(perm)
            for perm in sub_perms]
    masks = torch.full_like(losses_perm, np.inf)
    for i, t in enumerate(ts):
        n_speakers = n_speakers_list[i]
        indices = select_perm_indices(n_speakers, max_n_speakers)
        masks[i, indices] = 0
    losses_perm += masks

    min_loss = torch.sum(torch.min(losses_perm, dim=1)[0])  # [0] is the min values, [1] is indices
    n_frames = np.sum([t.shape[0] for t in ts])
    min_loss = min_loss / n_frames

    min_indices = torch.argmin(losses_perm, dim=1)
    labels_perm = [t[:, perms[idx]] for t, idx in zip(ts, min_indices)]
    labels_perm = [t[:, :n_speakers] for t, n_speakers in zip(labels_perm, n_speakers_list)]

    return min_loss, labels_perm


def standard_loss(ys, ts, label_delay=0):
    losses = [F.binary_cross_entropy_with_logits(y, t) * len(y) for y, t in zip(ys, ts)]
    loss = torch.sum(torch.stack(losses))
    n_frames = torch.sum([t.shape[0] for t in ts])
    loss = loss / n_frames
    return loss


def eda_batch_pit_loss(ys, ts, attractor_logits, attractor_loss_ratio=1.0):
    n_speakers = [t.shape[1] for t in ts]

    # compute attractor loss
    labels = torch.cat([torch.IntTensor([[1] * n_spk + [0]]) for n_spk in n_speakers], dim=1)
    attractor_loss = F.binary_cross_entropy_with_logits(attractor_logits, labels)

    max_n_speakers = max(n_speakers)
    ts_padded = pad_labels(ts, max_n_speakers)
    ys_padded = pad_results(ys, max_n_speakers)

    _, labels = batch_pit_n_speaker_loss(ys_padded, ts_padded, n_speakers)
    loss = standard_loss(ys, labels)

    total_loss = loss + attractor_loss * attractor_loss_ratio
    return total_loss, attractor_loss, loss



def calc_diarization_error(pred, label, label_delay=0):
    """
    Calculates diarization error stats for reporting.

    Args:
      pred (torch.FloatTensor): (T,C)-shaped pre-activation values
      label (torch.FloatTensor): (T,C)-shaped labels in {0,1}
      label_delay: if label_delay == 5:
           pred: 0 1 2 3 4 | 5 6 ... 99 100 |
          label: x x x x x | 0 1 ... 94  95 | 96 97 98 99 100
          calculated area: | <------------> |

    Returns:
      res: dict of diarization error stats
    """
    label = label[:len(label) - label_delay, ...]
    decisions = torch.sigmoid(pred[label_delay:, ...]) > 0.5
    n_ref = label.sum(axis=-1).long()
    n_sys = decisions.sum(axis=-1).long()
    res = {}
    res['speech_scored'] = (n_ref > 0).sum()
    res['speech_miss'] = ((n_ref > 0) & (n_sys == 0)).sum()
    res['speech_falarm'] = ((n_ref == 0) & (n_sys > 0)).sum()
    res['speaker_scored'] = (n_ref).sum()
    res['speaker_miss'] = torch.max((n_ref - n_sys), torch.zeros_like(n_ref)).sum()
    res['speaker_falarm'] = torch.max((n_sys - n_ref), torch.zeros_like(n_ref)).sum()
    n_map = ((label == 1) & (decisions == 1)).sum(axis=-1)
    res['speaker_error'] = (torch.min(n_ref, n_sys) - n_map).sum()
    res['correct'] = (label == decisions).sum() / label.shape[1]
    res['diarization_error'] = (
        res['speaker_miss'] + res['speaker_falarm'] + res['speaker_error'])
    res['frames'] = len(label)
    return res


def report_diarization_error(ys, labels):
    """
    Reports diarization errors
    Should be called with torch.no_grad

    Args:
      ys: B-length list of predictions (torch.FloatTensor)
      labels: B-length list of labels (torch.FloatTensor)
    """
    stats_avg = {}
    cnt = 0
    for y, t in zip(ys, labels):
        stats = calc_diarization_error(y, t)
        for k, v in stats.items():
            stats_avg[k] = stats_avg.get(k, 0) + float(v)
        cnt += 1
    
    stats_avg = {k:v/cnt for k,v in stats_avg.items()}
    return stats_avg
        
