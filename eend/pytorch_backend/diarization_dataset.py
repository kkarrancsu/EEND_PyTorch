#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.
#
import torch
import numpy as np
from eend import kaldi_data
from eend import feature

def _count_frames(data_len, size, step):
    # no padding at edges, last remaining samples are ignored
    return int((data_len - size + step) / step)


def _gen_frame_indices(
        data_length, size=2000, step=2000,
        use_last_samples=False,
        label_delay=0,
        subsampling=1,
        n_gpu=1, min_samps_per_gpu=2):
    i = -1
    for i in range(_count_frames(data_length, size, step)):
        st = i*step
        ed = i*step + size
        if (ed - st) >= n_gpu * min_samps_per_gpu:
            yield i * step, i * step + size
    if use_last_samples and i * step + size < data_length:
        if data_length - (i + 1) * step - subsampling * label_delay > 0:
            st = (i+1)*step
            ed = data_length
            if (ed - st) >= n_gpu * min_samps_per_gpu:
                yield (i + 1) * step, data_length


def my_collate(batch):
    data, target = list(zip(*batch))
    return [data, target]


class KaldiDiarizationDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir,
            chunk_size=2000,
            context_size=0,
            frame_size=1024,
            frame_shift=256,
            subsampling=1,
            rate=16000,
            input_transform=None,
            use_last_samples=False,
            label_delay=0,
            n_speakers=None,
            n_gpu=1,
            logger=None
            ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.context_size = context_size
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.subsampling = subsampling
        self.input_transform = input_transform
        self.n_speakers = n_speakers
        self.chunk_indices = []
        self.label_delay = label_delay
        self.min_samps_per_gpu = n_gpu
        self.n_gpu = n_gpu   # defaults to being set to 1, which is OK for CPU training
        self.logger = logger

        self.data = kaldi_data.KaldiData(self.data_dir)

        # make chunk indices: filepath, start_frame, end_frame
        for rec in self.data.wavs:
            data_len = int(self.data.reco2dur[rec] * rate / frame_shift)
            data_len = int(data_len / self.subsampling)
            for st, ed in _gen_frame_indices(
                    data_len, chunk_size, chunk_size, use_last_samples,
                    label_delay=self.label_delay,
                    subsampling=self.subsampling,
                    n_gpu=self.n_gpu, min_samps_per_gpu=self.min_samps_per_gpu):
                self.chunk_indices.append(
                        (rec, st * self.subsampling, ed * self.subsampling))
        print(len(self.chunk_indices), " chunks")

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, i):
        rec, st, ed = self.chunk_indices[i]
        Y, T = feature.get_labeledSTFT(
            self.data,
            rec,
            st,
            ed,
            self.frame_size,
            self.frame_shift,
            self.n_speakers)
        # Y: (frame, num_ceps)
        YY = feature.transform(Y, self.input_transform)
        # Y_spliced: (frame, num_ceps * (context_size * 2 + 1))
        Y_spliced = feature.splice(YY, self.context_size)
        # Y_ss: (frame / subsampling, num_ceps * (context_size * 2 + 1))
        
        # To enable Multi-GPU support, Y_ss.shape[0] > n_gpu, ideally, it should be
        # a multiple of n_gpu.  Thus, we add in a check to determine whether
        # subsampling should be skipped, in the case of "end-of-sequence" situations
        # where the subsampling would create less samples than the # of gpu's, 
        # which creates a problem of not being able to distribute the samples
        # across all the gpus.
        if Y_spliced.shape[0]//self.subsampling >= self.n_gpu:
            Y_ss, T_ss = feature.subsample(Y_spliced, T, self.subsampling)
        else:
            actual_samps_per_gpu = Y_spliced.shape[0]/self.n_gpu
            if actual_samps_per_gpu >= self.min_samps_per_gpu:
                Y_ss, T_ss = Y_spliced, T
                str_to_log = 'Not subsampling item: %d to enable multi-gpu training' % (i, )
                if self.logger is not None:
                    self.logger.warning(str_to_log)
                else:
                    print(str_to_log)
                #import pdb; pdb.set_trace()
            else:
                """
                # pad w/ replicated data to enable multi-gpu training
                # NOTE: if you get any gradient blow-ups, this might be a good place to investigate
                #  and determine a better strategy for how to replicate teh data smartly,
                #  but I suspect this won't actually be a problem. 
                n_repeat = self.n_gpu // Y_spliced.shape[0] + 1
                Y_ss = np.repeat(Y_spliced, n_repeat, axis=0)
                T_ss = np.repeat(T, n_repeat, axis=0)
                str_to_log ='Duplicating data to item: %d by: %d to enable multi-gpu training' % (i, n_repeat, ) 
                if self.logger is not None:
                    logger.warning(str_to_log)
                else:
                    print(str_to_log)
                """
                # we shouldn't get here, due to the updates in the _gen_frame_indices function
                raise Exception("you are trying to use data which is unsuppored with multi-gpu traiing!")

        Y_ss = torch.from_numpy(Y_ss).float()
        T_ss = torch.from_numpy(T_ss).float()
        return Y_ss, T_ss
