# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.

import math

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from torch.optim.lr_scheduler import _LRScheduler

from torch.nn.utils.rnn import pad_sequence, pack_sequence, pad_packed_sequence

class NoamScheduler(_LRScheduler):
    """
    See https://arxiv.org/pdf/1706.03762.pdf
    lrate = d_model**(-0.5) * \
            min(step_num**(-0.5), step_num*warmup_steps**(-1.5))
    Args:
        d_model: int
            The number of expected features in the encoder inputs.
        warmup_steps: int
            The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(NoamScheduler, self).__init__(optimizer, last_epoch)

        # the initial learning rate is set as step = 1
        if self.last_epoch == -1:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
            self.last_epoch = 0
        print(self.d_model)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.d_model ** (-0.5) * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


class DiarizationTransformerEncoder(nn.Module):
    def __init__(self, n_speakers, in_size, n_heads, n_units, n_layers,
                 dim_feedforward=2048, dropout=0.5, has_pos=False):
        """ Self-attention-based diarization transformer (embedding) model.

        Args:
          n_speakers (int): Number of speakers in recording
          in_size (int): Dimension of input feature vector
          n_heads (int): Number of attention heads
          n_units (int): Number of units in a self-attention block
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(DiarizationTransformerEncoder, self).__init__()
        self.n_speakers = n_speakers
        self.in_size = in_size
        self.n_heads = n_heads
        self.n_units = n_units
        self.n_layers = n_layers
        self.has_pos = has_pos

        self.src_mask = None
        self.encoder = nn.Linear(in_size, n_units)
        self.encoder_norm = nn.LayerNorm(n_units)
        if self.has_pos:
            self.pos_encoder = PositionalEncoding(n_units, dropout)
        encoder_layers = TransformerEncoderLayer(n_units, n_heads, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)
        #self.decoder.bias.data.zero_()
        #self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, has_mask=False, activation=None):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != src.size(1):
                mask = self._generate_square_subsequent_mask(src.size(1)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = pad_sequence(src, padding_value=-1, batch_first=True)

        # src: (B, T, E)
        src = self.encoder(src)
        src = self.encoder_norm(src)
        # src: (T, B, E)
        src = src.transpose(0, 1)
        if self.has_pos:
            # src: (T, B, E)
            src = self.pos_encoder(src)
        # output: (T, B, E)
        output = self.transformer_encoder(src, self.src_mask)
        # output: (B, T, E)
        output = output.transpose(0, 1)

        return output

    def get_attention_weight(self, src):
        # NOTE: NOT IMPLEMENTED CORRECTLY!!!
        attn_weight = []

        def hook(module, input, output):
            # attn_output, attn_output_weights = multihead_attn(query, key, value)
            # output[1] are the attention weights
            attn_weight.append(output[1])

        handles = []
        for l in range(self.n_layers):
            handles.append(self.transformer_encoder.layers[l].self_attn.register_forward_hook(hook))

        self.eval()
        with torch.no_grad():
            self.forward(src)

        for handle in handles:
            handle.remove()
        self.train()

        return torch.stack(attn_weight)


class TransformerModel(nn.Module):
    def __init__(self, n_speakers, in_size, n_heads, n_units, n_layers,
                 dim_feedforward=2048, dropout=0.5, has_pos=False):
        """ Self-attention-based diarization model.

        Args:
          n_speakers (int): Number of speakers in recording
          in_size (int): Dimension of input feature vector
          n_heads (int): Number of attention heads
          n_units (int): Number of units in a self-attention block
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(TransformerModel, self).__init__()
        self.transformer_encoder = DiarizationTransformerEncoder(n_speakers, in_size, n_heads, n_units, n_layers,
                                                                 dim_feedforward=dim_feedforward,
                                                                 dropout=dropout,
                                                                 has_pos=has_pos)
        self.decoder = nn.Linear(n_units, n_speakers)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, has_mask=False, activation=None):
        ilens = [x.shape[0] for x in src]
        # output: (B, T, E)
        output = self.transformer_encoder(src, has_mask, activation)

        # output: (B, T, C)
        output = self.decoder(output)

        if activation:
            output = activation(output)

        output = [out[:ilen] for out, ilen in zip(output, ilens)]
        return output


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional information to each time step of x
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EncoderDecoderAttractor(nn.Module):
    # PyTorch port of:
    #  https://github.com/hitachi-speech/EEND/blob/95216f0025dde8f0358d71412be046a36a030b33/eend/chainer_backend/encoder_decoder_attractor.py#L11
    def __init__(self, n_units, encoder_dropout=0.1, decoder_dropout=0.1):
        super(EncoderDecoderAttractor, self).__init__()

        n_layers = 1
        self.encoder = nn.LSTM(input_size=n_units, hidden_size=n_units, num_layers=n_layers, dropout=encoder_dropout, batch_first=True)
        self.decoder = nn.LSTM(input_size=n_units, hidden_size=n_units, num_layers=n_layers, dropout=decoder_dropout, batch_first=True)
        self.counter = nn.Linear(n_units, 1)

        self.n_units = n_units

    def compute_attractors(self, xs, zeros):
        _, hidden = self.encoder(xs)
        attractors, _ = self.decoder(zeros, hidden)
        return attractors

    def forward(self, xs, n_speakers):
        # the +1 comes from Fig.1 in the EDA paper.
        # Note that an additional attractor is added, but label is set to 0
        #  to enable the algorithm
        zs = [torch.zeros((n_spk + 1, self.n_units), dtype=torch.float32, device=xs[0].device) for n_spk in n_speakers]
        zeros = pack_sequence(zs, enforce_sorted=False)
        attractors = self.compute_attractors(xs, zeros)
        
        # unpack the packed sequence
        attractors, attractor_lens = pad_packed_sequence(attractors, batch_first=True)
        attractor_logits = self.counter(attractors)  # NOTE: this may fail if not all same lens??
                                                     # in which case you may need to do something
                                                     # similar to what was done below w/ the for-loop
        attractor_logits = attractor_logits.squeeze()

        # The final attractor does not correspond to a speaker so remove it
        aa = [attractors[ii,0:n_speakers[ii],:] for ii in range(len(n_speakers))]

        return aa, attractor_logits


class TransformerEDAModel(nn.Module):
    def __init__(self, n_speakers, in_size, n_heads, n_units, n_layers,
                 dim_feedforward=2048,
                 xformer_dropout=0.1, attractor_encoder_dropout=0.1, attractor_decoder_dropout=0.1,
                 attractor_loss_ratio=1.0, has_pos=False):
        """ Self-attention-based diarization model, with attractor network to determine number of speakers
        # PyTorch port of: https://github.com/hitachi-speech/EEND/blob/95216f0025dde8f0358d71412be046a36a030b33/eend/chainer_backend/models.py#L409
        Args:
          n_speakers (int): Number of speakers in recording
          in_size (int): Dimension of input feature vector
          n_heads (int): Number of attention heads
          n_units (int): Number of units in a self-attention block
          n_layers (int): Number of transformer-encoder layers
          # TODO: update!! dropout (float): dropout ratio
        """
        super(TransformerEDAModel, self).__init__()

        self.transformer_encoder = DiarizationTransformerEncoder(n_speakers, in_size, n_heads, n_units, n_layers,
                                                                 dim_feedforward=dim_feedforward,
                                                                 dropout=xformer_dropout,
                                                                 has_pos=has_pos)
        self.eda = EncoderDecoderAttractor(
            n_units,
            encoder_dropout=attractor_encoder_dropout,
            decoder_dropout=attractor_decoder_dropout,
        )

        self.attractor_loss_ratio = attractor_loss_ratio

    def forward(self, src, ts, has_mask=False, activation=None):
        ilens = [x.shape[0] for x in src]
        # output: (B, T, E)
        emb = self.transformer_encoder(src, has_mask, activation)

        # run data through the eda network
        n_speakers = [t.shape[1] for t in ts]
        attractors, attractor_logits = self.eda(emb, n_speakers)

        # use the attractor & multiply w/ embedding
        ys = [torch.matmul(emb[ii,:ilens[ii]], attractors[ii].T) for ii in range(emb.size()[0])]
        # TODO: do I need to apply the sigmoid?  - I don't think so ...

        return ys, attractor_logits


if __name__ == "__main__":
    model = TransformerModel(5, 40, 4, 512, 2, dropout=0.1)
    input = torch.randn(8, 500, 40)
    print("Model output:", model(input).size())
    print("Model attention:", model.get_attention_weight(input).size())
    print("Model attention sum:", model.get_attention_weight(input)[0][0][0].sum())
