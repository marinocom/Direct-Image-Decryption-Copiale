"""RNN-based encoder modules."""

import numpy as np
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .cnns import create_vgg, VGG_EMBEDDING_SIZE


class VggRNNEncoder(nn.Module):
    """RNN-based encoder with a VGG Backbone."""

    def __init__(
        self,
        vgg_type: int,
        vgg_bn: bool,
        vgg_pretrain: bool,
        hidden_size: int,
        layers: int,
        height: int,
        width: int,
        dropout: float,
        sum_hidden: bool = False,
    ):
        super(VggRNNEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = layers
        self.height = height
        self.width = width
        self.dropout = dropout
        self.sum_hidden = sum_hidden

        self.backbone = create_vgg(
            vgg_type=vgg_type,
            batchnorm=vgg_bn,
            headless=True,
            pretrained=vgg_pretrain,
            keep_maxpool=False,
        )
        self.backbone_dropout = nn.Dropout2d(p=dropout)

        self.rnn = nn.GRU(
            (self.height // 16) * VGG_EMBEDDING_SIZE,
            self.hidden_size,
            self.n_layers,
            dropout=self.dropout,
            bidirectional=True,
        )

    @staticmethod
    def _sum_directions(x: torch.FloatTensor) -> torch.FloatTensor:
        return x[:, :, : x.shape[-1] // 2] + x[:, :, x.shape[-1] // 2 :]

    def forward(
        self,
        src: torch.FloatTensor,
        src_len: torch.LongTensor,
        hidden: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Extract features from a batch of images and pass them through an RNN.

        Parameters
        ----------
        src : TensorType
            A batch of images of shape N x C x H x W, where N is the batch size, C is
            the number of channels, H is the height and W is the width of the images.
            Number of channels is assumed to be 3.
        src_len : TensorType
            Length in intermediate columns of the input image. This is used to account
            for padding.
        hidden : Optional[TensorType]
            An initial hidden state that may be provided to condition the encoding of
            the input images. Can either be a S x N x H Tensor, where S is the sequence
            length, N is the batch size and H is the hidden dimension size, or None, in
            which case a full-zero initial context is used. By default None.

        Returns
        -------
        Tuple[TensorType, TensorType]
            The final
        """
        batch_size = src.shape[0]
        out = self.backbone(src)

        # (batch, channels, height, width)
        out = self.backbone_dropout(out)
        out = out.permute(3, 0, 2, 1)
        # (width, batch, height, channels)

        out = out.contiguous()
        out = out.view(-1, batch_size, self.height // 16 * 512)
        # (width, batch, channels * height)

        out = pack_padded_sequence(
            out,
            src_len,
            batch_first=False,
            enforce_sorted=False,
        )
        output, hidden = self.rnn(out, hidden)
        output, output_len = pad_packed_sequence(
            output,
            batch_first=False,
            total_length=self.width // 16,
        )
        # Output: (width, batch, 2 * hidden)
        # Hidden: (2 * n_layers, batch, hidden)

        output = self._sum_directions(output)
        # (width, batch, hidden)

        if self.sum_hidden:
            final_hidden = hidden[1::2] + hidden[0::2]
        else:
            final_hidden = hidden[1::2]
        return output, final_hidden
