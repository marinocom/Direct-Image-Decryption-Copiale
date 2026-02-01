"""Miscellaneous collection of useful modules."""

import math
import torch
from torch import nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    """Generates the positional encoding for an input token."""

    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor) -> Tensor:
        """Apply positional encoding to a set of token embeddings.

        Parameters
        ----------
        token_embedding : Tensor
            A series of tokens of shape N x S x D, with N being the batch size, S being
            the sequence length and D the dimension of the embedding.

        Returns
        -------
        Tensor
            The input tokens with positional embedding and dropout applied to them.
        """
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class TokenEmbedding(nn.Module):
    """Embed input tokens."""

    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
