"""RNN-based decoder modules."""

from typing import Optional, Tuple

from torch import nn
from torch import TensorType
import torch


class RNNDecoder(nn.Module):
    """Implements an RNN-based decoder."""

    def __init__(
        self,
        hidden_size: int,
        embedding_size: int,
        vocab_size: int,
        nlayers: int,
        attention: nn.Module,
        dropout: float = 0.5,
        tradeoff_context_embed: Optional[float] = None,
        multinomial: bool = False,
    ) -> None:
        """Construct Decoder.

        Parameters
        ----------
        hidden_size : int
            Dimensionality of the hidden states of the gated recurrent unit stack.
        embedding_size : int
            Size of the token embeddings.
        vocab_size : int
            Size of the model's vocabulary.
        nlayers : int
            Number of layers in the gated recurrent unit stack.
        attention : nn.Module
            Attention block module.
        dropout : float, optional
            Amount of dropout to apply on the gated recurrent units, by default 0.5.
        tradeoff_context_embed : Optional[float], optional
            By how much to reduce the context embedding dimension, by default None
        multinomial : bool, optional
            Whether to change the input character by a multinomial distribution sampling
            version of it, by default False.
        """
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embedding_size
        self.nlayers = nlayers
        self.tradeoff = tradeoff_context_embed
        self.multinomial = multinomial
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, self.embed_size)
        self.attention = attention
        if self.tradeoff is not None:
            tradeoff_size = int(self.embed_size * self.tradeoff)
            self.context_shrink = nn.Linear(self.hidden_size, tradeoff_size)
            self.gru = nn.GRU(
                tradeoff_size + self.embed_size,
                self.hidden_size,
                self.nlayers,
                dropout=self.dropout,
            )
        else:
            self.gru = nn.GRU(
                self.embed_size + self.hidden_size,
                self.hidden_size,
                self.nlayers,
                dropout=self.dropout,
            )
        self.out = nn.Linear(self.hidden_size, vocab_size)

    def forward(
        self,
        in_char: torch.FloatTensor,
        hidden: torch.FloatTensor,
        encoder_output: torch.FloatTensor,
        attn_proj: torch.FloatTensor,
        src_len: torch.LongTensor,
        prev_attn: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute a single decoding step.

        Parameters
        ----------
        in_char : TensorType
            A one-hot encoded representation of the input character.
        hidden : TensorType
            The previous hidden state of the Decoder.
        encoder_output : TensorType
            Output of the encoder of the model. It is a S x N x H tensor, where N is the
            batch size, S is the sequence length and H is the hidden dimension size.
        attn_proj : TensorType
            The layer-projected version of the output of the encoder. It is a N x S x H
            tensor, where N is the batch size, S is the sequence length and H is the
            hidden dimension size.
        src_len : TensorType
            Length in intermediate columns of the input image. This is used to account
            for padding.
        prev_attn : TensorType
            The attention weights of the previous iteration. It is a N x S tensor, where
            N is the batch size and S is the sequence length.

        Returns
        -------
        Tuple[TensorType, TensorType, TensorType]
            The output of the model at timestep t, the last hidden state and the
            attention weights. They are N x V, N x H and N x S tensors respectively,
            where N is the batch size, V is the vocab size, H is the hidden size and S
            is the maximum sequence length. The output contains the logit probabilities
            of each class of the model.
        """
        attn_weights = self.attention(hidden, attn_proj, src_len, prev_attn)
        attn_weights = attn_weights.unsqueeze(-1)
        # (batch, seqlen, 1)

        encoder_output = encoder_output.permute(1, 2, 0)
        # (batch, hidden, seqlen)
        context = torch.bmm(encoder_output, attn_weights)
        # (batch, hidden, 1)
        context = context.squeeze(2)
        # (batch, hidden)

        if self.tradeoff is not None:
            context = self.context_shrink(context)

        if self.multinomial and self.training:
            top1 = torch.multinomial(in_char, 1)
        else:
            top1 = torch.argmax(in_char, dim=1)
        embed_char = self.embedding(top1)
        # (batch, embedding)

        in_dec = torch.cat((embed_char, context), 1)
        # (batch, hidden + embedding)
        in_dec = in_dec.unsqueeze(0)
        # (1, batch, hidden + embedding) -> For sequence length
        output, latest_hidden = self.gru(in_dec, hidden.contiguous())
        # Output: (1, batch, hidden)
        # Hidden: (layers, hidden)
        output = output.squeeze(0)
        # (batch, hidden)
        output = self.out(output)
        # (batch, vocab)

        return (
            output,
            latest_hidden,
            attn_weights.squeeze(2),
        )


class RNN2HeadDecoder(nn.Module):
    """Implements an RNN-based decoder."""

    def __init__(
        self,
        hidden_size: int,
        prm_embedding_size: int,
        sec_embedding_size: int,
        prm_vocab_size: int,
        sec_vocab_size: int,
        nlayers: int,
        attention: nn.Module,
        dropout: float = 0.5,
        feed_secondary: bool = True,
    ) -> None:
        """Construct Decoder.

        Parameters
        ----------
        hidden_size : int
            Dimensionality of the hidden states of the gated recurrent unit stack.
        prm_embedding_size : int
            Size of the primary token embeddings.
        sec_embedding_size : int
            Size of the secondary token embeddings.
        prm_vocab_size: int
            Size of the model's primary vocabulary.
        sec_vocab_size: int
            Size of the model's secondary vocabulary.
        nlayers : int
            Number of layers in the gated recurrent unit stack.
        attention : nn.Module
            Attention block module.
        dropout : float, optional
            Amount of dropout to apply on the gated recurrent units, by default 0.5.
        feed_secondary: bool
            Whether to feed the secondary output back into the autoregressive decoder.
        """
        super().__init__()
        self.feed_secondary = feed_secondary

        self.prm_embedding = nn.Embedding(prm_vocab_size, prm_embedding_size)
        self.sec_embedding = nn.Embedding(sec_vocab_size, sec_embedding_size)
        self.attention = attention
        if not feed_secondary:
            self.gru = nn.GRU(
                prm_embedding_size + hidden_size,
                hidden_size,
                nlayers,
                dropout=dropout,
            )
        else:
            self.gru = nn.GRU(
                sec_embedding_size + prm_embedding_size + hidden_size,
                hidden_size,
                nlayers,
                dropout=dropout,
            )
        self.activ = nn.ReLU()
        self.prm_out = nn.Linear(hidden_size, prm_vocab_size)
        self.sec_out = nn.Linear(hidden_size, sec_vocab_size)

    def forward(
        self,
        in_primary: torch.FloatTensor,
        in_secondary: torch.FloatTensor,
        hidden: torch.FloatTensor,
        encoder_output: torch.FloatTensor,
        attn_proj: torch.FloatTensor,
        src_len: torch.LongTensor,
        prev_attn: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute a single decoding step.

        Parameters
        ----------
        in_primary : torch.FloatTensor
            A one-hot encoded representation of the primary input character.
        in_secondary : torch.FloatTensor
            A one-hot encoded representation of the secondary input character.
        hidden : torch.FloatTensor
            The previous hidden state of the Decoder.
        encoder_output : torch.FloatTensor
            Output of the encoder of the model. It is a S x N x H tensor, where N is the
            batch size, S is the sequence length and H is the hidden dimension size.
        attn_proj : torch.FloatTensor
            The layer-projected version of the output of the encoder. It is a N x S x H
            tensor, where N is the batch size, S is the sequence length and H is the
            hidden dimension size.
        src_len : torch.LongTensor
            Length in intermediate columns of the input image. This is used to account
            for padding.
        prev_attn : torch.FloatTensor
            The attention weights of the previous iteration. It is a N x S tensor, where
            N is the batch size and S is the sequence length.

        Returns
        -------
        Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
            The output of the model at timestep t, the last hidden state and the
            attention weights. They are N x V, N x H and N x S tensors respectively,
            where N is the batch size, V is the vocab size, H is the hidden size and S
            is the maximum sequence length. The output contains the logit probabilities
            of each class of the model.
        """
        attn_weights = self.attention(hidden, attn_proj, src_len, prev_attn)
        attn_weights = attn_weights.unsqueeze(-1)
        # (batch, seqlen, 1)

        encoder_output = encoder_output.permute(1, 2, 0)
        # (batch, hidden, seqlen)
        context = torch.bmm(encoder_output, attn_weights)
        # (batch, hidden, 1)
        context = context.squeeze(2)
        # (batch, hidden)

        in_primary = torch.argmax(in_primary, dim=1)
        in_primary = self.prm_embedding(in_primary)
        # (batch, embedding)

        if self.feed_secondary:
            in_secondary = torch.argmax(in_secondary, dim=1)
            in_secondary = self.sec_embedding(in_secondary)
            # (batch, embedding)

            in_dec = torch.cat((in_primary, in_secondary, context), 1)
            # (batch, hidden + embedding1 + embedding2)

        else:
            in_dec = torch.cat((in_primary, context), 1)
            # (batch, hidden + embedding1)

        in_dec = in_dec.unsqueeze(0)
        # (1, batch, hidden + embedding) -> For sequence length
        features, latest_hidden = self.gru(in_dec, hidden.contiguous())
        # Output: (1, batch, hidden)
        # Hidden: (layers, hidden)
        features = features.squeeze(0)
        features = self.activ(features)
        # (batch, hidden)
        prm_logits = self.prm_out(features)
        sec_logits = self.sec_out(features)
        # (batch, vocab)

        return (
            prm_logits,
            sec_logits,
            latest_hidden,
            attn_weights.squeeze(2),
        )
