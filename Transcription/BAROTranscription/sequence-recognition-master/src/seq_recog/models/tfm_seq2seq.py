"""Transformer-based Encoder-Decoder models."""

from typing import List, Optional
from warnings import warn

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor, TensorType
from vit_pytorch.extractor import Extractor
from vit_pytorch import ViT

from .misc import TokenEmbedding, PositionalEncoding

from .model_zoo import ModelZoo
from .losses.focal_loss import SequenceFocalLoss
from .base_model import BaseModel, ModelOutput
from .base_model import BaseModelConfig
from ..data.base_dataset import BatchedSample, BaseDataConfig, BaseVocab


class TransformerSeq2SeqConfig(BaseModelConfig):
    """Settings for Transformer-Based Seq2Seq models."""

    loss_function: str = "cross-entropy"
    focal_loss_gamma: float = 1.0
    focal_loss_alpha: float = 0.25
    label_smoothing: float = 0.0
    loss_weights: Optional[List[float]] = None


class TransformerSeq2Seq(BaseModel):
    """Transformer-based Sequence to Sequence transcription models."""

    MODEL_CONFIG = TransformerSeq2SeqConfig

    def __init__(
        self, model_config: TransformerSeq2SeqConfig, data_config: BaseDataConfig
    ) -> None:
        """Initialise Model."""
        super().__init__(model_config, data_config)

        self.tgt_mask = nn.Parameter(
            self._get_tgt_mask(data_config.target_seqlen),
            requires_grad=False,
        )

        if model_config.loss_function == "focal":
            self.loss = SequenceFocalLoss(
                gamma=model_config.focal_loss_gamma,
                alpha=model_config.focal_loss_alpha,
                label_smoothing=model_config.label_smoothing,
                ignore_index=BaseVocab.PAD_INDEX,
                class_weights=torch.tensor(model_config.loss_weights)
                if model_config.loss_weights is not None
                else None,
            )
        else:
            self.loss = nn.CrossEntropyLoss(
                weight=torch.tensor(model_config.loss_weights)
                if model_config.loss_weights is not None
                else None,
                ignore_index=BaseVocab.PAD_INDEX,
                label_smoothing=model_config.label_smoothing,
            )

    def compute_batch(self, batch: BatchedSample, device: torch.device) -> ModelOutput:
        """Generate the model's output for a single input batch.

        Parameters
        ----------
        batch: BatchedSample
            A model input batch encapsulated in a BatchedSample named tuple.
        device: torch.device
            Device where the training is happening in order to move tensors.

        Returns
        -------
        output: torch.Tensor
            The output of the model for the input batch.
        """
        images = batch.img.to(device)
        transcript = batch.gt.to(device)

        output = self.forward(images, transcript)
        return ModelOutput(output=output)

    def compute_loss(
        self, batch: BatchedSample, output: ModelOutput, device: torch.device
    ) -> torch.float32:
        """Generate the model's loss for a single input batch and output.

        Parameters
        ----------
        batch: BatchedSample
            A model input batch encapsulated in a BatchedSample named tuple.
        output:ModelOutput
            The output of the model for the input batch encapsulated in a
            ModelOutput class.
        device: torch.device
            Device where the training is happening in order to move tensors.

        Returns
        -------
        torch.float32
            The model's loss for the given input.
        """
        output = output.output
        output = output.view(-1, output.shape[-1])
        transcript = batch.gt.to(device)[:, 1:].reshape(-1)

        return self.loss(output, transcript)

    @staticmethod
    def _get_tgt_mask(seqlen: int) -> torch.FloatTensor:
        mask = torch.ones(seqlen - 1, seqlen - 1)
        mask = (torch.triu(mask) == 1).transpose(0, 1).float()
        mask = mask.masked_fill(mask == 0, float("-inf"))
        mask = mask.masked_fill(mask == 1, float(0.0))

        return mask


class ViTSeq2SeqTransformerConfig(TransformerSeq2SeqConfig):
    """Settings for Transformer-Based Seq2Seq models."""

    freeze_backbone: bool
    output_units: int
    model_dim: int
    patch_size: int
    enc_layers: int
    enc_heads: int
    mlp_dim: int
    dropout: float
    emb_dropout: float
    pretrained_backbone: Optional[str]
    dec_heads: int
    norm_first: bool
    activation: str
    dec_layers: int


@ModelZoo.register_model
class ViTSeq2SeqTransformer(TransformerSeq2Seq):
    """Implements a full seq2seq transformer using vit_pytorch."""

    MODEL_CONFIG = ViTSeq2SeqTransformerConfig

    ACTIVATIONS = {
        "relu": F.relu,
        "gelu": F.gelu,
    }

    def __init__(
        self, model_config: ViTSeq2SeqTransformerConfig, data_config: BaseDataConfig
    ) -> None:
        """Initialise Model."""
        super().__init__(model_config, data_config)

        self.freeze_backbone = model_config.freeze_backbone

        self.embedder = TokenEmbedding(
            model_config.output_units,
            model_config.model_dim,
        )
        vit = ViT(
            image_size=(data_config.target_shape[1], data_config.target_shape[0]),
            patch_size=model_config.patch_size,
            num_classes=model_config.output_units,
            dim=model_config.model_dim,
            depth=model_config.enc_layers,
            heads=model_config.enc_heads,
            mlp_dim=model_config.mlp_dim,
            dropout=model_config.dropout,
            emb_dropout=model_config.emb_dropout,
        )

        if model_config.pretrained_backbone is not None:
            weights = torch.load(model_config.pretrained_backbone)
            del weights["mlp_head.1.weight"]
            del weights["mlp_head.1.bias"]
            missing, unexpected = vit.load_state_dict(weights, strict=False)

            if missing or unexpected:
                warn(
                    f"There are missing or unexpected weights in the loaded"
                    f"encoder model: \n{'='*50}\nMissing: {missing}\n{'='*50}\n"
                    f"Unexpected:{unexpected}"
                )

        self.encoder = Extractor(vit)

        if self.freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.pos_encoding = PositionalEncoding(
            model_config.model_dim, model_config.emb_dropout
        )
        self.max_inference_len = data_config.target_seqlen
        self.vocab_size = model_config.output_units
        self.norm = nn.LayerNorm(model_config.model_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_config.model_dim,
            nhead=model_config.dec_heads,
            dim_feedforward=model_config.model_dim,
            dropout=model_config.dropout,
            batch_first=True,
            norm_first=model_config.norm_first,
            activation=self.ACTIVATIONS[model_config.activation],
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=model_config.dec_layers, norm=self.norm
        )
        self.linear = nn.Linear(model_config.model_dim, model_config.output_units)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y) -> torch.FloatTensor:
        _, x = self.encoder(x)  # Batch, Tok + 1, Dim

        if self.training:
            yemb = self.embedder(y[:, :-1])  # Batch, Seqlen, Dim
            yemb = self.pos_encoding(yemb)  # Batch, Seqlen, Dim

            # Embed the target sequence and positionally encode it
            x = self.decoder(
                yemb,
                x,
                tgt_mask=self.tgt_mask,
                tgt_key_padding_mask=y[:, 1:] == BaseVocab.PAD_INDEX,
            )
            x = self.linear(x)  # Batch, Seqlen, Class
            # x = self.softmax(x)
        else:
            newy = torch.zeros(y.shape[0], y.shape[1] + 1, *y.shape[2:]).to(y.device)
            newy[:, 0] = y[:, 0]
            y = newy

            outputs = torch.zeros(
                y.shape[0], self.max_inference_len - 1, self.vocab_size
            ).to(y.device)
            for ii in range(1, self.max_inference_len):
                y_t = y[:, :ii]
                y_t = self.embedder(y_t)
                y_t = self.pos_encoding(y_t)
                out = self.decoder(y_t, x, tgt_mask=self.tgt_mask[:ii, :ii])
                out = self.linear(out)

                outputs[:, ii - 1, :] = out[:, ii - 1, :]
                y[:, ii] = torch.argmax(out, -1)[:, ii - 1]

            x = outputs
        return x
