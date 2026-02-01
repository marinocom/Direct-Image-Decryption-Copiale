"""Module containing all models that implement a CTC Loss."""

import torch
from torch import nn

import numpy as np
from vit_pytorch import ViT
from vit_pytorch.extractor import Extractor

from .base_model import BaseModel as BaseInferenceModel, BaseModelConfig, ModelOutput
from .misc import PositionalEncoding
from .model_zoo import ModelZoo
from .cnns import (
    create_resnet,
    create_vgg,
    BaroCNN,
    LightweightCNN,
    RESNET_EMBEDDING_SIZES,
    VGG_EMBEDDING_SIZE,
)
from ..data.base_dataset import BaseDataConfig, BatchedSample


class CTCModelConfig(BaseModelConfig):
    """Shared configuration for CTC Models."""

    ...


class CTCModel(BaseInferenceModel):
    """Model that uses a CTC loss."""

    MODEL_CONFIG = CTCModelConfig

    def __init__(
        self, model_config: CTCModelConfig, data_config: BaseDataConfig
    ) -> None:
        """Construct a model with a CTC Loss."""
        super().__init__(model_config, data_config)
        self.loss = nn.CTCLoss()

    def compute_batch(self, batch: BatchedSample, device: torch.device) -> ModelOutput:
        """Generate the model's output for a single input batch.

        Parameters
        ----------
        batch: BatchedSample
            A model input batch encapsulated in a BatchedSample object.
        device: torch.device
            What device the model is being trained on.
        Returns
        -------
        output: ModelOutput
            The output of the model for the input batch.
        """
        output = self(batch.img.to(device))

        return ModelOutput(output=output)

    def compute_loss(
        self, batch: BatchedSample, output: ModelOutput, device: torch.device
    ) -> torch.float32:
        """Generate the model's loss for a single input batch and output.

        Parameters
        ----------
        batch: BatchedSample
            A model input batch encapsulated in a BatchedSample named tuple.
        output: ModelOutput
            The output of the model for the input batch.
        device: torch.device
            Device where the model is being trained in.

        Returns
        -------
        torch.float32
            The model's loss for the given input.
        """
        output = output.output
        columns = output.shape[0]
        target_shape = batch.img[0].shape[-1]
        batch_lens = batch.gt_len.numpy()
        input_lengths = batch.curr_shape[0] * (columns / target_shape)
        input_lengths = np.ceil(input_lengths.numpy()).astype(int)
        input_lengths = np.where(input_lengths < batch_lens, batch_lens, input_lengths)
        input_lengths = input_lengths.tolist()

        batch_loss = self.loss(
            output,
            batch.gt.to(device),
            input_lengths,
            tuple(batch.gt_len.tolist()),
        )
        return batch_loss


class FullyConvCTCConfig(CTCModelConfig):
    """Configuration for a Fully Convolutional CTC Model."""

    width_upsampling: int
    kern_upsampling: int
    intermediate_units: int
    output_units: int
    resnet_type: int
    pretrained: bool = True


@ModelZoo.register_model
class FullyConvCTC(CTCModel):
    """A fully convolutional CTC model with convolutional upsampling."""

    MODEL_CONFIG = FullyConvCTCConfig

    def __init__(
        self,
        model_config: FullyConvCTCConfig,
        data_config: BaseDataConfig,
    ) -> None:
        """Initialise FullyConv model from parameters.

        Parameters
        ----------
        model_config: FullyConvCTCConfig
            Configuration object for the model.
        data_config: BaseDataConfig
            Configuration for input data formatting.
        """
        super().__init__(model_config, data_config)

        self._model_config = model_config
        self._data_config = data_config
        self._backbone = create_resnet(
            self._model_config.resnet_type, self._model_config.pretrained
        )
        self._pooling = nn.AdaptiveAvgPool2d((1, None))

        self._upsample = nn.ConvTranspose2d(
            in_channels=RESNET_EMBEDDING_SIZES[self._model_config.resnet_type],
            out_channels=self._model_config.intermediate_units,
            kernel_size=(1, self._model_config.kern_upsampling),
            stride=(1, self._model_config.width_upsampling),
        )
        self._activation = nn.ReLU()
        self._output = nn.Conv2d(
            kernel_size=1,
            in_channels=self._model_config.intermediate_units,
            out_channels=self._model_config.output_units,
        )
        self._softmax = nn.LogSoftmax(dim=-1)

        if model_config.model_weights:
            self.load_weights(model_config.model_weights)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Compute the transcription of the input batch images x.

        Parameters
        ----------
        x: torch.Tensor
            Batch of input tensor images of shape N x 3 x H x W where N is the
            Batch size, 3 is the number of channels and H, W are the height and
            width of the input.

        Returns
        -------
        torch.Tensor
            A W' x N x C matrix containing the log likelihood of every class at
            every time step where W' is the width of the output sequence,
            N is the batch size and C is the number of output classes. This
            matrix may be used in a CTC loss.
        """
        # x: N x 3   x H       x W
        x = self._backbone(x)  # N x 512 x H // 32 x W // 32
        x = self._pooling(x)  # N x 512 x 1       x W // 32
        x = self._upsample(x)  # N x INT x 1       x ~(W // 32 - 1) * K
        x = self._activation(x)
        x = self._output(x)  # N x CLS x~(W // 32 - 1) * K
        x = x.squeeze(2)  # N x INT x ~(W // 32 - 1) * K
        x = x.permute((2, 0, 1))  # ~(W // 32 - 1) * K x N x  CLS
        y = self._softmax(x)

        return y


class BaroCRNNConfig(CTCModelConfig):
    """Configuration for the Baró CTC Model."""

    lstm_hidden_size: int
    lstm_layers: int
    blstm: bool
    dropout: float
    output_classes: int


@ModelZoo.register_model
class BaroCRNN(CTCModel):
    """CRNN Model based on Arnau Baró's CTC OMR model."""

    MODEL_CONFIG = BaroCRNNConfig

    def __init__(
        self, model_config: BaroCRNNConfig, data_config: BaseDataConfig
    ) -> None:
        """Initialise Baró CRNN from parameters.

        Parameters
        ----------
        model_config: BaroCRNNConfig
            Configuration object for the model.
        data_config: BaseDataConfig
            Configuration for input data formatting.
        """
        super().__init__(model_config, data_config)

        self.model_config = model_config
        self.data_config = data_config

        self.directions = 2 if self.model_config.blstm else 1

        self.backbone = BaroCNN(self.model_config.dropout)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=self.model_config.lstm_hidden_size,
            num_layers=self.model_config.lstm_layers,
            dropout=self.model_config.dropout,
            bidirectional=self.model_config.blstm,
        )
        self.linear = nn.Linear(
            self.model_config.lstm_hidden_size, self.model_config.output_classes
        )
        self.log_softmax = nn.LogSoftmax(-1)

        if model_config.model_weights:
            self.load_weights(model_config.model_weights)

    def forward(self, x) -> torch.FloatTensor:
        """Compute the transcription of the input batch images x.

        Parameters
        ----------
        x: torch.Tensor
            Batch of input tensor images of shape N x 3 x H x W where N is the
            Batch size, 3 is the number of channels and H, W are the height and
            width of the input.

        Returns
        -------
        torch.Tensor
            A W' x N x C matrix containing the log likelihood of every class at
            every time step where W' is the width of the output sequence,
            N is the batch size and C is the number of output classes. This
            matrix may be used in a CTC loss.
        """
        x = self.backbone(x)
        x = x.permute(2, 0, 1)  # Length, Batch, Hidden
        x, _ = self.lstm(x)  # Length, Batch, Hidden * Directions

        if self.directions > 1:
            seq_len, batch_size, hidden_size = x.shape

            x = x.view(
                seq_len, batch_size, self.directions, hidden_size // self.directions
            )
            x = x.sum(axis=2)

        x = self.linear(x)  # Length, Batch, Classes
        x = self.log_softmax(x)

        return x


class ResnetCRNNConfig(CTCModelConfig):
    """Configuration for the Baró CTC Model."""

    resnet_type: int
    lstm_layers: int
    lstm_hidden_size: int
    upsampling_kern: int
    upsampling_stride: int
    blstm: bool
    dropout: float
    output_classes: int


@ModelZoo.register_model
class ResnetCRNN(CTCModel):
    """CRNN Model with a ResNet as backcbone."""

    MODEL_CONFIG = ResnetCRNNConfig

    def __init__(
        self, model_config: ResnetCRNNConfig, data_config: BaseDataConfig
    ) -> None:
        """Initialise Baró CRNN from parameters.

        Parameters
        ----------
        model_config: ResnetCRNNConfig
            Configuration object for the model.
        data_config: BaseDataConfig
            Configuration for input data formatting.
        """
        super().__init__(model_config, data_config)

        self.model_config = model_config
        self.data_config = data_config

        self.directions = 2 if self.model_config.blstm else 1
        self.hidden_size = RESNET_EMBEDDING_SIZES[self.model_config.resnet_type]

        self.backbone = create_resnet(self.model_config.resnet_type, headless=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))
        self.upsample = nn.ConvTranspose2d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=(1, self.model_config.upsampling_kern),
            stride=(1, self.model_config.upsampling_stride),
        )
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.model_config.lstm_hidden_size,
            num_layers=self.model_config.lstm_layers,
            bidirectional=self.model_config.blstm,
            dropout=self.model_config.dropout,
        )
        self.output_layer = nn.Linear(
            self.model_config.lstm_hidden_size, self.model_config.output_classes
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)

        if model_config.model_weights:
            self.load_weights(model_config.model_weights)

    def forward(self, x) -> torch.Tensor:
        """Compute the transcription of the input batch images x.

        Parameters
        ----------
        x: torch.Tensor
            Batch of input tensor images of shape N x 3 x H x W where N is the
            Batch size, 3 is the number of channels and H, W are the height and
            width of the input.

        Returns
        -------
        torch.Tensor
            A W' x N x C matrix containing the log likelihood of every class at
            every time step where W' is the width of the output sequence,
            N is the batch size and C is the number of output classes. This
            matrix may be used in a CTC loss.
        """
        x = self.backbone(x)  # Batch, Channels, Height, Width // 16
        x = self.avg_pool(x)  # Batch, Channels, 1, Width // 16
        x = self.upsample(x)  # Batch, Channels, 1, Length
        x = x.squeeze(2)  # Batch, Channels, Length
        x = x.permute(2, 0, 1)  # Length, Batch, Hidden
        x, _ = self.lstm(x)  # Length, Batch, Hidden * Directions

        if self.directions > 1:
            seq_len, batch_size, hidden_size = x.shape
            x = x.view(
                seq_len, batch_size, self.directions, hidden_size // self.directions
            )
            x = x.sum(axis=2)

        x = self.output_layer(x)  # Length, Batch, Classes
        x = self.log_softmax(x)

        return x


class VggCRNNConfig(CTCModelConfig):
    """Configuration for the Baró CTC Model."""

    vgg_type: int
    vgg_batchnorm: bool
    vgg_pretrained: bool
    lstm_layers: int
    lstm_hidden_size: int
    blstm: bool
    dropout: float
    output_classes: int


@ModelZoo.register_model
class VggCRNN(CTCModel):
    """CRNN Model with a ResNet as backcbone."""

    MODEL_CONFIG = VggCRNNConfig

    def __init__(
        self, model_config: VggCRNNConfig, data_config: BaseDataConfig
    ) -> None:
        """Initialise Baró CRNN from parameters.

        Parameters
        ----------
        model_config: ResnetCRNNConfig
            Configuration object for the model.
        data_config: BaseDataConfig
            Configuration for input data formatting.
        """
        super().__init__(model_config, data_config)

        self.model_config = model_config
        self.data_config = data_config

        self.directions = 2 if self.model_config.blstm else 1
        self.hidden_size = VGG_EMBEDDING_SIZE

        self.backbone = create_vgg(
            vgg_type=self.model_config.vgg_type,
            batchnorm=self.model_config.vgg_batchnorm,
            headless=True,
            pretrained=self.model_config.vgg_pretrained,
            keep_maxpool=False,
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))

        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.model_config.lstm_hidden_size,
            num_layers=self.model_config.lstm_layers,
            bidirectional=self.model_config.blstm,
            dropout=self.model_config.dropout,
        )
        self.output_layer = nn.Linear(
            self.model_config.lstm_hidden_size, self.model_config.output_classes
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)

        if model_config.model_weights:
            self.load_weights(model_config.model_weights)

    def forward(self, x) -> torch.Tensor:
        """Compute the transcription of the input batch images x.

        Parameters
        ----------
        x: torch.Tensor
            Batch of input tensor images of shape N x 3 x H x W where N is the
            Batch size, 3 is the number of channels and H, W are the height and
            width of the input.

        Returns
        -------
        torch.Tensor
            A W' x N x C matrix containing the log likelihood of every class at
            every time step where W' is the width of the output sequence,
            N is the batch size and C is the number of output classes. This
            matrix may be used in a CTC loss.
        """
        x = self.backbone(x)  # Batch, Channels, Height, Width // 16
        x = self.avg_pool(x)  # Batch, Channels, 1, Width // 16
        x = x.squeeze(2)  # Batch, Channels, Length
        x = x.permute(2, 0, 1)  # Length, Batch, Hidden
        x, _ = self.lstm(x)  # Length, Batch, Hidden * Directions

        if self.directions > 1:
            seq_len, batch_size, hidden_size = x.shape
            x = x.view(
                seq_len, batch_size, self.directions, hidden_size // self.directions
            )
            x = x.sum(axis=2)

        x = self.output_layer(x)  # Length, Batch, Classes
        x = self.log_softmax(x)

        return x


class CTCCNNTransformerConfig(CTCModelConfig):
    """Configuration for the CTC CNN Transformer model."""

    nheads: int
    d_encoder: int
    d_ffw: int
    encoder_layers: int
    dropout: float
    activation: str
    output_classes: int


@ModelZoo.register_model
class CTCCNNTransformer(CTCModel):
    """CTC model with a transformer encoder and a CNN as feature extractor.

    CTC model that uses a CNN as a feature extractor and processes it through a
    transformer encoder. This architecture is implemented by analogy to the contents of
    paper [1].

    [1] A. Ríos-Vila, J. M. Iñesta, and J. Calvo-Zaragoza, “On the Use of Transformers
    for End-to-End Optical Music Recognition,” in Pattern Recognition and Image
    Analysis, A. J. Pinho, P. Georgieva, L. F. Teixeira, and J. A. Sánchez, Eds., in
    Lecture Notes in Computer Science. Cham: Springer International Publishing, 2022,
    pp. 470-481. doi: 10.1007/978-3-031-04881-4_37.

    """

    MODEL_CONFIG = CTCCNNTransformerConfig

    def __init__(
        self, model_config: CTCCNNTransformerConfig, data_config: BaseDataConfig
    ) -> None:
        """Initialise CTC CNN Transformer from parameters.

        Parameters
        ----------
        model_config: CTCCNNTransformerConfig
            Configuration object for the model.
        data_config: BaseDataConfig
            Configuration for input data formatting.
        """
        super().__init__(model_config, data_config)

        self.model_config = model_config
        self.data_config = data_config

        self.backbone = LightweightCNN()
        self.mapping = nn.Linear(
            data_config.target_shape[1] * 8,
            self.model_config.d_encoder,
        )
        self.positional_encoding = PositionalEncoding(
            emb_size=self.model_config.d_encoder,
            dropout=model_config.dropout,
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_config.d_encoder,
            nhead=self.model_config.nheads,
            dim_feedforward=self.model_config.d_ffw,
            dropout=self.model_config.dropout,
            activation=self.model_config.activation,
        )
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer,
            self.model_config.encoder_layers,
        )
        self.output = nn.Linear(
            self.model_config.d_encoder,
            self.model_config.output_classes,
        )
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x) -> torch.Tensor:
        """Compute the transcription of the input batch images x.

        Parameters
        ----------
        x: torch.Tensor
            Batch of input tensor images of shape N x 3 x H x W where N is the
            Batch size, 3 is the number of channels and H, W are the height and
            width of the input.

        Returns
        -------
        torch.Tensor
            A W' x N x C matrix containing the log likelihood of every class at
            every time step where W' is the width of the output sequence,
            N is the batch size and C is the number of output classes. This
            matrix may be used in a CTC loss.
        """
        batch_size, _, height, width = x.shape
        x = self.backbone(x)
        x = self.mapping(x)  # N x W x F
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = self.output(x)  # N x W x C
        x = x.permute((1, 0, 2))  # W x N x C

        x = self.logsoftmax(x)

        return x


class CTCVITModelConfig(CTCModelConfig):
    """Configuration for the CTC CNN Transformer model."""

    patch_size: int
    model_dim: int
    enc_layers: int
    enc_heads: int
    mlp_dim: int
    dropout: float
    emb_dropout: float
    out_classes: int


@ModelZoo.register_model
class CTCVITModel(CTCModel):
    """CTC model with a ViT as a feature extractor and encoder.

    Model implemented using the design in [1].

    [1] A. Ríos-Vila, J. M. Iñesta, and J. Calvo-Zaragoza, “On the Use of Transformers
    for End-to-End Optical Music Recognition,” in Pattern Recognition and Image
    Analysis, A. J. Pinho, P. Georgieva, L. F. Teixeira, and J. A. Sánchez, Eds., in
    Lecture Notes in Computer Science. Cham: Springer International Publishing, 2022,
    pp. 470-481. doi: 10.1007/978-3-031-04881-4_37.

    """

    MODEL_CONFIG = CTCVITModelConfig

    def __init__(
        self, model_config: CTCVITModelConfig, data_config: BaseDataConfig
    ) -> None:
        """Initialise Baró CRNN from parameters.

        Parameters
        ----------
        model_config: CTCVITModelConfig
            Configuration object for the model.
        data_config: BaseDataConfig
            Configuration for input data formatting.
        """
        super().__init__(model_config, data_config)

        self.model_config = model_config
        self.data_config = data_config

        vit = ViT(
            image_size=(
                self.data_config.target_shape[1],
                self.data_config.target_shape[0],
            ),
            patch_size=self.model_config.patch_size,
            num_classes=self.model_config.out_classes,
            dim=self.model_config.model_dim,
            depth=self.model_config.enc_layers,
            heads=self.model_config.enc_heads,
            mlp_dim=self.model_config.mlp_dim,
            dropout=self.model_config.dropout,
            emb_dropout=self.model_config.emb_dropout,
        )
        self.backbone = Extractor(vit)
        self.output = nn.Linear(
            self.model_config.model_dim,
            self.model_config.out_classes,
        )
        self.log_softmax = nn.LogSoftmax(-1)

    def forward(self, x) -> torch.Tensor:
        """Compute the transcription of the input batch images x.

        Parameters
        ----------
        x: torch.Tensor
            Batch of input tensor images of shape N x 3 x H x W where N is the
            Batch size, 3 is the number of channels and H, W are the height and
            width of the input.

        Returns
        -------
        torch.Tensor
            A W' x N x C matrix containing the log likelihood of every class at
            every time step where W' is the width of the output sequence,
            N is the batch size and C is the number of output classes. This
            matrix may be used in a CTC loss.
        """
        _, x = self.backbone(x)  # N x (ntoks + 1) x F
        x = self.output(x)
        x = x.permute((1, 0, 2))
        x = self.log_softmax(x)

        return x
