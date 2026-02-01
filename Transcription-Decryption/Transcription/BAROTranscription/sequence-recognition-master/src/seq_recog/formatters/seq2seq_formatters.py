"""Implementation of conversions from a Seq2Seq output model to anything else."""

from typing import Dict, List

import numpy as np
from numpy.typing import ArrayLike
import torch

from .base_formatter import BaseFormatter
from ..data.base_dataset import BatchedSample
from ..models.base_model import ModelOutput


class GreedyTextDecoder(BaseFormatter):
    """Generate an unpadded token sequence from a Seq2Seq output."""

    KEY_TEXT = "text"
    KEY_TEXT_CONF = "text_confidences"
    KEYS = [KEY_TEXT, KEY_TEXT_CONF]

    def __init__(self, confidences: bool = False) -> None:
        """Construct GreedyTextDecoder object."""
        super().__init__()
        self._confidences = confidences

    def __call__(
        self, model_output: ModelOutput, batch: BatchedSample
    ) -> List[Dict[str, ArrayLike]]:
        """Convert a model output to a token sequence.

        Parameters
        ----------
        model_output: ModelOutput
            The output of a Seq2Seq model. Should contain a N x S x C matrix, where S
            is the sequence length, N is the batch size and C is the number of classes.
        batch: BatchedSample
            Batch information.

        Returns
        -------
        List[Dict[str, ArrayLike]]
            A List of sequences of tokens corresponding to the decoded output and the
            output confidences encapsulated within a dictionary.
        """
        model_output = model_output.output
        indices = np.argmax(model_output, -1)
        confs = np.take_along_axis(model_output, indices[:, :, None], -1).squeeze(-1)

        output = [
            {self.KEY_TEXT: ii, self.KEY_TEXT_CONF: conf}
            for ii, conf in zip(indices, confs)
        ]
        return output
