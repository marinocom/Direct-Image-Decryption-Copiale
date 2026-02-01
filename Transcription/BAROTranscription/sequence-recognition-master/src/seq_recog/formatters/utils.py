"""Miscellaneous formatting utilities."""

from typing import Any, Dict, List
from warnings import warn

from .base_formatter import BaseFormatter
from ..data.base_dataset import BatchedSample
from ..models.base_model import ModelOutput


class Compose(BaseFormatter):
    """Combine various formatters into one single call."""

    def __init__(self, formatters: List[BaseFormatter]):
        """Initialise composition of formatters.

        Parameters
        ----------
        formatters: List[BaseFormatter]
            List of formatters to be computed for a single output.
        """
        self.formatters = formatters
        self._keys = [x for fmt in self.formatters for x in fmt.KEYS]

        if len(set(self._keys)) != (len(self._keys)):
            warn(
                "There are duplicate key names within the composition formatter."
                "This will lead to some results being overwritten. Double check "
                "your class definitions for formatters."
            )

    def __call__(
        self, model_output: ModelOutput, batch: BatchedSample
    ) -> List[Dict[str, Any]]:
        """Compute multiple formatters between a set of predictions and the GT.

        Parameters
        ----------
        model_output: ModelOutput
            The output of a model.
        batch: BatchedSample
            Batch information if needed.

        Returns
        -------
        List[Dict[str, Any]]
            List of formatted predictions.
        """
        output = []

        for fmt in self.formatters:
            current = fmt(model_output, batch)

            if not len(output):
                output = current
            else:
                output = [old | curr for old, curr in zip(output, current)]
        return output

    def keys(self) -> List[str]:
        """Get the set of keys this formatter can produce in the output dictionary.

        Returns
        -------
        List[str]
            List of keys that can be used by this formatter.
        """
        return self._keys


class AddFilename(BaseFormatter):
    """Add filename into the formatting dictionary."""

    KEY_FILENAME = "filename"
    KEYS = [KEY_FILENAME]

    def __call__(
        self, model_output: ModelOutput, batch: BatchedSample
    ) -> List[Dict[str, Any]]:
        """Provide the filename to the formatted dict.

        Parameters
        ----------
        model_output: ModelOutput
            The output of a model.
        batch: BatchedSample
            Batch information if needed. Must contain a filename field with a Path.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dicts where keys are the names of the formatting
            techniques and the values are the formatted outputs.
        """
        return [{self.KEY_FILENAME: x} for x in batch.filename]


class AddGroundTruthText(BaseFormatter):
    """Add ground truth into the formatting dictionary."""

    KEY_GT = "gt_text"
    KEYS = [KEY_GT]

    def __call__(
        self, model_output: ModelOutput, batch: BatchedSample
    ) -> List[Dict[str, Any]]:
        """Provide the filename to the formatted dict.

        Parameters
        ----------
        model_output: ModelOutput
            The output of a model.
        batch: BatchedSample
            Batch information if needed.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dicts where keys are the names of the formatting
            techniques and the values are the formatted outputs.
        """
        return [{self.KEY_GT: x[:l] for x, l in zip(batch.gt, batch.og_len)}]


class AddGroundTruthSegm(BaseFormatter):
    """Add ground truth into the formatting dictionary."""

    KEY_GT = "gt_segm"
    KEYS = [KEY_GT]

    def __call__(
        self, model_output: ModelOutput, batch: BatchedSample
    ) -> List[Dict[str, Any]]:
        """Provide the filename to the formatted dict.

        Parameters
        ----------
        model_output: ModelOutput
            The output of a model.
        batch: BatchedSample
            Batch information if needed.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dicts where keys are the names of the formatting
            techniques and the values are the formatted outputs.
        """
        return [{self.KEY_GT: x[:l] for x, l in zip(batch.segm, batch.og_len)}]
