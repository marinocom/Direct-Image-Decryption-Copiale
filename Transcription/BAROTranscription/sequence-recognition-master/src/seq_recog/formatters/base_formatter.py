"""Base formatter class implementation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch

from ..data.base_dataset import BatchedSample
from ..models.base_model import ModelOutput


class BaseFormatter(ABC):
    """Abstracts converting from a model output to the desired format."""

    KEYS = []

    @abstractmethod
    def __call__(
        self, model_output: ModelOutput, batch: BatchedSample
    ) -> List[Dict[str, Any]]:
        """Convert a model output to any other formatting.

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
        raise NotImplementedError

    def keys(self) -> List[str]:
        """Return the list of dictionary keys associated with this formatter.

        Returns
        -------
        List[str]
            The list of keys this formatter can generate.
        """
        return self.KEYS
