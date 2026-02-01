"""Miscellaneous metric utilities."""

from typing import Any, Dict, List
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike

from .base_metric import BaseMetric
from ..data.generic_decrypt import BatchedSample


class Compose(BaseMetric):
    """Combine various metrics into one single call."""

    METRIC_NAME = "compose"

    def __init__(self, metrics: List[BaseMetric]):
        """Initialise composition of metrics.

        The first metric in the list is the one that shall be used for optimisation
        criteria during training.

        Parameters
        ----------
        metrics: List[BaseMetric]
            List of metrics to be computed for a single output.
        """
        super().__init__()
        self.metrics = metrics

        self._keys = [x for mtr in self.metrics for x in mtr.keys()]
        self._agg_keys = [x for mtr in self.metrics for x in mtr.keys()]

        if len(set(self.KEYS)) != self.KEYS:
            warn(
                "There are duplicate key names within the composition metric."
                "This will lead to some results being overwritten. Double check "
                "your class definitions for metrics."
            )

    def __call__(
        self, model_output: List[Dict[str, Any]], batch: BatchedSample
    ) -> List[Dict[str, ArrayLike]]:
        """Compute multiple metrics between a set of predictions and the GT.

        Parameters
        ----------
        model_output: List[Dict[str, Any]]
            The output of a model after being properly formatted.
        batch: BatchedSample
            Batch information if needed.

        Returns
        -------
        Dict[str, ArrayLike]
            A value array that measures how far from the GT is each prediction.
        """
        output = []

        for metric in self.metrics:
            current = metric(model_output, batch)

            if not len(output):
                output = current
            else:
                output = [old | curr for old, curr in zip(output, current)]
        return output

    def maximise(self) -> bool:
        """Return whether the first metric is maximising or not."""
        return self.metrics[0].maximise()

    def aggregate(self, metrics: List[Dict[str, ArrayLike]]) -> Dict[str, Any]:
        """Aggregate a set of predictions to return the average edit distance.

        Parameters
        ----------
        metrics: Dict[str, ArrayLike]
            Array of predictions from the metric.

        Returns
        -------
        Dict[str, float]
            Average of seqiou predictions for all bounding boxes.
        """
        output = {}

        for metric in self.metrics:
            output[metric.METRIC_NAME] = metric.aggregate(metrics)

        return output

    def keys(self) -> List[str]:
        return self._keys

    def agg_keys(self) -> List[str]:
        return self._agg_keys
