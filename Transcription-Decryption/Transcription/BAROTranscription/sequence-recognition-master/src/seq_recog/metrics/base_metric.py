"""Implementation of a base metric object."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from numpy.typing import ArrayLike

from ..data.base_dataset import BatchedSample


class BaseMetric(ABC):
    """Compute the difference between a set of predictions and the GT."""

    METRIC_NAME = "base_metric"
    KEYS = [METRIC_NAME]
    AGG_KEYS = []

    @abstractmethod
    def __call__(
        self, output: List[Dict[str, Any]], batch: BatchedSample
    ) -> List[Dict[str, ArrayLike]]:
        """Compute the difference between a set of predictions and the GT.

         Parameters
         ----------
         output: List[Dict[str, Any]]
             The output of a model after being properly formatted.
         batch: BatchedSample
             Batch information if needed.

         Returns
         -------
        List[Dict[str, ArrayLike]]
             A value array that measures how far from the GT is each prediction.
        """
        raise NotImplementedError

    @abstractmethod
    def maximise(self) -> bool:
        """Return whether this is a maximising metric or not.

        Returns
        -------
        bool
            True if this is a bigger-is-better metric. False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def aggregate(self, metrics: List[Dict[str, ArrayLike]]) -> List[Dict[str, Any]]:
        """Aggregate a set of predictions from the given metric.

        Parameters
        ----------
        metrics: Dict[str, ArrayLike]
            List of predictions from the metric.

        Returns
        -------
        Dict[str, Any]
            A set of aggregate values summarising an entire prediction.
        """
        raise NotImplementedError

    def keys(self) -> List[str]:
        """Return the list of dictionary keys associated with this metric.

        Returns
        -------
        List[str]
            The list of keys this metric can generate.
        """
        return self.KEYS

    def agg_keys(self) -> List[str]:
        """Return the list of aggregation dictionary keys associated with this metric.

        Returns
        -------
        List[str]
            The list of keys this metric can generate when aggregating stuff.
        """
        return self.AGG_KEYS
