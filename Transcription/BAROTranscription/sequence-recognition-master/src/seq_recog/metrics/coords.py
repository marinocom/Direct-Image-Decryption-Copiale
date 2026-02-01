"""Coordinate-based metrics."""

from .base_metric import BaseMetric
from typing import Any, Dict, List

import numpy as np
from numpy.typing import ArrayLike
from ..data.generic_decrypt import BatchedSample
from ..utils.ops import seqiou


class SeqIoU(BaseMetric):
    """Sequence-level Intersection over Union metric."""

    METRIC_NAME = "seqiou"
    KEYS = [METRIC_NAME]
    AGG_KEYS = ["mean_iou", "iou_hits_25", "iou_hits_50", "iou_hits_75"]

    def __init__(self) -> None:
        """Initialise Object."""
        super().__init__()

    def __call__(
        self, output: List[Dict[str, Any]], batch: BatchedSample
    ) -> List[Dict[str, ArrayLike]]:
        """Compute the IoU of the output sequences and the ground truth.

        Parameters
        ----------
        output: List[Dict[str, Any]]
            The output of a model after being properly formatted. Dicts must
            contain a "coords1d" key.
        batch: BatchedSample
            Batch information if needed.

        Returns
        -------
        Dict[str, ArrayLike]
            The IoU for each bounding box for each element in the sequence.
        """
        out = []

        for model_out, gt, ln in zip(output, batch.segm.numpy(), batch.og_len):
            iou = seqiou(model_out["coords1d"], gt[:ln])
            out.append({"seqiou": iou})

        return out

    def maximise(self) -> bool:
        """Return whether this is a maximising metric or not.

        Returns
        -------
        bool
            True if this is a bigger-is-better metric. False otherwise.
        """
        return True

    def aggregate(self, metrics: List[Dict[str, ArrayLike]]) -> Dict[str, float]:
        """Aggregate a set of predictions to return the average seqiou.

        Parameters
        ----------
        metrics: Dict[str, ArrayLike]
            List of predictions from the metric.

        Returns
        -------
        float
            Average of seqiou predictions for all bounding boxes.
        """
        preds = np.concatenate([pred["seqiou"] for pred in metrics])
        hits25 = np.count_nonzero(preds >= 0.25) / (len(preds) or 1)
        hits50 = np.count_nonzero(preds >= 0.50) / (len(preds) or 1)
        hits75 = np.count_nonzero(preds >= 0.75) / (len(preds) or 1)
        return {
            "mean_iou": np.mean(preds),
            "iou_hits_25": hits25,
            "iou_hits_50": hits50,
            "iou_hits_75": hits75,
        }
