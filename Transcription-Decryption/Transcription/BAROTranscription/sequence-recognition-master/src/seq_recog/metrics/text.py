"""Text-based metrics."""

from typing import Any, Dict, List

import numpy as np
from numpy.typing import ArrayLike

from .base_metric import BaseMetric
from ..data.base_dataset import BatchedSample, BaseVocab
from ..utils.ops import levenshtein


class Levenshtein(BaseMetric):
    """Levenshtein metric."""

    METRIC_NAME = "levenshtein"
    KEYS = [METRIC_NAME]
    AGG_KEYS = []

    def __init__(self, vocab: BaseVocab, padded: bool = False) -> None:
        super().__init__()
        self.vocab = vocab
        self.padded = padded

    def __call__(
        self, output: List[Dict[str, Any]], batch: BatchedSample
    ) -> List[Dict[str, ArrayLike]]:
        """Compute the difference between a set of predictions and the GT.

        Parameters
        ----------
        output: List[Dict[str, Any]]
            The output of a model after being properly formatted. Dicts must
            contain a "text" key.
        batch: BatchedSample
            Batch information if needed.

        Returns
        -------
        Dict[str, ArrayLike]
            A value array that measures how far from the GT is each prediction.
        """
        out = []

        for model_out, gt, ln in zip(output, batch.gt, batch.gt_len):
            text = model_out["text"]
            gt = gt.numpy()
            if self.padded:
                text = self.vocab.unpad(text)
                gt = self.vocab.unpad(gt)
            else:
                ln = ln.item()
                gt = gt[:ln]
            lev = levenshtein(text, gt)[0]
            out.append({"levenshtein": lev})

        return out

    def maximise(self) -> bool:
        """Return whether this is a maximising metric or not.

        Returns
        -------
        bool
            True if this is a bigger-is-better metric. False otherwise.
        """
        return False

    def aggregate(self, metrics: List[Dict[str, ArrayLike]]) -> Dict[str, Any]:
        """Aggregate a set of predictions to return the average edit distance.

        Parameters
        ----------
        metrics: Dict[str, ArrayLike]
            List of predictions from the metric.

        Returns
        -------
        Dict[str, float]
            Average of seqiou predictions for all bounding boxes.
        """
        preds = np.array([pred["levenshtein"] for pred in metrics])
        return {"mean_levenshtein": np.mean(preds), "std_levenshtein": np.std(preds)}


class WordAccuracy(BaseMetric):
    """WordAccuracy metric."""

    METRIC_NAME = "word_accuracy"
    KEYS = [METRIC_NAME]
    AGG_KEYS = [METRIC_NAME]

    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self, output: List[Dict[str, Any]], batch: BatchedSample
    ) -> List[Dict[str, ArrayLike]]:
        """Compute the difference between a set of predictions and the GT.

        Parameters
        ----------
        output: List[Dict[str, Any]]
            The output of a model after being properly formatted. Dicts must
            contain a "text" key.
        batch: BatchedSample
            Batch information if needed.

        Returns
        -------
        Dict[str, ArrayLike]
            A value array that measures how far from the GT is each prediction.
        """
        out = []

        for model_out, gt, ln in zip(output, batch.gt, batch.gt_len):
            ln = ln.item()
            gt_sample = gt[:ln].detach().cpu().numpy()
            pd_sample = model_out["text"]
            if len(gt_sample) != len(pd_sample):
                out.append({self.METRIC_NAME: False})
            else:
                out.append({self.METRIC_NAME: np.all(model_out["text"] == gt_sample)})

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
        """Aggregate a set of predictions to return the average edit distance.

        Parameters
        ----------
        metrics: Dict[str, ArrayLike]
            List of predictions from the metric.

        Returns
        -------
        Dict[str, float]:
            Average of seqiou predictions for all bounding boxes encapsulated in a dict.
        """
        preds = np.array([pred[self.METRIC_NAME] for pred in metrics])
        return {self.METRIC_NAME: np.mean(preds)}
