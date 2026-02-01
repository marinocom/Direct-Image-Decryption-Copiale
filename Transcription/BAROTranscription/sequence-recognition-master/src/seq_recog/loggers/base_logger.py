"""Class dedicated to outputting the results from an experiment."""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from numpy.typing import ArrayLike

from seq_recog.data.base_dataset import BatchedSample
from seq_recog.metrics.base_metric import BaseMetric
from seq_recog.formatters.base_formatter import BaseFormatter
from seq_recog.utils.io import load_pickle_prediction


class BaseLogger(ABC):
    """Logger class for run information."""

    def load_prediction(self, path: Path) -> Dict[str, Any]:
        """Load a prediction pickle file into memory.

        Parameters
        ----------
        path: Path
            Path to the prediction file.

        Returns
        -------
        Dict[str, Any]
            A dictionary whose keys are metric fields and values are the stored numpy
            arrays.
        """
        return load_pickle_prediction(path)

    @abstractmethod
    def process_and_log(
        self,
        output: Any,
        batch: BatchedSample,
    ) -> None:
        """Process a batch of model outputs and log them to a file.

        Parameters
        ----------
        output: Any
            The output for a single batch of the model. Must be batch-wise iterable.
        batch: BatchedSample
            An input batch with ground truth and filename data.
        """
        raise NotImplementedError

    @abstractmethod
    def aggregate(self) -> Dict[str, Any]:
        """Aggregate logged metrics.

        Returns
        -------
        Dict[str: Any]
            A dict whose keys are aggregate names and values are the aggregations of
            values related to a metric.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Cleanup logger class and close all related files."""
        raise NotImplementedError


class SimpleLogger(BaseLogger):
    """Logger class for run information."""

    def __init__(
        self,
        path: Path,
        formatter: BaseFormatter,
        metric: BaseMetric,
        log_results: bool = True,
        *args,
    ) -> None:
        """Set up the class and base paths.

        Parameters
        ----------
        path: Path
            The logging path for the epoch and the mode (train / valid / test) within
            the experiment.
        formatter: BaseFormatter
            Object to convert the output from a model to a processable or measurable
            output.
        metric: BaseMetric
            Object that measures fidelity to the desired processable output.
        log_results: bool = True
            Whether to store processable outputs or not (makes sense during validation
            or if this is the final output for some task, but not much during training
            aside from generating tons of superfluous data).
        """
        super().__init__()
        self._path = path
        self._formatter = formatter
        self._metric = metric
        self._log_results = log_results

        self._metric_paths = {
            name: open(path / f"metric_{name}.pkl", "ba")
            for name in self._metric.keys()
        }

        if self._log_results:
            self._result_paths = {
                name: open(path / f"results_{name}.pkl", "ba")
                for name in self._formatter.keys()
            }

    def _write_metrics(
        self,
        metric_res: List[Dict[str, ArrayLike]],
        names: List[str],
    ) -> None:
        """Write metric results to the corresponding metric file.

        Parameters
        ----------
        metric_res: List[Dict[str, ArrayLike]]
            A list of metric results where each element is a dictionary of metrics
            corresponding to an input image file.
        names: List[str]
            A list of file names aligned to each metric_res element.
        """
        write_dict = {
            metric_name: {
                img_name: out[metric_name] for img_name, out in zip(names, metric_res)
            }
            for metric_name in self._metric.keys()
        }
        for k, v in write_dict.items():
            f_out = self._metric_paths[k]
            pickle.dump(v, f_out)
            f_out.flush()

    def _write_results(
        self,
        final_res: List[Dict[str, ArrayLike]],
        names: List[str],
    ) -> None:
        """Write final results to the corresponding metric file.

        Parameters
        ----------
        metric_res: List[Dict[str, ArrayLike]]
            A list of final results where each element is a dictionary of metrics
            corresponding to an input image file.
        names: List[str]
            A list of file names aligned to each final_res element.
        """
        write_dict = {
            result_name: {
                img_name: out[result_name] for img_name, out in zip(names, final_res)
            }
            for result_name in self._formatter.keys()
        }
        for k, v in write_dict.items():
            f_out = self._result_paths[k]
            pickle.dump(v, f_out)
            f_out.flush()

    def process_and_log(
        self,
        output: Any,
        batch: BatchedSample,
    ) -> None:
        """Process a batch of model outputs and log them to a file.

        Parameters
        ----------
        output: Any
            The output for a single batch of the model. Must be batch-wise iterable.
        batch: BatchedSample
            An input batch with ground truth and filename data.
        """
        results = self._formatter(output, batch)
        metrics = self._metric(results, batch)

        fnames = [x.split("/")[-1] for x in batch.fname]

        self._write_metrics(metrics, fnames)
        if self._log_results:
            self._write_results(results, fnames)

    def aggregate(self) -> Dict[str, Any]:
        """Aggregate logged metrics.

        Returns
        -------
        Dict[str: Any]
            A dict whose keys are aggregate names and values are the aggregations of
            values related to a metric.
        """
        predictions = None
        for name in self._metric.keys():
            loaded = self.load_prediction(self._path / f"metric_{name}.pkl")
            if predictions is None:
                predictions = {fn: {name: value} for fn, value in loaded.items()}
            else:
                for fn, val in loaded.items():
                    predictions[fn][name] = val

        return self._metric.aggregate([x for x in predictions.values()])

    def close(self):
        """Cleanup logger class and close all related files."""
        for v in self._metric_paths.values():
            v.close()

        if self._log_results:
            for v in self._result_paths.values():
                v.close()
