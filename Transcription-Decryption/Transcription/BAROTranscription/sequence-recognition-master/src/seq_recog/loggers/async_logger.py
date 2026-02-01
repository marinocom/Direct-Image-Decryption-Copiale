"""Logger that performs tasks separately from the main process and thread."""

import multiprocessing as mp
import pickle
from pathlib import Path
from typing import Any, Dict, List

from seq_recog.data.base_dataset import BatchedSample
from seq_recog.metrics.base_metric import BaseMetric
from seq_recog.formatters.base_formatter import BaseFormatter
from seq_recog.loggers.base_logger import BaseLogger


class AsyncLogger(BaseLogger):
    """Asynchronous logger class that performs logging outside the main thread."""

    def __init__(
        self,
        path: Path,
        formatter: BaseFormatter,
        metric: BaseMetric,
        log_results: bool = True,
        workers: int = 4,
    ) -> None:
        """Initialise wrapper.

        Parameters
        ----------
        logger: AsyncLogger
            A logging class that is apt for parallelisation.
        workers: int
            Max number of parallel processes. Note that two extra processes are created
            for writers.
        """
        super().__init__()
        self._path = path
        self._mgr = mp.Manager()
        self._pool = mp.Pool(processes=workers)
        self._writer_pool = mp.Pool(processes=2)

        self._metric_queue = self._mgr.Queue()
        self._result_queue = self._mgr.Queue()

        self._formatter = formatter
        self._metric = metric
        self._log_results = log_results

        self._processor = AsyncProcessor(
            self._formatter, self._metric, self._log_results
        )

        self._jobs = []
        writer_a = self._writer_pool.apply_async(
            _writer, (path, self._metric.keys(), "metric", self._metric_queue)
        )
        self._jobs.append(writer_a)
        if self._log_results:
            writer_b = self._writer_pool.apply_async(
                _writer, (path, self._formatter.keys(), "results", self._result_queue)
            )
            self._jobs.append(writer_b)

    def process_and_log(
        self,
        output: Any,
        batch: BatchedSample,
    ) -> None:
        """Asynchronously perform processing and logging of batches.

        Parameters
        ----------
        output: Any
            The output for a single batch of the model. Must be batch-wise iterable.
        batch: BatchedSample
            An input batch with ground truth and filename data.
        """
        self._jobs.append(
            self._pool.apply_async(
                self._processor,
                (output, batch, self._metric_queue, self._result_queue),
                error_callback=_error_callback,
            )
        )

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
        self._pool.close()
        self._pool.join()

        self._metric_queue.put("kill")
        self._result_queue.put("kill")
        self._writer_pool.close()
        self._writer_pool.join()


def _writer(
    path: Path,
    names: List[str],
    fname_base: str,
    q: mp.Queue,
) -> None:
    metric_paths = {
        name: open(path / f"{fname_base}_{name}.pkl", "ba") for name in names
    }

    while True:
        content = q.get()
        if content == "kill":
            for v in metric_paths.values():
                v.close()
            return
        img_names, output = content
        write_dict = {
            name: {img_name: out[name] for img_name, out in zip(img_names, output)}
            for name in names
        }
        for k, v in write_dict.items():
            f_out = metric_paths[k]
            pickle.dump(v, f_out)
            f_out.flush()


class AsyncProcessor:
    """Performs output conversion and metric computations."""

    def __init__(
        self,
        formatter: BaseFormatter,
        metric: BaseMetric,
        log_output: bool,
    ) -> None:
        """Construct async processor object.

        Parameters
        ----------
        formatter: BaseFormatter
            Result formatting object.
        metric: BaseMetric
            Metric computation object.
        log_output: bool
            Whether or not to log formatted outputs.
        """
        self._formatter = formatter
        self._metric = metric
        self._log_results = log_output

    def __call__(
        self,
        output: Any,
        batch: BatchedSample,
        metric_q: mp.Queue,
        result_q: mp.Queue,
    ) -> None:
        """Call the processor in order to generate results and write them to output.

        Parameters
        ----------
        output: Any
            Model output.
        batch: BatchedSample
            Batch of input data.
        metric_q: mp.Queue
            Queue to write metric results.
        result_q: mp.Queue
            Queue to write formatted results.
        """
        fnames = [x.split("/")[-1] for x in batch.fname]
        results = self._formatter(output, batch)

        if self._log_results:
            result_q.put((fnames, results))

        metrics = self._metric(results, batch)
        metric_q.put((fnames, metrics))


def _error_callback(e: Exception) -> None:
    print(f"ERROR! {str(e)}")
    raise e
