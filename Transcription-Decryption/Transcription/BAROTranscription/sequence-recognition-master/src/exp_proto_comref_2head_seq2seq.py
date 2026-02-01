"""Experiment with an RNN-based Seq2Seq model."""

from pathlib import Path

from seq_recog.data.base_dataset import (
    BaseVocab,
    BaseDataConfig,
)
from seq_recog.data.comref_dataset import load_proto_comref_splits
from seq_recog.experiments.base_experiment import Experiment, ExperimentConfig
from seq_recog.experiments.configurations import ProtoComrefDirectoryConfig
from seq_recog.formatters import seq2seq_formatters
from seq_recog.loggers.base_logger import SimpleLogger
from seq_recog.loggers.async_logger import AsyncLogger
from seq_recog.metrics import text
from seq_recog.models.rnn_seq2seq import KangSeq2Seq2Head, KangSeq2Seq2HeadConfig
from seq_recog.trainers.base_trainer import BaseTrainer, BaseTrainerConfig
from seq_recog.validators.base_validator import BaseValidator


class ProtoComref2HeadSeq2SeqExperimentConfig(ExperimentConfig):
    """Global experiment settings."""

    dirs: ProtoComrefDirectoryConfig
    data: BaseDataConfig
    model: KangSeq2Seq2HeadConfig
    train: BaseTrainerConfig


class ProtoComref2HeadSeq2SeqExperiment(Experiment):
    """Object modelling the Experiment with Arnau Baró's CRNN model."""

    EXPERIMENT_CONFIG = ProtoComref2HeadSeq2SeqExperimentConfig

    def __init__(self):
        """Initialise object."""
        super().__init__()

        self.prm_vocab, self.sec_vocab = None, None

    def initialise_everything(self) -> None:
        """Initialise all member variables for the class."""
        # Data
        self.prm_vocab = BaseVocab(self.cfg.dirs.prm_vocab_data)
        self.sec_vocab = BaseVocab(self.cfg.dirs.sec_vocab_data)
        self.train_data, self.valid_data, self.test_data = load_proto_comref_splits(
            Path(self.cfg.dirs.splits_file),
            self.prm_vocab,
            self.sec_vocab,
            self.cfg.data,
            True,
        )

        # Formatters
        self.training_formatter = seq2seq_formatters.GreedyTextDecoder()
        self.valid_formatter = seq2seq_formatters.GreedyTextDecoder()

        # Metrics
        self.training_metric = text.Levenshtein(self.prm_vocab, padded=True)
        self.valid_metric = text.Levenshtein(self.prm_vocab, padded=True)

        # Model and training-related
        self.model = KangSeq2Seq2Head(self.cfg.model, self.cfg.data)
        self.validator = BaseValidator(
            self.valid_data,
            self.valid_formatter,
            self.valid_metric,
            Path(self.cfg.dirs.results_dir),
            self.cfg.train.batch_size,
            0 if self.debug else self.cfg.train.workers,
            "valid",
            AsyncLogger,
        )
        self.tester = BaseValidator(
            self.test_data,
            self.valid_formatter,
            self.valid_metric,
            Path(self.cfg.dirs.results_dir),
            self.cfg.train.batch_size,
            0 if self.debug else self.cfg.train.workers,
            "test",
            AsyncLogger,
        )

        self.trainer = BaseTrainer(
            self.model,
            self.train_data,
            self.cfg.train,
            Path(self.cfg.dirs.results_dir),
            self.validator,
            self.training_formatter,
            self.training_metric,
            None,
            AsyncLogger,
        )


if __name__ == "__main__":
    exp = ProtoComref2HeadSeq2SeqExperiment()
    exp.main()
