"""Experiment with a CTC CNN Transformer model."""

from pathlib import Path

from seq_recog.data.base_dataset import (
    BaseVocab,
    BaseDataConfig,
)
from seq_recog.data.comref_dataset import load_comref_splits
from seq_recog.experiments.base_experiment import Experiment, ExperimentConfig
from seq_recog.experiments.configurations import ComrefDirectoryConfig
from seq_recog.formatters import ctc_formatters
from seq_recog.loggers.base_logger import SimpleLogger
from seq_recog.metrics import text
from seq_recog.models.ctc_models import CTCVITModel, CTCVITModelConfig
from seq_recog.trainers.base_trainer import BaseTrainer, BaseTrainerConfig
from seq_recog.validators.base_validator import BaseValidator


class VitCTCExperimentConfig(ExperimentConfig):
    """Global experiment settings."""

    dirs: ComrefDirectoryConfig
    data: BaseDataConfig
    model: CTCVITModelConfig
    train: BaseTrainerConfig


class CNNTformerExperiment(Experiment):
    """Object modelling the Experiment with Arnau Baró's CRNN model."""

    EXPERIMENT_CONFIG = VitCTCExperimentConfig

    def __init__(self):
        """Initialise object."""
        super().__init__()

    def initialise_everything(self) -> None:
        """Initialise all member variables for the class."""
        # Data
        self.vocab = BaseVocab(self.cfg.dirs.vocab_data)
        self.train_data, self.valid_data, self.test_data = load_comref_splits(
            Path(self.cfg.dirs.splits_file),
            self.vocab,
            self.cfg.data,
            False,
        )

        # Formatters
        self.training_formatter = ctc_formatters.GreedyTextDecoder()
        self.valid_formatter = ctc_formatters.GreedyTextDecoder()

        # Metrics
        self.training_metric = text.Levenshtein(self.vocab)
        self.valid_metric = text.Levenshtein(self.vocab)

        # Model and training-related
        self.model = CTCVITModel(self.cfg.model, self.cfg.data)
        self.validator = BaseValidator(
            self.valid_data,
            self.valid_formatter,
            self.valid_metric,
            Path(self.cfg.dirs.results_dir),
            self.cfg.train.batch_size,
            0 if self.debug else self.cfg.train.workers,
            "valid",
            SimpleLogger,
        )
        self.tester = BaseValidator(
            self.test_data,
            self.valid_formatter,
            self.valid_metric,
            Path(self.cfg.dirs.results_dir),
            self.cfg.train.batch_size,
            0 if self.debug else self.cfg.train.workers,
            "test",
            SimpleLogger,
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
            SimpleLogger,
        )


if __name__ == "__main__":
    exp = CNNTformerExperiment()
    exp.main()
