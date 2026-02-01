"""Base experiment types and configs."""

import json
import signal
import sys
import threading
import multiprocessing
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from pathlib import Path
from shutil import copyfile
from typing import Dict, Optional, Tuple, Type
from pydantic import BaseModel

import wandb

from .configurations import DirectoryConfig
from ..data.base_dataset import BaseDataConfig
from ..models.base_model import BaseModelConfig
from ..trainers.base_trainer import BaseTrainerConfig


class ExperimentConfig(BaseModel):
    """Global experiment settings."""

    exp_name: str
    description: str
    cipher: Optional[str] = None
    wandb_mode: str
    wandb_project: str

    dirs: DirectoryConfig
    data: BaseDataConfig
    model: BaseModelConfig
    train: BaseTrainerConfig

    @classmethod
    def generate_template(cls) -> Dict:
        """Generate a config representation for the current experiment."""

        def _generate_subdict(tt):
            output = {}
            for name, field in tt.items():
                subt = field.type_
                if hasattr(subt, "__fields__"):
                    output[name] = _generate_subdict(subt.__fields__)
                else:
                    if hasattr(subt, "__name__"):
                        output[name] = subt.__name__
                    elif hasattr(subt, "_name"):
                        output[name] = subt._name
                    else:
                        output[name] = str(subt)
            return output

        return _generate_subdict(cls.__fields__)


class Experiment(ABC):
    """Implements a single experiment."""

    EXPERIMENT_CONFIG = ExperimentConfig

    def __init__(self):
        """Initialise Experiment."""
        self.cfg, self.test_weights, self.debug = self.setup()

        self.train_data, self.valid_data, self.test_data = None, None, None
        self.training_formatter, self.valid_formatter = None, None
        self.training_metric, self.valid_metric = None, None
        self.model = None

        self.trainer, self.validator, self.tester = None, None, None

        self.initialise_everything()

    @staticmethod
    def _load_configuration(
        config_path: str,
        args: Namespace,
        config_type: Type,
    ) -> ExperimentConfig:
        path = Path(config_path)
        with open(path, "r") as f_config:
            cfg = json.load(f_config)
            cfg["exp_name"] = path.stem + ("_test" if args.test else "")

            if args.base_data_dir is not None:
                cfg["dirs"]["base_data_dir"] = args.base_data_dir
            if args.results_dir is not None:
                cfg["dirs"]["results_dir"] = args.results_dir
            if args.batch_size is not None:
                cfg["train"]["batch_size"] = args.batch_size
        cfg = config_type(**cfg)

        return cfg

    @staticmethod
    def _setup_dirs(
        cfg: DirectoryConfig,
        exp_name: str,
    ) -> DirectoryConfig:
        results_path = Path(cfg.results_dir) / exp_name
        results_path.mkdir(exist_ok=True, parents=True)
        cfg.results_dir = str(results_path)

        base_data_path = Path(cfg.base_data_dir)

        for key in cfg.__dict__.keys():
            if key not in ["results_dir", "results_dir"]:
                newpath = base_data_path / cfg.__dict__[key]
                assert newpath.exists(), f"{str(newpath)} does not exist."
                cfg.__dict__[key] = str(newpath)

        return cfg

    def setup(self) -> Tuple[ExperimentConfig, str, bool]:
        """Load configuration and set up paths.

        :returns: Singleton configuration object.
        """
        signal.signal(signal.SIGINT, self.sigint_handler)
        parser = ArgumentParser(
            description="Model training framework.",
        )
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "--config_path",
            type=str,
            metavar="<PATH TO INPUT CONFIG JSON FILE>",
            help="Configuration path for the experiment",
            required=False,
        )
        group.add_argument(
            "--get_template",
            type=str,
            metavar="<PATH TO OUTPUT CONFIG JSON FILE>",
            help="If and where to produce a configuration template",
            default=None,
            required=False,
        )
        parser.add_argument(
            "--test",
            type=str,
            help="Test the input experiment with the provided parameter weights",
            default=None,
            required=False,
        )
        parser.add_argument(
            "--debug",
            help="Sets the code in debug mode.",
            action="store_true",
            required=False,
        )
        parser.add_argument(
            "--base_data_dir",
            metavar="<PATH TO BASE DATA FOLDER>",
            help="Override the base data directory.",
            type=str,
            default=None,
            required=False,
        )
        parser.add_argument(
            "--results_dir",
            metavar="<PATH TO RESULTS FOLDER>",
            help="Override the results directory.",
            type=str,
            default=None,
            required=False,
        )
        parser.add_argument(
            "--batch_size",
            metavar="<BATCH SIZE VALUE>",
            help="Override the batch size for training.",
            type=int,
            default=None,
            required=False,
        )

        args = parser.parse_args()

        if args.get_template:
            with open(args.get_template, "w") as f_json:
                json.dump(self.EXPERIMENT_CONFIG.generate_template(), f_json, indent=4)
                quit()

        cfg = self._load_configuration(args.config_path, args, self.EXPERIMENT_CONFIG)
        self._setup_dirs(
            cfg.dirs,
            cfg.exp_name,
        )

        wandb.init(
            project=cfg.wandb_project,
            dir=cfg.dirs.results_dir,
            config=cfg.dict(),
            mode=cfg.wandb_mode,
            save_code=True,
            notes=cfg.description,
        )

        cfg.dirs.results_dir = Path(cfg.dirs.results_dir) / wandb.run.name
        cfg.dirs.results_dir.mkdir(exist_ok=False, parents=False)

        fname = "config.json"

        copyfile(args.config_path, Path(cfg.dirs.results_dir) / fname)

        return cfg, args.test, args.debug

    def sigint_handler(
        self,
        signal,
        frame,
    ) -> None:
        if not (
            multiprocessing.current_process().name == "MainProcess"
            and threading.main_thread() == threading.current_thread()
        ):
            return
        """Avoid zombie processes when killing through ctrl + c."""
        option = input(
            "SIGINT detected. Do you want to kill the training process? [y / n]: "
        )
        if option == "y":
            if self.trainer and self.trainer.curr_logger:
                self.trainer.curr_logger.close()

            if self.validator and self.validator.logger:
                self.validator.logger.close()

            if self.tester and self.tester.logger:
                self.tester.logger.close()

            option2 = input("Should test on the best weights be run? [y / n]: ")
            if option2 != "n":
                try:
                    self.model.load_weights(str(self.trainer.best_fname))
                except FileNotFoundError:
                    print("Could not load best weights. Shutting down...")
                    sys.exit(1)
                self.tester.validate(self.model, 0, 0, self.trainer.device)
            sys.exit(0)
        else:
            return

    @abstractmethod
    def initialise_everything(self) -> None:
        """Initialise all member variables for the class."""
        raise NotImplementedError

    def main(self) -> None:
        """Run the experiment."""
        if self.test_weights:
            self.model.load_weights(self.test_weights)
            self.tester.validate(self.model, 0, 0, self.trainer.device)
            sys.exit(0)
        else:
            self.trainer.train()
            sys.exit(0)
