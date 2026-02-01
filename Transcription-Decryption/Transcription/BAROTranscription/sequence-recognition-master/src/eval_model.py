"""Generate display images for a prediction and evaluate results."""

import json
import re

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict

import numpy as np
from seq_recog.data.base_dataset import (
    BaseVocab,
)
from seq_recog.utils.decoding import Prediction, PredictionGroup
from seq_recog.utils.io import load_pickle_prediction


RE_NUMBER = re.compile(r"(\-?[0-9]+),(\-?[0-9]+)")
RE_DOT = re.compile(r".")


class AlignmentEvaluator:
    """Perform evaluation of models, ensembles of models and old predictions."""

    def __init__(self, args: Namespace) -> None:
        self._groups = {}
        self._vocab = BaseVocab(args.vocab)
        self._groundtruth = self._load_gt(args.groundtruth)
        self._fs_conv = self._create_fewshot_conversor(self._vocab)

        pickles = args.pickles or []
        fewshot = args.fewshot or []
        prediction = args.prediction or []

        self._npreds = len(pickles + fewshot + prediction)

        for p in pickles:
            self._groups = self._load_pickle_predictions(
                p,
                self._groups,
                self._groundtruth,
            )

        for p in fewshot:
            self._groups = self._load_few_shot_predictions(
                p,
                self._groups,
                self._groundtruth,
            )

        for p in prediction:
            self._groups = self._load_json_predictions(
                p,
                self._groups,
                self._groundtruth,
            )

    def evaluate(self):
        if self._npreds > 1:
            for fn, pg in self._groups.items():
                anchors = pg.find_anchors(0.75)

    def _create_fewshot_conversor(self, vocab: BaseVocab) -> Dict:
        tokens = vocab.vocab2index.keys()
        fewshot = {tok: self._remove_filesystem_chars(tok) for tok in tokens}
        fewshot = {v: k for k, v in fewshot.items()}

        return fewshot

    @staticmethod
    def _remove_filesystem_chars(token: str) -> str:
        token = token.lower()
        token = token.replace(r".", "dt")
        token = token.replace(r'"', "qt")
        token = token.replace(r":", "cl")
        token = token.replace(r"'", "ap")
        return token

    def _load_gt(self, ipath: Path) -> Dict[str, Any]:
        """Load the ground truth file and prepare it for numpy processing.

        Parameters
        ----------
        ipath: Path
            Path for the ground truth file.

        Returns
        -------
        Dict[str, Any]
            Dictionary that represents the input ground truth file. It has keys for
            each image in the dataset and the values are dicts with "segm" and "ts"
            keys. Each stores the segmentation and the transcription of the elements
            in the images.
        """
        output = {}
        with open(ipath, "r") as f_in:
            groundtruth = json.load(f_in)

        for fn, val in groundtruth.items():
            transcript = val["ts"]
            transcript = transcript.split(" ")
            transcript = self._vocab.encode(transcript)
            transcript = np.array(transcript)

            segm = val["segm"]
            segm = np.array(segm)

            output[fn] = {"segm": segm, "ts": transcript}
        return output

    def _load_pickle_predictions(
        self,
        path: Path,
        groups: Dict[str, PredictionGroup],
        gt: Dict,
        reverse: bool = False,
    ) -> Dict[str, PredictionGroup]:
        coord_file = path / "results_coords1d.pkl"
        confs_file = path / "results_coords1d_confidences.pkl"

        assert coord_file.exists(), "Coordinate file does not exist"
        assert confs_file.exists(), "Confidence file does not exist"

        coords = load_pickle_prediction(coord_file)
        confs = load_pickle_prediction(confs_file)

        files = list(set(coords.keys()) & set(confs.keys()))
        for k in files:
            file_coords = coords[k]
            file_confs = confs[k]
            gt_seq = self._groundtruth[k]["ts"]

            pred = Prediction(file_coords, file_confs, gt_seq)
            try:
                groups[k].add_prediction(pred, path.parent.stem)
            except KeyError:
                groups[k] = PredictionGroup([pred], gt_seq, [path.parent.stem])

        return groups

    def _load_few_shot_predictions(
        self,
        path: Path,
        groups: Dict[str, PredictionGroup],
        gt: Dict,
        reverse: bool = False,
    ) -> Dict[str, PredictionGroup]:
        assert (path / "boxes").exists(), "The boxes path does not exist"
        assert (path / "text").exists(), "The text path does not exist"

        bboxes = {}

        for fbbox in (path / "boxes").glob("*.txt"):
            with open(fbbox, "r") as f_in:
                data = f_in.read()

            data = data.split("\n")[:-1]
            data = map(RE_NUMBER.match, data)
            data = map(lambda x: x.groups(), data)
            data = [[int(a), int(b)] for a, b in data]
            data = np.array(data)

            bboxes[fbbox.stem] = data // 4

        pred_text = {}
        for ftext in (path / "text").glob("*.txt"):
            with open(ftext, "r") as f_in:
                data = f_in.read()
            data = data.split(" ")
            data = [self._fs_conv[x] for x in data]
            data = self._vocab.encode(data)
            pred_text[ftext.stem] = np.array(data)

        for fn in set(bboxes.keys()) & set(pred_text.keys()):
            text = pred_text[fn]
            coords = bboxes[fn]
            gt_seq = self._groundtruth[fn]["ts"]
            pred = Prediction.from_detector(text, gt_seq, coords)
            try:
                groups[fn].add_prediction(pred, path.parent.stem)
            except KeyError:
                groups[fn] = PredictionGroup([pred], gt_seq, [fn])
        return groups

    def _load_json_predictions(
        self,
        path: Path,
        groups: Dict[str, PredictionGroup],
        gt: Dict,
        reverse: bool = False,
    ) -> Dict[str, PredictionGroup]:
        raise NotImplementedError


def setup() -> Namespace:
    """Load command-line arguments and set up stuff.

    Returns
    -------
    Namespace
        Input arguments encapsulated in a namespace object.
    """
    parser = ArgumentParser()

    parser.add_argument(
        "groundtruth",
        help="Path to a ground truth file in json format.",
        type=Path,
    )
    parser.add_argument(
        "vocab",
        help="Path to a vocabulary file in json format.",
        type=Path,
    )
    parser.add_argument(
        "--prediction",
        help="Path to a prediction path or file (old json format).",
        action="append",
        type=Path,
    )
    parser.add_argument(
        "--pickles",
        help="Path to an epoch output with pickle files.",
        action="append",
        type=Path,
    )
    parser.add_argument(
        "--fewshot",
        help="Path to the output of a few-shot model.",
        action="append",
        type=Path,
    )

    args = parser.parse_args()
    return args


def main(args: Namespace) -> None:
    """."""
    args = setup()
    evaluator = AlignmentEvaluator(args)
    evaluator.evaluate()


if __name__ == "__main__":
    args = setup()
    main(args)
