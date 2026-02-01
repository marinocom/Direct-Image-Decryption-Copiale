"""Import an intermediate pickle file and convert it to JSON for readability."""

import json
import pickle as pkl
import numpy as np

from argparse import ArgumentParser, Namespace
from pathlib import Path

from seq_recog.utils.io import load_pickle_prediction
from seq_recog.data.base_dataset import BaseVocab


def setup() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "file",
        help="What file to extract information from",
        type=Path,
        metavar="<PATH>",
    )
    parser.add_argument(
        "--text",
        help="Specifies that it is a text file, hence it needs a vocabulary file.",
        type=Path,
        default=None,
        metavar="<PATH TO VOCAB>",
    )
    args = parser.parse_args()
    return args


def unarray(x):
    try:
        x = x.tolist()
    except AttributeError:
        ...
    return x


def main(args: Namespace) -> None:
    predictions = load_pickle_prediction(args.file)

    if args.text is not None:
        vocab = BaseVocab(args.text)
        predictions = {
            k: vocab.decode(vocab.unpad(unarray(v))) for k, v in predictions.items()
        }
    else:
        predictions = {k: unarray(v) for k, v in predictions.items()}

    with open(args.file.parent / (args.file.stem + ".json"), "w") as f_out:
        json.dump(predictions, f_out, indent=4)


if __name__ == "__main__":
    main(setup())
