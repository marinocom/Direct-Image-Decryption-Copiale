"""Draw some predictions on test partitions."""

import json
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path

from seq_alignment.utils.visualisation import display_prediction
from seq_alignment.utils.io import load_pickle_prediction


def main(args: Namespace) -> None:
    with open(args.dataset_file, 'r') as f_data:
        gt_coords = json.load(f_data)
    img_path = args.dataset_folder
    pred_path = args.prediction_path

    out_path = pred_path / "images"
    out_path.mkdir(exist_ok=True)

    pred_coords = load_pickle_prediction(pred_path / "results_coords1d.pkl")
    for ii, (k, v) in enumerate(pred_coords.items()):
        if ii >= args.nimgs:
            break
        display_prediction(
            str(img_path / k),
            pred_coords[k],
            np.array(gt_coords[k]["segm"]),
            str(out_path / (k + ".png"))
        )


def setup() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "dataset_file",
        help="Path to a dataset ground truth file.",
        type=Path,
    )
    parser.add_argument(
        "dataset_folder",
        help="Path to the folder where images are stored.",
        type=Path,
    )
    parser.add_argument(
        "prediction_path",
        help="Path to a set of predictions from a test epoch.",
        type=Path,
    )
    parser.add_argument(
        "nimgs",
        help="Number of images to produce",
        type=int,
        default=25,
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main(setup())
