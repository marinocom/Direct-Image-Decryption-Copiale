"""
Adapt old notation format into a confier json file.
"""

import json
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Dict, Tuple


RE_FNAME = re.compile(r"(coord|gt)_(train|valid|test)\..*")
RE_LINE = re.compile(r"(\w+\.\w+)\|(.*)")
RE_GTRUTH = re.compile(r"~")
RE_COORDS = re.compile(r" ")


def setup() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "root_path",
        help="Base path for the dataset files",
        type=str,
    )
    parser.add_argument(
        "--coord_separator",
        help="Separator for coordinate files",
        type=str,
        default=" ",
    )
    parser.add_argument(
        "--gt_separator", help="Separator for ground truth files", type=str, default=" "
    )
    args = parser.parse_args()
    return args


def get_files(root_path: Path) -> Dict[str, Dict[str, Path]]:
    files = [x for x in root_path.glob("*.txt") if not x.is_dir()]

    fileorg = {part: {} for part in ["train", "valid", "test"]}

    for x in files:
        parts = RE_FNAME.match(x.name)
        if parts is None:
            continue
        ftype, partition = parts.groups()

        fileorg[partition][ftype] = x

    return fileorg


def split_lines(fpath: Path, separator: object) -> Dict[str, List[str]]:
    output = {}
    with open(fpath, "r") as f_coord:
        contents = f_coord.read()
    contents = contents.split("\n")

    if not len(contents[-1]):
        contents = contents[:-1]

    for line in contents:
        fn, elements = RE_LINE.match(line).groups()
        output[fn] = separator.split(elements)

    return output


def main(args: Namespace) -> None:

    gt_sep = re.compile(args.gt_separator)
    coord_sep = re.compile(args.coord_separator)

    root_path = Path(args.root_path)
    files = get_files(root_path)

    for partition, files in files.items():
        gt_lines = split_lines(files["gt"], gt_sep)

        if "coord" in files:
            coord_lines = split_lines(files["coord"], coord_sep)
            all_files = list(set(list(coord_lines.keys()) + list(gt_lines.keys())))

            out_file = {
                file: {
                    "ts": "~".join(gt_lines[file]),
                    "segm": [
                        (x1, x2)
                        for x1, x2 in zip(
                            coord_lines[file][:-1:2], coord_lines[file][1::2]
                        )
                    ],
                }
                for file in all_files
            }
        else:
            out_file = {
                file: {
                    "ts": "~".join(gt_lines[file]),
                    "segm": None,
                }
                for file in gt_lines.keys()
            }

        with open(root_path / f"gt_{partition}.json", "w") as f_out:
            json.dump(out_file, f_out)


if __name__ == "__main__":
    main(setup())
