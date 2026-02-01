import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import re
import xml.etree.ElementTree as ET

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, NamedTuple, Tuple


RE_CIPHER = re.compile(r"(\w+)_(\w+)")
MIN_SUBWORD_LENGTH = 16


def imshow(img: np.array) -> None:
    plt.figure()
    plt.imshow(img)
    plt.show()
    plt.close()


root_path = Path("/home/ptorras/Documents/Datasets/decrypt/Validated")
output_path = Path("/home/ptorras/Documents/Datasets/decrypt_cleanup")

split_lut = {"training": "train", "valid": "valid", "test": "test"}


class BoundingBox(NamedTuple):
    x1: int
    x2: int
    y1: int
    y2: int


gt_data = {}

for folder in root_path.iterdir():
    cipher, split = RE_CIPHER.match(folder.name).groups()
    cipher = cipher.lower()
    split = split.lower()

    img_output_path = output_path / cipher / "words"
    img_output_path.mkdir(parents=True, exist_ok=True)

    for page in folder.iterdir():
        page_name = page.name

        img_fname = (
            f"{page_name}.png"
            if (page / f"{page_name}.png").exists()
            else f"{page_name}.jpg"
            if (page / f"{page_name}.jpg").exists()
            else f"{page_name}.jpeg"
        )
        ann_fname = f"{img_fname}.xml"

        xml_root = ET.parse(str(page / ann_fname)).getroot()

        img = cv2.imread(str(page / img_fname))
        height, width, channels = img.shape

        print("\t", page)

        for ii, line_element in enumerate(xml_root.iter("line")):

            if not len(list(line_element)):
                print(f"No Symbols in line {ii} of {page_name}")
                continue

            line_symbols = list(line_element.iter("symbol"))

            for subword_length in range(MIN_SUBWORD_LENGTH, len(line_symbols)):
                for sindex in range(len(line_symbols) - subword_length):
                    curr_slice = line_symbols[sindex : sindex + subword_length]

                    slice_transcript = []
                    slice_bboxes = []

                    for symbol_element in curr_slice:
                        symbol = symbol_element.attrib["text"]
                        slice_bboxes.append(
                            tuple(
                                [
                                    int(symbol_element.attrib[k])
                                    for k in ["x1", "x2", "y1", "y2"]
                                ]
                            )
                        )
                        slice_transcript.append(symbol)

                    if cipher not in gt_data:
                        gt_data[cipher] = {}
                    curr_cipher = gt_data[cipher]

                    if split not in curr_cipher:
                        curr_cipher[split] = {}
                    curr_dict = curr_cipher[split]

                    slice_bboxes = np.array(slice_bboxes)
                    slice_bounding = BoundingBox(
                        max(slice_bboxes[:, 0].min(), 0),
                        min(slice_bboxes[:, 1].max(), width),
                        max(slice_bboxes[:, 2].min(), 0),
                        min(slice_bboxes[:, 3].max(), height),
                    )

                    img_slice = img[
                        slice_bounding.y1 : slice_bounding.y2,
                        slice_bounding.x1 : slice_bounding.x2,
                        :,
                    ]

                    line_name = f"{page_name}_l{ii:04d}_wl{subword_length:03d}_w{sindex:04d}.png"

                    width_coords = slice_bboxes[:, :2]
                    width_coords = width_coords - width_coords[0, 0]

                    curr_dict[line_name] = {
                        "segm": width_coords.tolist(),
                        "ts": " ".join(slice_transcript),
                    }
                    cv2.imwrite(
                        str(output_path / cipher / "words" / line_name), img_slice
                    )

    with open(output_path / cipher / f"gt_words_{split}.json", "w") as f_json:
        json.dump(gt_data[cipher][split], f_json)
