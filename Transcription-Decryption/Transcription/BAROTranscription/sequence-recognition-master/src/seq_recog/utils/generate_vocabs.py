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
MIN_SUBWORD_LENGTH = 12


def imshow(img: np.array) -> None:
    plt.figure()
    plt.imshow(img)
    plt.show()
    plt.close()


root_path = Path("/home/ptorras/Documents/Datasets/decrypt/Validated")
output_path = Path("/home/ptorras/Documents/Datasets/decrypt_cleanup")

split_lut = {"training": "train", "valid": "valid", "test": "test"}


token_data = {}

for folder in root_path.iterdir():
    cipher, split = RE_CIPHER.match(folder.name).groups()
    cipher = cipher.lower()
    split = split.lower()

    if cipher not in token_data:
        token_data[cipher] = []

    for page in folder.iterdir():
        xml_file = [x for x in page.iterdir() if x.suffix == ".xml"][0]
        root_elm = ET.parse(xml_file).getroot()

        symbols = [
            symbol.attrib["text"]
            for line in root_elm.iter("line")
            for symbol in line.iter("symbol")
        ]
        token_data[cipher] = list(set(token_data[cipher] + symbols))


#%%

for cipher in token_data.keys():
    with open(output_path / cipher.lower() / "vocab.json", "w") as f_out:
        json.dump({"labels": token_data[cipher]}, f_out)
