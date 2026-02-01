"""Base types for datasets in the model collection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from PIL import Image
import torch.utils.data as D
import torch
from torchvision import transforms as T

from pydantic import BaseModel

from .augmentations import PIPELINES


@dataclass
class DatasetSample:
    """Encapsulates the information of a single sample of the dataset."""

    gt: ArrayLike
    gt_len: int
    segm: Optional[ArrayLike]
    fname: str


# @dataclass
# class BatchedSample:
#     """Represents a batch of samples which are ready for inference."""
#
#     img: Union[ArrayLike, torch.FloatTensor]
#     gt: Union[ArrayLike, torch.LongTensor]
#     gt_len: Union[ArrayLike, torch.LongTensor]
#     fname: Tuple
#     og_shape: Tuple
#     curr_shape: Tuple
#     segm: Optional[Union[ArrayLike, torch.LongTensor]]


# FIXME NamedTuples cannot be extended => should convert to dataclass
# @dataclass
class BatchedSample(NamedTuple):
    """Represents a batch of samples which are ready for inference."""

    img: Union[ArrayLike, torch.FloatTensor]
    gt: Union[ArrayLike, torch.LongTensor]
    gt_sec: Union[ArrayLike, torch.LongTensor]
    gt_len: Union[ArrayLike, torch.LongTensor]
    fname: Tuple
    og_shape: Tuple
    curr_shape: Tuple
    segm: Optional[Union[ArrayLike, torch.LongTensor]]


class BaseVocab:
    """Converts back and forth between text and indices of tokens."""

    blank = "<BLANK>"
    go_tok = "<GO>"
    stop_tok = "<STOP>"
    pad_tok = "<PAD>"

    BLANK_INDEX = 0
    GO_INDEX = 1
    STOP_INDEX = 2
    PAD_INDEX = 3

    def __init__(self, path: str) -> None:
        """Initialise the vocab object with the file pointed at the input path.

        Parameters
        ----------
        path: str
            Path to a vocab json file. This json file should have a "labels" member
            containing an ordered list of all possible tokens.
        """
        with open(path, "r") as f_labels:
            jlabels = json.load(f_labels)

        self.tokens = [self.blank, self.go_tok, self.stop_tok, self.pad_tok]
        self.vocab = self.tokens + jlabels["labels"]

        self.vocab2index = {x: ii for ii, x in enumerate(self.vocab)}
        self.index2vocab = {v: k for k, v in self.vocab2index.items()}

    def __len__(self):
        """Get the number of tokens in the vocabulary.

        Returns
        -------
        int
            Number of tokens in the vocab.
        """
        return len(self.vocab)

    def encode(self, labels: List[str]) -> List[int]:
        """Convert the input token sequence into a list of integers.

        Parameters
        ----------
        labels: List[str]
            List of textual labels for a text transcript of an image.

        Returns
        -------
        List[int]
            List of integers representing the input sequence.
        """
        return [self.vocab2index[x] for x in labels]

    def decode(self, encoded: List[int]) -> List[str]:
        """Convert the input token sequence back to a list of textual tokens.

        Parameters
        ----------
        encoded: List[int]
            List of indices for a Decrypt transcript.

        Returns
        -------
        List[str]
            List of tokens for the underlying sequence.
        """
        return [self.index2vocab[x] for x in encoded]

    def pad(
        self,
        encoded: List[int],
        pad_len: int,
        special: bool = False,
    ) -> ArrayLike:
        """Pad input sequence to a fixed width using special tokens.

        Parameters
        ----------
        encoded : List[int]
            List of indices for a Decrypt transcript.
        pad_len : int
            Expected length of the output sequence.
        special : bool, optional
            Whether to insert go and end tokens in the transcript, by default False.

        Returns
        -------
        ArrayLike
            List of indices with a go and an end token at the beginning
            and the end of the sequence (if the special flag is passed as True) plus
            padding tokens to match the max sequence length provided as argument.
        """
        padded = np.full(pad_len, self.PAD_INDEX)
        if special:
            assert len(encoded) + 2 <= pad_len
            padded[1 : len(encoded) + 1] = encoded
            padded[0] = self.GO_INDEX
            padded[len(encoded) + 1] = self.STOP_INDEX
        else:
            assert len(encoded) <= pad_len
            padded[: len(encoded)] = encoded
        return padded

    def unpad(self, padded: List[int]) -> List[int]:
        """Perform the inverse operation to the pad function.

        Parameters
        ----------
        padded : List[int]
            List containing a padded sequence of indices.

        Returns
        -------
        List[int]
            The same input sequence with padding and extra go/end tokens removed. Zeros
            are NOT removed (this depends on the algorithm that uses them, thus it is
            left at the discretion of the user).
        """
        output = []
        for x in padded:
            if x == self.vocab2index[self.stop_tok]:
                break
            if (
                x == self.vocab2index[self.pad_tok]
                or x == self.vocab2index[self.go_tok]
            ):
                continue
            output.append(x)
        return output

    def prepare_data(
        self,
        data_in: List[str],
        pad_len: int,
        special: bool = False,
    ) -> ArrayLike:
        """Perform encoding, padding and conversion to array.

        Parameters
        ----------
        data_in : List[str]
            Input sequence in token format.
        pad_len : int
            Length to which to pad the input sequence.
        special: bool
            Whether to insert special start / stop tokens. False by default.

        Returns
        -------
        ArrayLike
            The input sequence with padding in array form.
        """
        data = self.pad(self.encode(data_in), pad_len, special)

        return data


Width = int
Height = int


class BaseDataConfig(BaseModel):
    """Data-Model configuration settings."""

    target_shape: Tuple[Width, Height]
    target_seqlen: int
    aug_pipeline: Optional[str]
    stretch: Optional[Union[float, str]] = None  # Can be "fit" to fit to full size
    hflip: bool = False
    max_length: Optional[int] = None  # Allow only shorter or equal


class BaseDataset(D.Dataset):
    """Performs loading and management of datasets in JSON format."""

    DEFAULT_TRANSFORMS = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    RE_SEPARATOR = re.compile(" ")

    def __init__(
        self,
        image_folder: str,
        dataset_file: str,
        vocab: BaseVocab,
        config: BaseDataConfig,
        train: bool = True,
        special: bool = False,
    ) -> None:
        """Initialise Dataset object.

        Parameters
        ----------
        image_folder: str
            Path where the images of the dataset are stored.
        dataset_file: str
            Path pointing to the dataset json file.
        vocab: GenericDecryptVocab
            Vocabulary object to perform conversions.
        config: DataConfig
            Dataset config object for model parameters.
        train: bool
            Whether this dataset is used for training purposes (in which case data
            augmentation is enabled).
        special: bool
            Whether to use special tokens in padding. False by default.
        """
        super(BaseDataset).__init__()

        self._samples = []
        self._image_folder = Path(image_folder)  # Images Folder
        self._dataset_file = Path(dataset_file)  # GT File
        self._seqlen = config.target_seqlen
        self._target_shape = config.target_shape
        self._hflip = config.hflip
        self._stretch = config.stretch
        self._max_length = config.max_length
        self._vocab = vocab
        self._special = special

        aug_pipeline = PIPELINES[config.aug_pipeline] or [] if train else []
        self._aug_pipeline = T.Compose([*aug_pipeline, self.DEFAULT_TRANSFORMS])

        self._load_data()

    def _load_data(self) -> None:
        with open(self._dataset_file, "r") as f_gt:
            gt = json.load(f_gt)

        for fn, sample in gt.items():
            transcript = self.RE_SEPARATOR.split(sample["ts"])
            if "segm" in sample and sample["segm"] is not None:
                segm = np.array(sample["segm"], dtype=int)
                if self._hflip:
                    segm = abs(segm[::-1] - segm.max())
                    segm = segm[:, [1, 0]]
                    transcript = transcript[::-1]
                segmentation = np.full((self._seqlen, 2), -1)
                segmentation[: len(segm)] = segm
            else:
                if self._hflip:
                    transcript = transcript[::-1]
                segmentation = np.array([])

            gt_len = len(transcript)
            if self._max_length and gt_len > self._max_length:
                continue
            transcript = self._vocab.prepare_data(
                transcript, self._seqlen, self._special
            )

            self._samples.append(
                DatasetSample(
                    gt=transcript,
                    gt_len=gt_len,
                    segm=segmentation,
                    fname=str(self._image_folder / fn),
                )
            )

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns
        -------
        int:
            Integer with the number of samples in the dataset.
        """
        return len(self._samples)

    def _load_image(self, fname: str):
        img = Image.open(fname).convert("RGB")

        og_shape = img.size
        img_width, img_height = og_shape
        tgt_width, tgt_height = self._target_shape

        if isinstance(self._stretch, float) or self._stretch is None:
            factor = min(
                tgt_width / ((self._stretch or 1.0) * img_width),
                tgt_height / img_height,
            )
            new_shape = (
                int(img_width * factor * (self._stretch or 1.0)),
                int(img_height * factor),
            )
        elif isinstance(self._stretch, str) and self._stretch == "fit":
            new_shape = (int(tgt_width), int(tgt_height))
        else:
            raise Exception(f"Wrong parameter for stretching: {self._stretch}.")

        img = img.resize(new_shape)
        if self._hflip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        padded_img = Image.new(img.mode, self._target_shape, (255, 255, 255))
        padded_img.paste(img, (0, 0))
        padded_img = self._aug_pipeline(padded_img)

        return padded_img, og_shape, new_shape

    def __getitem__(self, index: int) -> BatchedSample:
        """Retrieve a single sample from the dataset as indexed.

        Parameters
        ----------
        index: int
            The index of the sample to retrieve.
        """
        sample = self._samples[index]

        img, og_shape, new_shape = self._load_image(sample.fname)

        return BatchedSample(
            img=img,
            gt=sample.gt,
            gt_sec=np.array([]),
            gt_len=sample.gt_len,  # (un)normalised_coords,
            fname=sample.fname,
            og_shape=og_shape,
            curr_shape=new_shape,
            segm=sample.segm,
        )
