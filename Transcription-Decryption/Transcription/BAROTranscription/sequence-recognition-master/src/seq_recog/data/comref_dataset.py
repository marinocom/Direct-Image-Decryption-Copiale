"""Implements loading of COMREF samples."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from torch.utils import data as D
from torch import TensorType

from seq_recog.data.base_dataset import (
    BaseDataset,
    BaseDataConfig,
    BaseVocab,
    DatasetSample,
    BatchedSample,
)


def load_comref_splits(
    splits_file: Path,
    vocab: BaseVocab,
    config: BaseDataConfig,
    special: bool,
) -> ComrefDataset:
    with open(splits_file, "r") as f_in:
        splits = json.load(f_in)
    root_path = splits_file.parent

    full_dataset = []

    for split in ["train", "valid", "test"]:
        datasets = []
        for work in splits[split]:
            gt_file = root_path / work / f"{work}.mtn"
            img_folder = root_path / work / "measures"

            datasets.append(
                ComrefDataset(
                    str(img_folder),
                    str(gt_file),
                    vocab,
                    config,
                    split == "train",
                    special,
                )
            )
        full_dataset.append(D.ConcatDataset(datasets))

    train_dataset, valid_dataset, test_dataset = full_dataset
    return train_dataset, valid_dataset, test_dataset


class ComrefDataset(BaseDataset):
    """Load COMREF samples for inference."""

    RE_PROPERTIES = re.compile(r"\[[^]]*\]")
    RE_SEPARATOR = re.compile(r",")

    def __init__(
        self,
        image_folder: str,
        dataset_file: str,
        vocab: BaseVocab,
        config: BaseDataConfig,
        train: bool = True,
        special: bool = False,
    ) -> None:
        super().__init__(image_folder, dataset_file, vocab, config, train, special)

    def _load_data(self) -> None:
        with open(self._dataset_file, "r") as f_gt:
            gt = json.load(f_gt)

        for fn, sample in gt.items():
            transcript = self.RE_PROPERTIES.sub("", sample)
            transcript = self.RE_SEPARATOR.split(transcript)

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
                    segm=np.array([]),
                    fname=str(self._image_folder / fn),
                )
            )


def load_proto_comref_splits(
    splits_file: Path,
    prm_vocab: BaseVocab,
    sec_vocab: BaseVocab,
    config: BaseDataConfig,
    special: bool = True,
) -> ProtoComrefDataset:
    with open(splits_file, "r") as f_in:
        splits = json.load(f_in)
    root_path = splits_file.parent

    full_dataset = []

    for split in ["train", "valid", "test"]:
        datasets = []
        for work in splits[split]:
            gt_file = root_path / work / f"{work}.mtn"
            img_folder = root_path / work / "measures"

            datasets.append(
                ProtoComrefDataset(
                    str(img_folder),
                    str(gt_file),
                    prm_vocab,
                    sec_vocab,
                    config,
                    split == "train",
                    special,
                )
            )
        full_dataset.append(D.ConcatDataset(datasets))

    train_dataset, valid_dataset, test_dataset = full_dataset
    return train_dataset, valid_dataset, test_dataset


class TwoHeadBatchedSample(BatchedSample):
    """Dataset batch that accounts for the presence of a second output."""

    ...

    # gt_sec: Union[ArrayLike, TensorType]


@dataclass
class TwoHeadDatasetSample(DatasetSample):
    """Dataset sample that accounts for the presence of a second output."""

    gt_sec: ArrayLike


class ProtoComrefDataset(BaseDataset):
    def __init__(
        self,
        image_folder: str,
        dataset_file: str,
        prm_vocab: BaseVocab,
        sec_vocab: BaseVocab,
        config: BaseDataConfig,
        train: bool = True,
        special: bool = True,
    ) -> None:
        self._sec_vocab = sec_vocab
        super().__init__(image_folder, dataset_file, prm_vocab, config, train, special)

    def _load_data(self) -> None:
        with open(self._dataset_file, "r") as f_mtn:
            curr = json.load(f_mtn)

        for fname, transcript in curr.items():
            tokens = [x["name"] for x in transcript]
            gt_len = len(tokens)
            tokens = self._vocab.prepare_data(
                tokens,
                self._seqlen,
                self._special,
            )
            groups = self._sec_vocab.prepare_data(
                [x["group"] for x in transcript],
                self._seqlen,
                self._special,
            )

            self._samples.append(
                TwoHeadDatasetSample(
                    gt=tokens,
                    gt_len=gt_len,
                    segm=np.array([]),
                    fname=str(self._image_folder / fname),
                    gt_sec=groups,
                )
            )

    def __getitem__(self, index: int) -> BatchedSample:
        """Retrieve a single sample from the dataset as indexed.

        Parameters
        ----------
        index: int
            The index of the sample to retrieve.
        """
        sample = self._samples[index]

        img, og_shape, new_shape = self._load_image(sample.fname)

        # return TwoHeadBatchedSample(
        return BatchedSample(
            img=img,
            gt=sample.gt,
            gt_len=sample.gt_len,  # (un)normalised_coords,
            fname=sample.fname,
            og_shape=og_shape,
            curr_shape=new_shape,
            segm=sample.segm,
            gt_sec=sample.gt_sec,
        )
