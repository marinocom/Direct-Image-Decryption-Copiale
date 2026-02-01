"""QOL collection of visualisation functions."""

from __future__ import annotations

from typing import (
    List,
    Optional,
    Tuple,
    Union,
)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from numpy.typing import ArrayLike
from pathlib import Path
from PIL import Image


def plotimg(img: ArrayLike) -> None:
    """Display an image in a singleton plot.

    :param img: Input image in array-like fashion.
    """
    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    plt.close()


def display_prediction(
    fname: str,
    pred_coords: ArrayLike,
    gt_coords: ArrayLike,
    output: str,
) -> None:
    """Display a prediction alongside the ground truth.

    :param fname: Path to the input image.
    :param pred_coords: 1D Prediction bounding boxes.
    :param gt_coords: 1D Ground Truth bounding boxes.
    :param output: A filename to store the output plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot()

    img = np.array(Image.open(fname))
    height, width, c = img.shape

    canvas = np.vstack([img, img, img])
    canvas[height : 2 * height, :, :] = 255

    ax.imshow(canvas)

    y1pd, y2pd = 2 * height, 3 * height
    y1gt, y2gt = 0, height

    colors = plt.cm.hsv(np.linspace(0, 1, len(gt_coords)))

    for ii, ((x1pd, x2pd), (x1gt, x2gt)) in enumerate(zip(pred_coords, gt_coords)):
        gt_width = x2gt - x1gt
        pd_width = x2pd - x1pd
        ax.add_patch(
            patches.Rectangle(
                (x1gt, y1gt),
                gt_width,
                height,
                color=colors[ii],
                alpha=0.5,
                linewidth=1,
            )
        )
        ax.add_patch(
            patches.Rectangle(
                (x1pd, y1pd),
                pd_width,
                height,
                color=colors[ii],
                alpha=0.5,
                linewidth=1,
            )
        )
        ax.arrow(
            x1gt + (gt_width / 2),
            y2gt,
            x1pd + (pd_width / 2) - (x1gt + (gt_width / 2)),
            height,
            color=colors[ii],
            linewidth=1,
        )
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)


def display_bboxes(
    fname: str,
    bboxes: ArrayLike,
    output: str,
) -> None:
    """Display a prediction on the image. Ignores negative bboxes.

    :param fname: Path to the input image.
    :param bboxes: 1D Prediction bounding boxes.
    :param output: A filename to store the output plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot()

    img = np.array(Image.open(fname))
    height, _, _ = img.shape

    colors = plt.cm.hsv(np.linspace(0, 1, len(bboxes)))

    ax.imshow(img)

    for ii, (x1, x2) in enumerate(bboxes):
        if x1 < 0 and x2 < 0:
            continue
        width = x2 - x1
        ax.add_patch(
            patches.Rectangle(
                (x1, 0),
                width,
                height,
                color=colors[ii],
                alpha=0.5,
                linewidth=1,
            )
        )
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)


def display_bboxes_loaded(
    img: Union[Image, ArrayLike, torch.Tensor],
    bboxes: ArrayLike,
    output: Optional[Path] = None,
) -> None:
    """Display bounding boxes on a loaded image. Ignores negative bboxes.

    Parameters
    ----------
    img: Union[Image, ArrayLike, torch.Tensor]
        An input image. Should be castable to array.
    bboxes: ArrayLike
        A list of bboxes in format N x 2.
    """
    img = np.array(img)

    fig = plt.figure()
    ax = fig.add_subplot()
    height, _, _ = img.shape

    ax.imshow(img)
    colors = plt.cm.hsv(np.linspace(0, 1, len(bboxes)))

    for ii, (x1, x2) in enumerate(bboxes):
        if x1 < 0 and x2 < 0:
            continue
        width = x2 - x1
        ax.add_patch(
            patches.Rectangle(
                (x1, 0),
                width,
                height,
                color=colors[ii],
                alpha=0.5,
                linewidth=1,
            )
        )
    if output is None:
        plt.show()
    else:
        fig.savefig(output, bbox_inches="tight", dpi=150)
    plt.close(fig)
