"""Some handcrafted data augmentation classes and utilities."""

from typing import Tuple

import numpy as np
import cv2

from torchvision import transforms as T
import torchvision.transforms.functional as F


PIPELINES = {
    "basic_decrypt": [
        T.GaussianBlur(3),
        T.RandomEqualize(),
        T.RandomPerspective(fill=255),
    ],
    "basic_comref": [
        T.RandomPerspective(fill=255),
        T.ColorJitter(0.2, 0.0, 0.2, 0.4),
        T.GaussianBlur(5),
    ],
    None: [],
}


class Blackout:
    """Simple debugging transform to remove image contents completely."""

    def __call__(self, image):
        """Remove image contents."""
        return image - image


class PadToMax:
    """Pad an image to a fixed maximum size."""

    def __init__(self, max_size: Tuple[int, int]):
        """Construct augmentation object.

        Parameters
        ----------
        max_size: Tuple[int, int]
            Maximum size to pad images to.
        """
        self.max_size = max_size

    def __call__(self, image):
        """Perform padding."""
        w, h = image.size
        hpf = int(self.max_size[0] - w)
        hp = max(hpf // 2, 0)
        vpf = int(self.max_size[1] - h)
        vp = max(vpf // 2, 0)
        padding = [hp + int(hpf % 2), vp + int(vpf % 2), hp, vp]
        return F.pad(image, padding, 0, "constant")


class ResizeKeepingRatio:
    """Resize an input image keeping its aspect ratio."""

    def __init__(self, max_size: Tuple[int, int]):
        self.max_size = max_size

    def __call__(self, image):
        """Perform resizing."""
        max_w, max_h = self.max_size
        w, h = image.size

        ratio = min(max_w / w, max_h / h)

        return F.resize(image, [int(h * ratio), int(w * ratio)])


class BinariseFixed:
    """Binarise a PIL image using a fixed previously-known threshold."""

    def __init__(self, threshold):
        self.lut = [int(x < threshold) * 255 for x in range(256)]

    def __call__(self, image):
        """Perform binarisation."""
        return image.convert("L").point(self.lut).convert("RGB")


class BinariseCustom:
    """Binarise a PIL image using a fixed provided threshold."""

    def __call__(self, image, threshold):
        """Perform binarisation."""
        return (
            image.convert("L").point(lambda x: int(x < threshold) * 255).convert("RGB")
        )


class KanungoNoise:
    """Applies Kanungo noise into a Numpy image."""

    def __init__(self, alpha=2.0, beta=2.0, alpha_0=1.0, beta_0=1.0, mu=0.05, k=2):
        """Apply Kanungo noise model to a binary image.

        T. Kanungo, R. Haralick, H. Baird, W. Stuezle, and D. Madigan.
        A statistical, nonparametric methodology for document degradation model
        validation. IEEE Transactions Pattern Analysis and Machine Intelligence
        22(11):1209 - 1223, 2000.

        Parameters
        ----------
        alpha: float
            Controls the probability of a foreground pixel flip
        alpha_0: float
            Controls the probability of a foreground pixel flip
        beta: float
            Controls the probability of a background pixel flip
        beta_0: float
            Controls the probability of a background pixel flip
        mu: float
            Constant probability of flipping for all pixels
        k: int
            Diameter of the disk structuring element for the closing operation

        Returns
        -------
        ArrayLike
            Binary image
        """
        self.alpha = alpha
        self.beta = beta
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.mu = mu
        self.k = k

    def __call__(self, img):
        """Perform addition of Kanungo noise.

        Parameters
        ----------
        img: ArrayLike
            Binary image, either [0..1] or [0..255]
        """
        H, W = img.shape
        img = img // np.max(img)
        dist = cv2.distanceTransform(1 - img, cv2.DIST_L1, 3)
        dist2 = cv2.distanceTransform(img, cv2.DIST_L1, 3)
        P = (self.alpha_0 * np.exp(-self.alpha * dist ** 2)) + self.mu
        P2 = (self.beta_0 * np.exp(-self.beta * dist2 ** 2)) + self.mu
        distorted = img.copy()
        distorted[((P > np.random.rand(H, W)) & (img == 0))] = 1
        distorted[((P2 > np.random.rand(H, W)) & (img == 1))] = 0
        closing = cv2.morphologyEx(
            distorted,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.k, self.k)),
        )
        return closing * 255


class ToNumpy:
    """Converts an input image to Numpy."""

    def __call__(self, image):
        return np.asarray(image, np.uint8)


class ToFloat:
    """Casts an input image to float."""

    def __call__(self, image):
        return image.astype(np.float32)


class StackChannels:
    """Converts an single-channel image to a 3-channel one."""

    def __call__(self, image):
        return np.stack([image] * 3, axis=-1)
