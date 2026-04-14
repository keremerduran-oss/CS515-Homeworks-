"""
AugMix data augmentation for improved robustness and uncertainty estimation.

Reference:
    Hendrycks, D., Mu, N., Cubuk, E. D., Zoph, B., Gilmer, J., & Lakshminarayanan, B. (2020).
    AugMix: A simple method to improve robustness and uncertainty under data shift.
    ICLR 2020. https://openreview.net/forum?id=S1gmrxHFvB
"""

import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance
from torchvision import transforms
from typing import List, Tuple


# ── Augmentation operations pool ─────────────────────────────────────────────

def autocontrast(pil_img: Image.Image, _level: float) -> Image.Image:
    """Apply auto contrast to the image."""
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img: Image.Image, _level: float) -> Image.Image:
    """Apply histogram equalization."""
    return ImageOps.equalize(pil_img)


def posterize(pil_img: Image.Image, level: float) -> Image.Image:
    """Reduce each pixel to a given number of bits."""
    level = int((level / 10.0) * 4)
    level = max(1, level)
    return ImageOps.posterize(pil_img, level)


def rotate(pil_img: Image.Image, level: float) -> Image.Image:
    """Rotate the image by level degrees."""
    degrees = (level / 10.0) * 30
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img: Image.Image, level: float) -> Image.Image:
    """Invert all pixels above a threshold."""
    threshold = int((level / 10.0) * 256)
    return ImageOps.solarize(pil_img, threshold)


def shear_x(pil_img: Image.Image, level: float) -> Image.Image:
    """Shear the image along the x-axis."""
    shear = (level / 10.0) * 0.3
    if np.random.uniform() > 0.5:
        shear = -shear
    return pil_img.transform(
        pil_img.size, Image.AFFINE, (1, shear, 0, 0, 1, 0),
        resample=Image.BILINEAR
    )


def shear_y(pil_img: Image.Image, level: float) -> Image.Image:
    """Shear the image along the y-axis."""
    shear = (level / 10.0) * 0.3
    if np.random.uniform() > 0.5:
        shear = -shear
    return pil_img.transform(
        pil_img.size, Image.AFFINE, (1, 0, 0, shear, 1, 0),
        resample=Image.BILINEAR
    )


def translate_x(pil_img: Image.Image, level: float) -> Image.Image:
    """Translate the image along the x-axis."""
    pixels = int((level / 10.0) * pil_img.size[0] * 0.33)
    if np.random.uniform() > 0.5:
        pixels = -pixels
    return pil_img.transform(
        pil_img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0),
        resample=Image.BILINEAR
    )


def translate_y(pil_img: Image.Image, level: float) -> Image.Image:
    """Translate the image along the y-axis."""
    pixels = int((level / 10.0) * pil_img.size[1] * 0.33)
    if np.random.uniform() > 0.5:
        pixels = -pixels
    return pil_img.transform(
        pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels),
        resample=Image.BILINEAR
    )


def color(pil_img: Image.Image, level: float) -> Image.Image:
    """Adjust color saturation."""
    factor = (level / 10.0) * 1.8 + 0.1
    return ImageEnhance.Color(pil_img).enhance(factor)


def contrast(pil_img: Image.Image, level: float) -> Image.Image:
    """Adjust contrast."""
    factor = (level / 10.0) * 1.8 + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(factor)


def brightness(pil_img: Image.Image, level: float) -> Image.Image:
    """Adjust brightness."""
    factor = (level / 10.0) * 1.8 + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(factor)


def sharpness(pil_img: Image.Image, level: float) -> Image.Image:
    """Adjust sharpness."""
    factor = (level / 10.0) * 1.8 + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(factor)


# All available augmentation operations
AUGMENTATIONS = [
    autocontrast, equalize, posterize, rotate, solarize,
    shear_x, shear_y, translate_x, translate_y,
    color, contrast, brightness, sharpness,
]


def augment_and_mix(
    image: Image.Image,
    severity: int = 3,
    mixture_width: int = 3,
    chain_depth: int = -1,
    aug_prob_coeff: float = 1.0,
) -> Image.Image:
    """Apply AugMix augmentation to a PIL image.

    Creates `mixture_width` augmentation chains, each applying 1-3 random
    operations, then mixes them using Dirichlet-sampled weights and blends
    with the original image using a Beta-sampled coefficient.

    Args:
        image: Input PIL image.
        severity: Severity level of each augmentation (1-10).
        mixture_width: Number of augmentation chains to mix.
        chain_depth: Depth of each chain (-1 = random 1-3).
        aug_prob_coeff: Alpha parameter for Dirichlet distribution.

    Returns:
        Augmented PIL image (same size as input).
    """
    ws = np.float32(
        np.random.dirichlet([aug_prob_coeff] * mixture_width)
    )
    m = np.float32(np.random.beta(aug_prob_coeff, aug_prob_coeff))

    image_arr = np.array(image).astype(np.float32)
    mix = np.zeros_like(image_arr)

    for i in range(mixture_width):
        image_aug = image.copy()
        depth = chain_depth if chain_depth > 0 else np.random.randint(1, 4)

        for _ in range(depth):
            op = np.random.choice(AUGMENTATIONS)
            level = np.random.uniform(0.1, severity)
            image_aug = op(image_aug, level)

        mix += ws[i] * np.array(image_aug).astype(np.float32)

    mixed = (1 - m) * image_arr + m * mix
    return Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8))


class AugMixTransform:
    """Callable wrapper for AugMix that can be inserted into a transform pipeline.

    Args:
        severity: Severity of each augmentation (1-10).
        mixture_width: Number of augmentation chains.
        chain_depth: Chain depth (-1 = random).
        alpha: Dirichlet alpha for mixing weights.
    """

    def __init__(
        self,
        severity: int = 3,
        mixture_width: int = 3,
        chain_depth: int = -1,
        alpha: float = 1.0,
    ) -> None:
        self.severity      = severity
        self.mixture_width = mixture_width
        self.chain_depth   = chain_depth
        self.alpha         = alpha

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply AugMix to a PIL image."""
        return augment_and_mix(
            img,
            severity      = self.severity,
            mixture_width = self.mixture_width,
            chain_depth   = self.chain_depth,
            aug_prob_coeff= self.alpha,
        )

    def __repr__(self) -> str:
        return (
            f"AugMixTransform(severity={self.severity}, "
            f"width={self.mixture_width}, depth={self.chain_depth})"
        )
