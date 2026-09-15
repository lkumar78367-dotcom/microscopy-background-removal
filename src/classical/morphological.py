"""Morphological background correction.

White top-hat (image minus its opening) keeps structures smaller than the structuring element and
removes everything larger, which is what makes it suitable for removing a broad illumination
gradient. It amplifies high-frequency content indiscriminately, so noise is enhanced along with
the cells, and an over-large element costs faint structure.

The structuring element used for local refinement is deliberately smaller than the one used for
the background estimate in `evaluation.reference`.
"""
from __future__ import annotations

import cv2
import numpy as np

from evaluation.reference import match_to_source, normalise

DEFAULT_TOPHAT_RADIUS_PX = 15
DEFAULT_DENOISE_RADIUS_PX = 2


def _ellipse(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def white_tophat(image: np.ndarray, radius: int = DEFAULT_TOPHAT_RADIUS_PX) -> np.ndarray:
    """White top-hat: image minus its morphological opening."""
    img = normalise(image).astype(np.float32)
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, _ellipse(radius))


def remove_background(
    image: np.ndarray,
    tophat_radius: int = DEFAULT_TOPHAT_RADIUS_PX,
    denoise_radius: int = DEFAULT_DENOISE_RADIUS_PX,
) -> np.ndarray:
    """White top-hat followed by a small opening to suppress the noise it amplifies."""
    corrected = white_tophat(image, radius=tophat_radius)
    if denoise_radius > 0:
        corrected = cv2.morphologyEx(corrected, cv2.MORPH_OPEN, _ellipse(denoise_radius))
    # Matched to the source histogram so the score is comparable with the other methods.
    return match_to_source(corrected, image)
