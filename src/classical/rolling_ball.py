"""Rolling Ball background subtraction.

Estimates the background as the surface traced by a ball of fixed radius rolled beneath the
greyscale intensity surface, then subtracts it. Corrects a slowly varying illumination gradient
well, but because the background is estimated from the image itself, low-contrast structures are
absorbed into the estimate and removed with it.
"""
from __future__ import annotations

import numpy as np
from skimage.restoration import rolling_ball

from evaluation.reference import match_to_source, normalise

DEFAULT_RADIUS_PX = 50


def remove_background(image: np.ndarray, radius: int = DEFAULT_RADIUS_PX) -> np.ndarray:
    """Subtract the rolling-ball background estimate and renormalise to [0, 1]."""
    img = normalise(image)
    background = rolling_ball(img, radius=radius)
    corrected = np.clip(img - background, 0.0, 1.0)
    # Matched to the source histogram so the score is comparable with the other methods.
    return match_to_source(corrected, img)
