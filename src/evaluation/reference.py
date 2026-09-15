"""Per-field-of-view reference image construction.

There is no ground-truth background-free image for real microscopy data, so every method is
scored against a constructed reference with a flat background and the cellular structure intact.
The construction and its parameters are fixed here so that the evaluation is reproducible.

Protocol (thesis Section 2.5.1):
    1. float conversion and normalisation to [0, 1]
    2. estimate the low-frequency illumination field L by morphological opening
    3. smooth L with a 2D third-order polynomial surface -> L_tilde
    4. flatten:  I - L_tilde
    5. histogram-match the result back to the original image
"""
from __future__ import annotations

import numpy as np
from skimage.exposure import match_histograms
from skimage.morphology import disk, opening

# Fixed protocol parameters. Changing these changes the reference, and therefore every metric.
OPENING_RADIUS_PX = 80
POLY_ORDER = 3


def normalise(image: np.ndarray) -> np.ndarray:
    """Convert to float64 and scale to [0, 1]. Constant images map to zeros."""
    img = image.astype(np.float64)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-12:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def _polynomial_surface(field: np.ndarray, order: int = POLY_ORDER) -> np.ndarray:
    """Least-squares fit of a 2D polynomial surface of the given order to `field`."""
    h, w = field.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x = xx.ravel() / max(w - 1, 1)
    y = yy.ravel() / max(h - 1, 1)

    terms = [
        (x ** i) * (y ** j)
        for i in range(order + 1)
        for j in range(order + 1 - i)
    ]
    design = np.column_stack(terms)
    coeffs, *_ = np.linalg.lstsq(design, field.ravel(), rcond=None)
    return (design @ coeffs).reshape(h, w)


def illumination_field(image: np.ndarray, radius: int = OPENING_RADIUS_PX) -> np.ndarray:
    """Estimate the slowly varying illumination field, smoothed to remove residual structure."""
    coarse = opening(image, disk(radius))
    return _polynomial_surface(coarse)


def match_to_source(corrected: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Put a method output on the same intensity scale as the source image.

    This matters more than it looks. PSNR is sensitive to a constant offset or a change of scale,
    so comparing a [0, 1]-rescaled method output against a histogram-matched reference measures
    the scale difference between them rather than the fidelity of the background removal. Every
    method output and the reference itself are therefore matched back to the same source
    histogram, which is the only way the numbers are comparable across methods.
    """
    return match_histograms(normalise(corrected), normalise(source))


def build_reference(image: np.ndarray, radius: int = OPENING_RADIUS_PX) -> np.ndarray:
    """Return the per-FOV reference image used as the target for PSNR and SSIM."""
    img = normalise(image)
    flattened = np.clip(img - illumination_field(img, radius=radius), 0.0, 1.0)
    return match_to_source(flattened, img)
