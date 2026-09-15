"""PSNR and SSIM with the settings fixed by the evaluation protocol.

Acceptance criteria were defined before the experiment was run:
    PSNR >= 26 dB and SSIM >= 0.80.
A method meets the criteria only if it clears both.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

DATA_RANGE = 1.0        # images are normalised to [0, 1]
SSIM_WIN_SIZE = 11      # 11x11 Gaussian window
SSIM_SIGMA = 1.5
K1, K2 = 0.01, 0.03

PSNR_THRESHOLD_DB = 26.0
SSIM_THRESHOLD = 0.80


@dataclass(frozen=True)
class Score:
    psnr_db: float
    ssim: float

    @property
    def meets_criteria(self) -> bool:
        return self.psnr_db >= PSNR_THRESHOLD_DB and self.ssim >= SSIM_THRESHOLD


def score(output: np.ndarray, reference: np.ndarray) -> Score:
    """Score one method output against the per-FOV reference image."""
    if output.shape != reference.shape:
        raise ValueError(f"shape mismatch: {output.shape} vs {reference.shape}")

    psnr = peak_signal_noise_ratio(reference, output, data_range=DATA_RANGE)
    ssim = structural_similarity(
        reference,
        output,
        data_range=DATA_RANGE,
        win_size=SSIM_WIN_SIZE,
        gaussian_weights=True,
        sigma=SSIM_SIGMA,
        K1=K1,
        K2=K2,
        use_sample_covariance=False,
    )
    return Score(psnr_db=float(psnr), ssim=float(ssim))


def summarise(scores: list[Score]) -> dict[str, float]:
    """Mean and standard deviation across a test set, as reported in the results table."""
    psnr = np.array([s.psnr_db for s in scores], dtype=float)
    ssim = np.array([s.ssim for s in scores], dtype=float)
    return {
        "psnr_mean": float(psnr.mean()),
        "psnr_sd": float(psnr.std(ddof=1)) if len(psnr) > 1 else 0.0,
        "ssim_mean": float(ssim.mean()),
        "ssim_sd": float(ssim.std(ddof=1)) if len(ssim) > 1 else 0.0,
        "n": int(len(scores)),
    }
