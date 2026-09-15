"""Augmentation pipeline (Albumentations), per thesis Section 2.4.2.

Every transform here is one the microscope itself could plausibly produce: a dish rotated on the
stage, a slightly different focus or exposure, a cell in a different orientation. Augmentations
that are not physically plausible for the modality teach the model to be invariant to things it
will never see, which costs capacity for nothing.
"""
from __future__ import annotations

import albumentations as A

ROTATION_LIMIT_DEG = 15
ZOOM_RANGE = (0.9, 1.1)
BRIGHTNESS_CONTRAST_LIMIT = 0.10
ELASTIC_P = 0.2


def train_transform(size: int = 256) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0,
            scale_limit=(ZOOM_RANGE[0] - 1.0, ZOOM_RANGE[1] - 1.0),
            rotate_limit=ROTATION_LIMIT_DEG,
            border_mode=0,
            p=0.7,
        ),
        A.ElasticTransform(alpha=1, sigma=50, p=ELASTIC_P),
        A.RandomBrightnessContrast(
            brightness_limit=BRIGHTNESS_CONTRAST_LIMIT,
            contrast_limit=BRIGHTNESS_CONTRAST_LIMIT,
            p=0.5,
        ),
        A.PadIfNeeded(size, size, border_mode=0),
        A.CenterCrop(size, size),
    ])


def eval_transform(size: int = 256) -> A.Compose:
    """No augmentation at evaluation time — only the geometry needed to match the input shape."""
    return A.Compose([
        A.PadIfNeeded(size, size, border_mode=0),
        A.CenterCrop(size, size),
    ])
