"""U-Net training.

Loss is binary cross-entropy and Dice at 50:50. BCE alone optimises per-pixel accuracy, which a
model can achieve while losing thin structures, because they are a small fraction of the pixels.
Dice measures overlap and so penalises exactly that failure. Weighting them equally is the
compromise: pixel accuracy without abandoning fine detail.

Optimiser and schedule follow the thesis configuration (Section 2.4.2).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import callbacks, optimizers

from unet.model import build_unet

EPOCHS = 80
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
DICE_SMOOTH = 1.0


def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    totals = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = (2.0 * intersection + DICE_SMOOTH) / (totals + DICE_SMOOTH)
    return 1.0 - tf.reduce_mean(dice)


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return 0.5 * tf.reduce_mean(bce) + 0.5 * dice_loss(y_true, y_pred)


def build_callbacks(checkpoint: Path) -> list:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    return [
        callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss",
                                  save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.3, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=12,
                                restore_best_weights=True, verbose=1),
        callbacks.CSVLogger("results/training_log.csv"),
    ]


def train(train_ds, val_ds, checkpoint: Path = Path("models/unet_best.h5")):
    # Mixed precision for throughput; the output layer is pinned to float32 in model.py.
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    model = build_unet()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999,
                                  clipnorm=1.0),
        loss=combined_loss,
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=build_callbacks(checkpoint),
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path,
                        help="dataset root containing train/ and val/ splits (split by FOV)")
    parser.add_argument("--checkpoint", default=Path("models/unet_best.h5"), type=Path)
    args = parser.parse_args()

    raise SystemExit(
        "Wire up your dataset loader here: build train_ds and val_ds from "
        f"{args.data}, split 70/15/15 by field of view, then call train(train_ds, val_ds)."
    )
