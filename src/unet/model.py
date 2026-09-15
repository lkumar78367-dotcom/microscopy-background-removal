"""Five-level U-Net with skip connections (TensorFlow 2.15).

Encoder block: two 3x3 convolutions, batch normalisation, ReLU, then 2x2 max pooling.
Decoder block: bilinear upsampling, concatenation with the matching encoder features, then the
same double convolution. 64 feature maps at level 1 rising to 1024 at the bottleneck; a single
sigmoid channel out.

The skip connections are the point: the encoder discards spatial precision as it downsamples, and
concatenating the encoder features back in at each decoder level is what lets the network suppress
a broad illumination gradient without also smoothing away cell boundaries.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model

BASE_FILTERS = 64
DEPTH = 4  # four downsampling steps -> five levels including the bottleneck
INPUT_SHAPE = (256, 256, 3)  # greyscale replicated across three channels


def _double_conv(x, filters: int, name: str):
    for i in (1, 2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                          name=f"{name}_conv{i}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.Activation("relu", name=f"{name}_relu{i}")(x)
    return x


def build_unet(input_shape: tuple[int, int, int] = INPUT_SHAPE,
               base_filters: int = BASE_FILTERS,
               depth: int = DEPTH) -> Model:
    inputs = layers.Input(shape=input_shape, name="input")

    skips = []
    x = inputs
    for level in range(depth):
        filters = base_filters * (2 ** level)
        x = _double_conv(x, filters, name=f"enc{level + 1}")
        skips.append(x)
        x = layers.MaxPooling2D(2, name=f"pool{level + 1}")(x)

    x = _double_conv(x, base_filters * (2 ** depth), name="bottleneck")

    for level in reversed(range(depth)):
        filters = base_filters * (2 ** level)
        x = layers.UpSampling2D(2, interpolation="bilinear", name=f"up{level + 1}")(x)
        x = layers.Concatenate(name=f"skip{level + 1}")([x, skips[level]])
        x = _double_conv(x, filters, name=f"dec{level + 1}")

    outputs = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32",
                            name="output")(x)
    return Model(inputs, outputs, name="unet_background_removal")


if __name__ == "__main__":
    build_unet().summary()
