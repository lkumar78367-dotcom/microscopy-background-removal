# Results

Mean ± SD on the held-out test set, computed against the per-field-of-view reference image defined
in [METHOD.md](METHOD.md).

| Method | PSNR (dB) | SSIM | PSNR ≥ 26 | SSIM ≥ 0.80 | Passes |
|---|---|---|---|---|---|
| Rolling Ball / opening | 24.1 ± 1.3 | 0.72 ± 0.04 | No | No | **No** |
| Morphological (white top-hat) | 25.6 ± 1.1 | 0.78 ± 0.03 | No | No | **No** |
| U-Net (retrained) | 28.4 ± 0.9 | 0.86 ± 0.02 | Yes | Yes | **Yes** |

## Reading the table

**The two classical methods fail for different reasons, and the pattern matches their mechanics.**

Rolling Ball is lowest on both metrics. It estimates the background from the image itself, so
anything low in contrast is indistinguishable from background and is subtracted along with it.
Visual inspection confirms this: global gradient correction is good, but faint low-brightness
structures are lost. That is a mechanism, not a bug.

White top-hat improves on Rolling Ball on both metrics (+1.5 dB, +0.06 SSIM) because it operates
on local structure rather than a global background surface, so object boundaries and separation
improve. But it amplifies high-frequency content indiscriminately, and noise is high-frequency
content. The margin to the SSIM threshold is only 0.02 — close enough that a different structuring
element could plausibly cross it, which is worth stating rather than glossing over.

The U-Net clears both thresholds with the tightest spread of the three (± 0.9 dB, ± 0.02 SSIM).
The narrow spread matters as much as the mean: it means performance is consistent across fields of
view rather than excellent on some and poor on others. It learns the foreground/background
distinction from examples instead of assuming a model of the background, which is why it can
suppress the gradient without smoothing away structure.

## What this does and does not show

**Does:** on this dataset, with this reference definition and these thresholds, only the U-Net
produces background removal adequate for reliable downstream analysis, and its advantage survives
the robustness check on the reference parameters.

**Does not:** that deep learning is the right choice in general. The classical methods are
seconds of compute, no training data, no GPU, and a handful of lines of code. When the correction
needed is a simple illumination gradient and the structures of interest are well-contrasted, they
are the correct engineering choice. The U-Net earns its cost — annotated data, training time,
compute — only when structural fidelity is the binding requirement.

**Does not:** that the U-Net will transfer. One dataset, one modality pairing, one magnification
regime. Generalisability was one of the research questions and this study could only partly
answer it.

## Relationship to downstream segmentation

PSNR and SSIM measure how faithfully the output reconstructs the reference. They do not measure
whether segmentation of that output is more accurate. A method could win on both metrics and still
not be the best preprocessing choice for a specific segmentation task, because segmentation cares
about boundary localisation in a way that neither metric captures directly. Measuring segmentation
accuracy (Dice, IoU) on the outputs of each method is the natural next step and is not covered
here.
