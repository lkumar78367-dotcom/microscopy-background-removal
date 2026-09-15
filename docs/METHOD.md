# Method and evaluation protocol

This document is the protocol as it was fixed **before** the experiment was run. Acceptance
criteria set after seeing results are not acceptance criteria.

## 1. Research question

Which background-removal method best balances suppression of the illumination gradient against
preservation of cellular structure in phase contrast and DIC microscopy, and does the ranking hold
across the parameters of the evaluation itself?

## 2. Acceptance criteria

| Metric | Threshold | Rationale |
|---|---|---|
| PSNR | ≥ 26 dB | Reconstruction fidelity adequate for downstream segmentation |
| SSIM | ≥ 0.80 | Structural content preserved, not merely intensity matched |

A method passes only if it clears **both**. 28–30 dB and 0.85–0.90 are strong for this modality.

## 3. Dataset

Integrated MiniGlioma–Usiigaci microscopy images, 256×256.

**Split: 70/15/15 by field of view, not by image.** Patches drawn from the same field of view are
not independent — they share the same illumination gradient and often the same cells. Splitting by
image would leak that structure across the train/test boundary and inflate the test score. This is
the single most important methodological decision in the study.

### Preprocessing

1. Greyscale conversion (colour carries no information in these modalities).
2. Normalisation to `[0, 1]`.
3. Optional histogram equalisation for visual inspection only — not applied before metrics.
4. Border-artefact handling and intensity-clipping treatment, to prevent edge effects during
   filtering.
5. Removal of duplicate and corrupted frames.

### Background vs signal

- **Background:** slowly varying illumination, halo and shadowing artefacts, gradient bias — low
  spatial frequency.
- **Signal:** cell bodies, boundaries and internal structure — higher spatial frequency, spatially
  localised.

The whole problem is that halo artefacts sit close to cell edges and therefore close to the signal
in both space and frequency, which is why a simple high-pass filter is not sufficient.

## 4. Methods compared

### 4.1 Rolling Ball

Background estimated as the surface traced by a ball of fixed radius rolled beneath the intensity
surface, then subtracted. Radius 50 px.

*Known weakness:* the background is estimated from the image itself, so low-contrast structures
are absorbed into the estimate and removed with the background.

### 4.2 Morphological filtering (white top-hat)

Image minus its morphological opening, with an elliptical structuring element of radius 15 px,
followed by a small opening (radius 2 px) to suppress amplified noise.

*Known weakness:* amplifies all high-frequency content, noise included; an over-large element
costs faint structure.

### 4.3 U-Net (retrained)

Five-level encoder/decoder with skip connections, TensorFlow 2.15.

| Setting | Value |
|---|---|
| Architecture | 2× (3×3 conv + BN + ReLU) per block, 2×2 max pool; bilinear upsampling + skip concat |
| Feature maps | 64 → 1024 at bottleneck |
| Output | single-channel sigmoid |
| Input | 256×256, greyscale replicated to 3 channels |
| Optimiser | Adam, β₁ = 0.9, β₂ = 0.999, lr 1e-3 |
| Scheduler | ReduceLROnPlateau, patience 5, factor 0.3 |
| Loss | 0.5 × BCE + 0.5 × Dice |
| Epochs / batch | 80 / 8, early stopping patience 12 |
| Stability | automatic mixed precision, gradient clipping at 1.0 |

**Why BCE + Dice and not BCE alone.** Thin structures are a small fraction of the pixels, so a
model can score well on per-pixel cross-entropy while losing them entirely. Dice measures overlap
and penalises exactly that failure. Equal weighting is the compromise between pixel accuracy and
fine detail.

### 4.4 Augmentation

Horizontal and vertical flips, ±15° rotation, 0.9–1.1× zoom, elastic distortion (p = 0.2), ±10%
brightness and contrast — via Albumentations.

Each of these is a variation the microscope could plausibly produce. Augmentations that are not
physically plausible for the modality teach invariance to things the model will never see.

## 5. Evaluation

### 5.1 The reference image

Real microscopy data has no ground-truth background-free image. Scoring a method against its own
input would measure how much it changed the image, not whether the change was correct. A per-field-
of-view reference `I_ref` is therefore constructed:

1. Convert to float, normalise to `[0, 1]`.
2. Estimate the low-frequency illumination field `L` by morphological opening, radius 80 px.
3. Smooth `L` with a 2D third-order polynomial surface → `L̃`.
4. Flatten: `I − L̃`.
5. Histogram-match back to the original `I`, so intensity statistics remain comparable.

`I_ref` is a defensible proxy for a flat-background image. It is not a ground truth, and the
[limitations](#7-limitations) say so.

### 5.2 Metric settings

PSNR and SSIM between each method's output and `I_ref`. Dynamic range `L = 1`; SSIM with an 11×11
Gaussian window, `K₁ = 0.01`, `K₂ = 0.03`. Reported as mean ± SD on the held-out test set.

### 5.3 Robustness check

The reference construction has free parameters, so a result that depends on them is not a result.

| Parameter varied | Range | Effect |
|---|---|---|
| Opening radius | 60–100 px | < 0.3 dB PSNR, < 0.01 SSIM |
| Gaussian sigma | 30–40 | < 0.3 dB PSNR, < 0.01 SSIM |

Both are smaller than the gap between any two methods, so the ranking is not an artefact of how
the reference was built.

## 6. Reproducing the results

```bash
pip install -r requirements.txt
python src/unet/train.py --data data/miniglioma_usiigaci   # produces models/unet_best.h5
python src/run_comparison.py --data data/miniglioma_usiigaci/test --out results/
```

`run_comparison.py` skips the U-Net and evaluates the classical methods only when no trained
weights are present, so the pipeline can be checked without a GPU.

## 7. Limitations

- One dataset; generalisation across microscopes, cell lines and magnifications is untested.
- The reference image is constructed, not measured.
- The U-Net needs annotated data, which is the practical barrier for labs that have the imaging
  problem but not the annotation budget.
- PSNR and SSIM measure reconstruction fidelity, not downstream segmentation accuracy.
