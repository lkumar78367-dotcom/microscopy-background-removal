# Background Removal for Phase Contrast and DIC Microscopy Images

A reproducible comparison of three background-removal methods for label-free microscopy —
**Rolling Ball**, **morphological white top-hat**, and a **retrained U-Net** — evaluated against
acceptance criteria defined *before* the experiment was run.

MSc Biomedical Engineering dissertation, University of Strathclyde, 2024–2025.
**Lokeshkumar Rajagopal** · [LinkedIn](https://linkedin.com/in/lokesh29)

---

## Why this problem matters

Phase contrast and DIC microscopy image living, unstained cells — no dyes, no phototoxicity, no
fixation. That is exactly why they are used for live-cell work, and exactly why they are hard to
analyse automatically. Both modalities convert optical path differences into intensity, so they
also convert *illumination* differences into intensity: uneven lighting, halo artefacts around
cell edges, and a slowly varying gradient across the field of view.

Downstream segmentation then measures the illumination as much as the biology. Background removal
is the preprocessing step that decides whether everything after it is trustworthy.

The trade-off runs in both directions. Remove too little and the gradient survives into the
segmentation. Remove too much and faint cellular structure goes with it. This project quantifies
where three common methods sit on that trade-off.

---

## Results

All figures are mean ± SD on a held-out test set, computed against a per-field-of-view reference
image (see [Evaluation protocol](#evaluation-protocol)).

| Method | PSNR (dB) | SSIM | Meets criteria |
|---|---|---|---|
| Rolling Ball / opening | 24.1 ± 1.3 | 0.72 ± 0.04 | No |
| Morphological filtering (white top-hat) | 25.6 ± 1.1 | 0.78 ± 0.03 | No |
| **U-Net (retrained)** | **28.4 ± 0.9** | **0.86 ± 0.02** | **Yes** |

**Acceptance criteria, fixed in advance: PSNR ≥ 26 dB and SSIM ≥ 0.80.**

Only the U-Net cleared both. The classical methods behaved as expected from their mechanics rather
than failing randomly:

- **Rolling Ball** corrects global illumination well, but it estimates the background from the
  image itself, so low-contrast structures are absorbed into the background estimate and removed
  with it. Good gradient correction, poor retention of faint detail.
- **White top-hat** sharpens boundaries and separates structures better, but it amplifies
  high-frequency content indiscriminately — including noise — and over-filtering costs weak cell
  detail. Better SSIM than Rolling Ball, still short of the threshold.
- **U-Net** learns the foreground/background distinction from examples rather than assuming a
  model of the background, which is why it holds structure while suppressing the gradient. The
  cost is annotated training data and compute, which is a real constraint, not a footnote.

The honest conclusion is not "deep learning wins". It is that the classical methods are adequate
and far cheaper when the correction needed is a simple illumination gradient, and that the U-Net
earns its cost only when structural fidelity is the binding requirement.

---

## Evaluation protocol

The measurement method matters more than the numbers, so it is specified here in full.

**The reference image problem.** There is no ground-truth "background-free" image for real
microscopy data. Comparing a method's output to the raw input measures how much it changed the
image, not how correct the change was. So a per-field-of-view reference `I_ref` is constructed and
every method is scored against it:

1. Convert to float and normalise to `[0, 1]`.
2. Estimate the low-frequency illumination field `L` by morphological opening (radius 80 px).
3. Smooth `L` with a 2D third-order polynomial surface to give `L̃`.
4. Form the flattened reference from `I − L̃`.
5. Histogram-match `I_ref` back to the original `I` so intensity statistics are comparable.

**Metric settings.** PSNR and SSIM are computed between each method's output and `I_ref`, with
dynamic range `L = 1`, an 11×11 Gaussian window, `K₁ = 0.01`, `K₂ = 0.03`. Reported as mean ± SD.

**Robustness check.** The reference construction has free parameters, so a result that depends on
them is not a result. Varying the opening radius over 60–100 px and the Gaussian sigma over 30–40
changed PSNR by less than 0.3 dB and SSIM by less than 0.01 across the test set — smaller than the
gap between any two methods, so the ranking is not an artefact of the reference.

---

## Method summary

**Dataset.** Integrated MiniGlioma–Usiigaci microscopy images, 256×256, split 70/15/15 **by field
of view** — not by image — so that patches from the same FOV cannot appear in both training and
test. Preprocessing: greyscale conversion, normalisation to `[0, 1]`, optional histogram
equalisation, border-artefact handling, and removal of duplicate and corrupted frames.

**Classical methods.** Rolling Ball background subtraction, and morphological operations
(erosion, dilation, opening, closing, white top-hat) via OpenCV, with a smaller structuring
element for local refinement than for the background estimate.

**U-Net.** Five-level encoder/decoder with skip connections. Encoder blocks: two 3×3 convolutions,
batch normalisation, ReLU, 2×2 max pooling. Decoder: bilinear upsampling with concatenation of the
matching encoder features. 64 feature maps at level 1 rising to 1024 at the bottleneck; single-
channel sigmoid output. TensorFlow 2.15.

**Training.** Adam (β₁ = 0.9, β₂ = 0.999), learning rate 1e-3, `ReduceLROnPlateau`
(patience 5, factor 0.3). Loss: binary cross-entropy + Dice at 50:50, balancing pixel accuracy
against overlap of fine structure. 80 epochs, batch size 8, early stopping (patience 12),
automatic mixed precision, gradient clipping at 1.0. Augmentation via Albumentations: horizontal
and vertical flips, ±15° rotation, 0.9–1.1× zoom, elastic distortion (p = 0.2), ±10% brightness
and contrast.

---

## Repository layout

```
src/
  classical/rolling_ball.py     Rolling Ball background subtraction
  classical/morphological.py    Opening, closing, white top-hat
  unet/model.py                 5-level U-Net architecture
  unet/train.py                 Training loop, loss, callbacks
  unet/augment.py               Albumentations pipeline
  evaluation/reference.py       Per-FOV reference image construction
  evaluation/metrics.py         PSNR and SSIM with the settings above
  run_comparison.py             Runs all three methods, writes results/metrics.csv
docs/
  METHOD.md                     Full protocol and acceptance criteria
  RESULTS.md                    Results table and interpretation
results/
  metrics.csv                   Per-image metrics
```

## Running it

```bash
pip install -r requirements.txt
python src/run_comparison.py --data data/miniglioma_usiigaci --out results/
```

The U-Net path expects trained weights at `models/unet_best.h5`; `src/unet/train.py` produces them.

---

## Limitations

Stated plainly, because a portfolio that only reports its strengths is not evidence of judgement:

- One dataset. Generalisation to other microscopes, cell lines and magnifications is untested, and
  generalisability was one of the research questions this work could only partly answer.
- The reference image is constructed, not measured. It is a defensible proxy for a flat-background
  ground truth, not a ground truth.
- The U-Net requires annotated data, which is the practical barrier to using it in labs that have
  the imaging problem but not the annotation budget.
- PSNR and SSIM measure reconstruction fidelity, not downstream segmentation accuracy. A method
  can win on both and still not be the best choice for a specific segmentation task.

---

## Licence

MIT — see [LICENSE](LICENSE).
