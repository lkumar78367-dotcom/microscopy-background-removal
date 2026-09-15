"""Run all three background-removal methods over a test set and write per-image metrics.

Usage:
    python src/run_comparison.py --data data/miniglioma_usiigaci/test --out results/

Writes results/metrics.csv (one row per image per method) and prints the summary table in the
same form as the results table in the README.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from skimage.io import imread

sys.path.insert(0, str(Path(__file__).parent))

from classical import morphological, rolling_ball                      # noqa: E402
from evaluation.metrics import Score, score, summarise                 # noqa: E402
from evaluation.reference import build_reference, match_to_source, normalise  # noqa: E402

IMAGE_SUFFIXES = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
UNET_WEIGHTS = Path("models/unet_best.h5")


def load_greyscale(path: Path) -> np.ndarray:
    img = imread(path)
    if img.ndim == 3:
        img = img.mean(axis=-1)
    return normalise(img)


def unet_predict(image: np.ndarray):
    """Run the trained U-Net. Returns None when no weights are present, so the classical
    methods can still be evaluated without a GPU or a training run."""
    if not UNET_WEIGHTS.exists():
        return None
    from tensorflow.keras.models import load_model  # imported lazily: heavy, and optional

    from unet.train import combined_loss, dice_loss

    model = load_model(UNET_WEIGHTS, custom_objects={
        "combined_loss": combined_loss, "dice_loss": dice_loss,
    })
    stacked = np.repeat(image[None, ..., None], 3, axis=-1)
    prediction = model.predict(stacked, verbose=0)[0, ..., 0]
    return match_to_source(prediction, image)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="directory of test images")
    parser.add_argument("--out", default=Path("results"), type=Path)
    args = parser.parse_args()

    images = sorted(p for p in args.data.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        print(f"No images found under {args.data}", file=sys.stderr)
        return 1

    methods = {
        "Rolling Ball / opening": rolling_ball.remove_background,
        "Morphological (white top-hat)": morphological.remove_background,
        "U-Net (retrained)": unet_predict,
    }

    rows: list[dict] = []
    collected: dict[str, list[Score]] = {name: [] for name in methods}

    for path in images:
        image = load_greyscale(path)
        reference = build_reference(image)
        for name, fn in methods.items():
            output = fn(image)
            if output is None:
                continue
            s = score(output, reference)
            collected[name].append(s)
            rows.append({
                "image": path.name, "method": name,
                "psnr_db": round(s.psnr_db, 3), "ssim": round(s.ssim, 4),
                "meets_criteria": s.meets_criteria,
            })

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "metrics.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(images)} images -> {csv_path}\n")
    print(f"{'Method':<32}{'PSNR (dB)':>16}{'SSIM':>16}{'Meets':>8}")
    for name, scores in collected.items():
        if not scores:
            print(f"{name:<32}{'skipped (no weights)':>40}")
            continue
        st = summarise(scores)
        meets = st["psnr_mean"] >= 26.0 and st["ssim_mean"] >= 0.80
        print(f"{name:<32}"
              f"{st['psnr_mean']:>10.1f} ± {st['psnr_sd']:<4.1f}"
              f"{st['ssim_mean']:>11.2f} ± {st['ssim_sd']:<4.2f}"
              f"{'yes' if meets else 'no':>8}")
    print("\nAcceptance criteria: PSNR >= 26 dB and SSIM >= 0.80 (defined before the experiment).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
