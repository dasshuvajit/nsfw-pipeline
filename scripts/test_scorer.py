#!/usr/bin/env python3
"""Smoke-test the ImageScorer.

Scores a single image end-to-end and prints every component score + flag.
First run is slow: CLIP ViT-L/14 + the LAION MLP head load (~1GB) and
insightface downloads buffalo_l (~300MB) into ~/.insightface/models/.
Subsequent runs are fast.

Pre-flight checklist:
  - `pip install -r requirements.txt` (with the Image/Scoring block enabled).
  - `models/aesthetic/sac+logos+ava1-l14-linearMSE.pth` exists. Download from
    https://github.com/christophschuhmann/improved-aesthetic-predictor
    — see PROJECT_GUIDE.md → "Setting up the image scorer".
  - Network access on first run so insightface can fetch buffalo_l.

Usage:
    python scripts/test_scorer.py path/to/image.png
    python scripts/test_scorer.py output/test/20260407_220705_test_0.png
    python scripts/test_scorer.py img.png --device cpu
    python scripts/test_scorer.py img.png --mlp-weights /custom/path.pth
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make src/ importable when running this file as a plain script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring.image_scorer import (  # noqa: E402
    ImageScorer,
    ScorerError,
    ScorerModelError,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image_path",
        help="Path to the image to score (PNG/JPG/etc.).",
    )
    parser.add_argument(
        "--mlp-weights",
        default=None,
        help="Override aesthetic MLP checkpoint path "
        "(default: models/aesthetic/sac+logos+ava1-l14-linearMSE.pth).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default=None,
        help="Force CLIP device. Default: auto (mps on Mac, cuda on Linux+GPU, "
        "cpu otherwise).",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path).expanduser()
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}", file=sys.stderr)
        return 1

    # Build the scorer. This is cheap — models load on first .score() call.
    scorer_kwargs = {}
    if args.mlp_weights:
        scorer_kwargs["aesthetic_weights"] = args.mlp_weights
    if args.device:
        scorer_kwargs["device"] = args.device
    scorer = ImageScorer(**scorer_kwargs)

    print(f"Scoring {image_path} ...")
    print("(first run loads CLIP + aesthetic MLP + insightface, ~30s)")
    start = time.time()
    try:
        result = scorer.score(image_path)
    except ScorerModelError as exc:
        print(f"\nERROR (model load failed): {exc}", file=sys.stderr)
        return 2
    except ScorerError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 3
    elapsed = time.time() - start

    print(f"\nOK — scored in {elapsed:.1f}s")
    print(f"  composite         {result['composite']:.3f}  (want >= 0.55)")
    print(f"  aesthetic         {result['aesthetic']:.2f}   / 10  (want > 5.5)")
    print(f"  blur (laplacian)  {result['blur']:.1f}       (want > 100)")
    print(f"  face_confidence   {result['face_confidence']:.3f}     (insightface det_score)")
    flags = result["flags"]
    if flags:
        print(f"  flags             {flags}")
    else:
        print("  flags             (none)")

    # Exit 4 if any flag fires so CI / shell callers can detect low-quality
    # outputs without parsing stdout. Zero = clean image.
    return 0 if not flags else 4


if __name__ == "__main__":
    sys.exit(main())
