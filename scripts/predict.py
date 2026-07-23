"""Inference entry point: predict on a new image or folder."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import predict  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict emotions on new images")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--input", required=True, help="Image file or directory of images")
    parser.add_argument(
        "--output-csv", default="outputs/predictions.csv", help="Where to write results CSV"
    )
    parser.add_argument(
        "--no-align", action="store_true",
        help="Skip MediaPipe face detection/alignment (use whole image)",
    )
    args = parser.parse_args()

    results = predict(
        image_path_or_dir=args.input,
        checkpoint=args.checkpoint,
        align_faces=not args.no_align,
        output_csv=args.output_csv,
    )
    for r in results:
        print(f"{r['filename']}: {r['predicted']} ({r['confidence']:.3f})")
    print(f"\n{len(results)} image(s) processed. CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
