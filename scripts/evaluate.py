"""Evaluation entry point: classification report + confusion matrix."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.dataset import build_datasets  # noqa: E402
from src.engine.evaluate import evaluate_model  # noqa: E402
from src.engine.utils import load_checkpoint  # noqa: E402
from src.models.classifier import build_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained emotion classifier")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--split", default="test", choices=["test", "val"], help="Split to evaluate")
    parser.add_argument(
        "--gradcam", action="store_true",
        help="Also generate Grad-CAM overlays for a sample of the split into outputs/gradcam/",
    )
    parser.add_argument(
        "--gradcam-samples", type=int, default=16, help="Number of Grad-CAM samples"
    )
    parser.add_argument(
        "--tta", action="store_true",
        help="Enable horizontal-flip test-time augmentation",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds = build_datasets(
        root=cfg.data.root,
        aligned_dir=cfg.data.aligned_dir,
        label_file=cfg.data.label_file,
        classes=cfg.data.classes,
        image_size=cfg.data.image_size,
        val_split=cfg.data.val_split,
        seed=cfg.project.seed,
        data_format=cfg.data.format,
        train_dir=cfg.data.train_dir,
        test_dir=cfg.data.test_dir,
    )
    dataset = test_ds if args.split == "test" else val_ds
    loader = DataLoader(
        dataset, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=True,
    )

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    logger.info(
        "Evaluating on '%s' split (%d samples)%s",
        args.split, len(dataset), " with TTA" if args.tta else "",
    )
    evaluate_model(
        model, loader, device, cfg.data.classes,
        output_dir=cfg.output.dir, print_report=True, tta=args.tta,
    )

    if args.gradcam:
        from src.interpret.gradcam import generate_gradcam  # local import (optional dep)

        n = min(args.gradcam_samples, len(dataset))
        images = torch.stack([dataset[i][0] for i in range(n)])
        generate_gradcam(
            model=model,
            images=images,
            output_dir=Path(cfg.output.dir) / "gradcam",
            classes=cfg.data.classes,
            target_layer=model.target_layer,
            device=device,
            filenames=[Path(dataset.samples[i][0]).stem for i in range(n)],
        )


if __name__ == "__main__":
    main()
