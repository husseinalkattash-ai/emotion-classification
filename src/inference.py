"""Inference on new images/folders: detect+align -> classify -> CSV output."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from src.data.dataset import build_transforms
from src.data.face_preprocess import detect_and_align
from src.engine.utils import load_checkpoint
from src.models.classifier import EmotionClassifier

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _gather_images(path: Path) -> List[Path]:
    """Return image files for a single-file or directory input."""
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)
    raise FileNotFoundError(f"Input path does not exist: {path}")


def load_model_from_checkpoint(
    checkpoint: str | Path, device: torch.device
) -> Tuple[EmotionClassifier, List[str], int]:
    """Rebuild the model from a checkpoint's saved metadata and weights.

    Returns:
        (model, classes, image_size).
    """
    ckpt = load_checkpoint(checkpoint, map_location=device)
    classes = list(ckpt["classes"])
    model = EmotionClassifier(
        backbone=ckpt.get("config_backbone", "efficientnet_b0"),
        num_classes=ckpt.get("num_classes", len(classes)),
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, classes, int(ckpt.get("image_size", 224))


@torch.no_grad()
def predict(
    image_path_or_dir: str | Path,
    checkpoint: str | Path,
    device: Optional[str | torch.device] = None,
    align_faces: bool = True,
    output_csv: str | Path | None = None,
) -> List[dict]:
    """Predict emotion(s) for a single image or a folder of images.

    Each image is passed through face detection/alignment (for raw images),
    then the evaluation transforms, then the model. Probabilities are the
    softmax over logits.

    Args:
        image_path_or_dir: Single image file or directory of images.
        checkpoint: Path to a trained checkpoint.
        device: Compute device (defaults to CUDA if available).
        align_faces: Run MediaPipe detect+align first; if no face is found,
            fall back to the whole image.
        output_csv: If given, write results to this CSV path.

    Returns:
        A list of result dicts: ``{filename, predicted, confidence, probs}``.
    """
    dev = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, classes, image_size = load_model_from_checkpoint(checkpoint, dev)
    transform = build_transforms(image_size, train=False)

    files = _gather_images(Path(image_path_or_dir))
    if not files:
        logger.warning("No images found at %s", image_path_or_dir)

    results: List[dict] = []
    for fpath in files:
        try:
            with Image.open(fpath) as im:
                image = im.convert("RGB")
        except Exception as exc:
            logger.warning("Could not open %s (%s); skipping.", fpath, exc)
            continue

        if align_faces:
            aligned = detect_and_align(image)
            if aligned is not None:
                image = aligned
            else:
                logger.warning("No face detected in %s; using whole image.", fpath.name)

        tensor = transform(image).unsqueeze(0).to(dev)
        probs = F.softmax(model(tensor), dim=1)[0].cpu()
        pred_idx = int(probs.argmax())
        results.append(
            {
                "filename": str(fpath),
                "predicted": classes[pred_idx],
                "confidence": float(probs[pred_idx]),
                "probs": {classes[i]: float(probs[i]) for i in range(len(classes))},
            }
        )
        logger.info("%s -> %s (%.3f)", fpath.name, classes[pred_idx], probs[pred_idx])

    if output_csv is not None and results:
        _write_csv(results, Path(output_csv), classes)
    return results


def _write_csv(results: Sequence[dict], out_path: Path, classes: Sequence[str]) -> None:
    """Write prediction results to a CSV with one probability column per class."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filename", "predicted", "confidence"] + [f"prob_{c}" for c in classes]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {"filename": r["filename"], "predicted": r["predicted"],
                   "confidence": f"{r['confidence']:.6f}"}
            for c in classes:
                row[f"prob_{c}"] = f"{r['probs'][c]:.6f}"
            writer.writerow(row)
    logger.info("Wrote predictions to %s", out_path)
