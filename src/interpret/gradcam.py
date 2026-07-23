"""Grad-CAM visualizations over the backbone's last conv layer."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization -> HxWx3 float array in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (tensor.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def compute_gradcam_overlay(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layer: torch.nn.Module,
    class_idx: int | None = None,
) -> np.ndarray:
    """Compute a single Grad-CAM overlay in memory (no disk I/O).

    Args:
        model: Trained model.
        input_tensor: A single preprocessed image, shape ``(1, 3, H, W)``.
        target_layer: Conv layer to attribute against (backbone last conv).
        class_idx: Target class; defaults to the model's argmax prediction.

    Returns:
        An HxWx3 uint8 RGB overlay (heatmap blended on the input image).
    """
    targets = [ClassifierOutputTarget(class_idx)] if class_idx is not None else None
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale = cam(input_tensor=input_tensor, targets=targets)[0]
    rgb = _denormalize(input_tensor[0])
    return show_cam_on_image(rgb, grayscale, use_rgb=True)


def generate_gradcam(
    model: torch.nn.Module,
    images: torch.Tensor,
    output_dir: str | Path,
    classes: Sequence[str],
    target_layer: torch.nn.Module,
    device: torch.device,
    filenames: Sequence[str] | None = None,
) -> List[Path]:
    """Generate and save Grad-CAM overlays for a batch of preprocessed images.

    Args:
        model: Trained model.
        images: Batch tensor ``(N, 3, H, W)`` (ImageNet-normalized).
        output_dir: Directory to write overlay PNGs (created if needed).
        classes: Class names in output order.
        target_layer: Conv layer to attribute against (backbone last conv).
        device: Compute device.
        filenames: Optional output basenames (without extension).

    Returns:
        List of written PNG paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    images = images.to(device)
    with torch.no_grad():
        preds = model(images).argmax(dim=1).cpu().tolist()

    cam = GradCAM(model=model, target_layers=[target_layer])
    written: List[Path] = []
    for i in range(images.size(0)):
        pred = preds[i]
        targets = [ClassifierOutputTarget(pred)]
        grayscale = cam(input_tensor=images[i : i + 1], targets=targets)[0]
        rgb = _denormalize(images[i])
        overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)

        base = filenames[i] if filenames and i < len(filenames) else f"sample_{i:03d}"
        out_path = output_dir / f"{base}_gradcam_{classes[pred]}.png"
        Image.fromarray(overlay).save(out_path)
        written.append(out_path)

    logger.info("Saved %d Grad-CAM overlays to %s", len(written), output_dir)
    return written
