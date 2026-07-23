"""Evaluation: metrics, classification report, confusion matrix."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend for containers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model over ``loader`` and return (y_true, y_pred) arrays.

    Args:
        tta: If True, apply test-time augmentation by averaging softmax
            probabilities over the image and its horizontal flip.
    """
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        probs = torch.softmax(model(images), dim=1)
        if tta:
            probs = probs + torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
        preds = probs.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())
    return np.asarray(y_true), np.asarray(y_pred)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[str]
) -> Dict[str, object]:
    """Compute accuracy, macro-F1, and the per-class classification report.

    Returns:
        Dict with keys ``accuracy``, ``macro_f1``, and ``report`` (dict form).
    """
    labels = list(range(len(classes)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "report": classification_report(
            y_true, y_pred, labels=labels, target_names=list(classes),
            output_dict=True, zero_division=0,
        ),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[str],
    out_path: str | Path,
    normalize: bool = True,
) -> Path:
    """Plot and save a confusion-matrix heatmap.

    Args:
        y_true, y_pred: Ground-truth and predicted 0-indexed labels.
        classes: Class names in output order.
        out_path: Where to save the PNG.
        normalize: Row-normalize (recall per true class) if True.

    Returns:
        The path the figure was saved to.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    fmt = "d"
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        fmt = ".2f"

    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=list(classes), yticklabels=list(classes), cbar=True,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved confusion matrix to %s", out_path)
    return out_path


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    classes: Sequence[str],
    output_dir: str | Path | None = None,
    print_report: bool = True,
    tta: bool = False,
) -> Dict[str, object]:
    """Evaluate a model: metrics + optional confusion matrix + printed report.

    Args:
        model: Trained model.
        loader: DataLoader over the evaluation split.
        device: Compute device.
        classes: Class names in output order.
        output_dir: If given, save ``confusion_matrix.png`` there.
        print_report: Print the sklearn classification report.
        tta: Enable horizontal-flip test-time augmentation.

    Returns:
        The metrics dict from :func:`compute_metrics`.
    """
    y_true, y_pred = collect_predictions(model, loader, device, tta=tta)
    metrics = compute_metrics(y_true, y_pred, classes)

    if print_report:
        report_txt = classification_report(
            y_true, y_pred, labels=list(range(len(classes))),
            target_names=list(classes), zero_division=0,
        )
        logger.info("Classification report:\n%s", report_txt)
        print(report_txt)
        print(f"Accuracy: {metrics['accuracy']:.4f}  Macro-F1: {metrics['macro_f1']:.4f}")

    if output_dir is not None:
        plot_confusion_matrix(
            y_true, y_pred, classes, Path(output_dir) / "confusion_matrix.png"
        )
    return metrics
