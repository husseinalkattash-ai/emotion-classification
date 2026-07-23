"""Tests for the EmotionClassifier model."""
from __future__ import annotations

import pytest
import torch

from src.models.classifier import EmotionClassifier


@pytest.mark.parametrize("backbone", ["efficientnet_b0", "resnet50"])
def test_forward_output_shape(backbone: str) -> None:
    model = EmotionClassifier(backbone=backbone, num_classes=7, pretrained=False)
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 7)


def test_freeze_ratio_freezes_params() -> None:
    model = EmotionClassifier(backbone="efficientnet_b0", pretrained=False, freeze_ratio=1.0)
    assert all(not p.requires_grad for p in model.features.parameters())
    # Head must remain trainable.
    assert any(p.requires_grad for p in model.head.parameters())


def test_target_layer_exists() -> None:
    model = EmotionClassifier(backbone="efficientnet_b0", pretrained=False)
    assert isinstance(model.target_layer, torch.nn.Module)


def test_unsupported_backbone_raises() -> None:
    with pytest.raises(ValueError):
        EmotionClassifier(backbone="vgg16", pretrained=False)
