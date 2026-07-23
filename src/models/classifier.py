"""EmotionClassifier: pretrained torchvision backbone + custom head."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torchvision import models

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)

_SUPPORTED_BACKBONES = ("efficientnet_b0", "resnet50")


class EmotionClassifier(nn.Module):
    """Transfer-learning classifier for facial emotion recognition.

    Loads a pretrained backbone, freezes the first ``freeze_ratio`` of its
    parameters, and attaches a small classification head. Returns raw logits
    (no softmax) so it can be used directly with ``CrossEntropyLoss``.

    Args:
        backbone: One of ``efficientnet_b0`` or ``resnet50``.
        num_classes: Number of output classes.
        pretrained: Load ImageNet-pretrained weights.
        freeze_ratio: Fraction of backbone parameter groups to freeze.
        dropout: Dropout probability in the head.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_ratio: float = 0.75,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if backbone not in _SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone}' (supported: {_SUPPORTED_BACKBONES})"
            )
        self.backbone_name = backbone
        self.features, in_features = self._build_backbone(backbone, pretrained)
        self._freeze_backbone(freeze_ratio)

        # Head: GlobalAvgPool -> Linear(256) -> ReLU -> Dropout -> Linear(num_classes).
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _build_backbone(backbone: str, pretrained: bool) -> tuple[nn.Module, int]:
        """Return the convolutional feature extractor and its output channels."""
        if backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.efficientnet_b0(weights=weights)
            features = net.features  # (B, 1280, H', W')
            in_features = 1280
        else:  # resnet50
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            net = models.resnet50(weights=weights)
            # Drop avgpool + fc; keep conv trunk -> (B, 2048, H', W').
            features = nn.Sequential(*list(net.children())[:-2])
            in_features = 2048
        return features, in_features

    def _freeze_backbone(self, freeze_ratio: float) -> None:
        """Freeze the first ``freeze_ratio`` of backbone parameters."""
        params = list(self.features.parameters())
        n_freeze = int(len(params) * freeze_ratio)
        for p in params[:n_freeze]:
            p.requires_grad = False
        logger.info(
            "Froze %d/%d backbone parameter tensors (freeze_ratio=%.2f)",
            n_freeze, len(params), freeze_ratio,
        )

    @property
    def target_layer(self) -> nn.Module:
        """Last conv-bearing module of the backbone (for Grad-CAM)."""
        if self.backbone_name == "efficientnet_b0":
            return self.features[-1]
        # resnet50: features is a Sequential; last block is layer4.
        return self.features[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape ``(batch, num_classes)``."""
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)


def build_model(config: "Config") -> EmotionClassifier:
    """Construct an :class:`EmotionClassifier` from a :class:`Config`."""
    return EmotionClassifier(
        backbone=config.model.backbone,
        num_classes=config.data.num_classes,
        pretrained=config.model.pretrained,
        freeze_ratio=config.model.freeze_ratio,
        dropout=config.model.dropout,
    )
