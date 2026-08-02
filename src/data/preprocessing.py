"""Image preprocessing: normalization constants and the transform pipelines.

This module owns everything that happens to an image *after* it is read from
disk and *before* it reaches the model:

* ImageNet normalization statistics (required by the pretrained backbones).
* The training pipeline, which adds augmentation to fight overfitting.
* The evaluation pipeline, which is deterministic so results are reproducible.

Face detection and alignment for raw inference images live in
``face_preprocess.py``; dataset reading and label mapping live in ``dataset.py``.
"""
from __future__ import annotations

from typing import Callable, List

from torchvision import transforms

# ImageNet normalization (pretrained backbones expect this).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(
    image_size: int,
    train: bool,
    horizontal_flip: bool = True,
    rotation_deg: float = 10.0,
    brightness: float = 0.1,
    contrast: float = 0.0,
    saturation: float = 0.0,
    rrc_scale_min: float = 1.0,
    random_erasing: float = 0.0,
) -> Callable:
    """Return torchvision transforms for train (augmented) or eval (deterministic).

    Args:
        image_size: Target square size in pixels.
        train: If True, apply augmentation; otherwise deterministic resize only.
        horizontal_flip: Enable random horizontal flip (train only).
        rotation_deg: Max random rotation in degrees (train only).
        brightness, contrast, saturation: ColorJitter factors (train only).
        rrc_scale_min: If < 1.0, use RandomResizedCrop with this minimum area
            scale (a mild random zoom/crop); 1.0 disables it (plain resize).
        random_erasing: Probability for RandomErasing (0 disables). Applied on
            the tensor after normalization.

    Returns:
        A composed torchvision transform.
    """
    if train:
        if rrc_scale_min < 1.0:
            first: Callable = transforms.RandomResizedCrop(
                image_size, scale=(rrc_scale_min, 1.0), ratio=(0.9, 1.1)
            )
        else:
            first = transforms.Resize((image_size, image_size))
        ops: List[Callable] = [first]
        if horizontal_flip:
            ops.append(transforms.RandomHorizontalFlip())
        if rotation_deg:
            ops.append(transforms.RandomRotation(rotation_deg))
        if brightness or contrast or saturation:
            ops.append(
                transforms.ColorJitter(
                    brightness=brightness, contrast=contrast, saturation=saturation
                )
            )
        ops += [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        if random_erasing > 0:
            ops.append(transforms.RandomErasing(p=random_erasing))
        return transforms.Compose(ops)

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
