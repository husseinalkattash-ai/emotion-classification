"""RAF-DB dataset reading, splitting, and the (critical) label mapping.

RAF-DB basic emotions use an official 1-indexed labeling. We map those IDs to
the model's 0-indexed class order (``config.data.classes``) via an explicit
mapping — never assume alphabetical ordering.

Image preprocessing (normalization + augmentation pipelines) lives in
``preprocessing.py``; it is re-exported here for backward compatibility.
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.data.preprocessing import IMAGENET_MEAN, IMAGENET_STD, build_transforms

logger = logging.getLogger(__name__)

__all__ = [
    "IMAGENET_MEAN", "IMAGENET_STD", "build_transforms",
    "RAFDB_ID_TO_NAME", "RAFDB_LABEL_MAP", "build_label_map",
    "RAFDBDataset", "build_datasets",
]

# Official RAF-DB basic-emotion labeling: 1-indexed ID -> emotion name.
# DO NOT reorder — this is fixed by the dataset authors.
RAFDB_ID_TO_NAME: Dict[int, str] = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happy",
    5: "sad",
    6: "angry",
    7: "neutral",
}


def build_label_map(classes: Sequence[str]) -> Dict[int, int]:
    """Build ``RAFDB_LABEL_MAP``: RAF-DB 1-indexed ID -> 0-indexed model index.

    The model index is the position of the emotion name within ``classes``
    (the config output order). Raises if any RAF-DB emotion is missing from
    ``classes``.

    Args:
        classes: The config class list defining the model's output order.

    Returns:
        Dict mapping each RAF-DB 1-indexed ID to its 0-indexed model index.
    """
    name_to_index = {name: idx for idx, name in enumerate(classes)}
    label_map: Dict[int, int] = {}
    for rafdb_id, name in RAFDB_ID_TO_NAME.items():
        if name not in name_to_index:
            raise ValueError(
                f"RAF-DB emotion '{name}' (id {rafdb_id}) not found in config classes: "
                f"{list(classes)}"
            )
        label_map[rafdb_id] = name_to_index[name]
    return label_map


# Default map assuming the canonical config order
# [surprise, fear, disgust, happy, sad, angry, neutral].
RAFDB_LABEL_MAP: Dict[int, int] = build_label_map(
    [RAFDB_ID_TO_NAME[i] for i in range(1, 8)]
)


def _aligned_filename(original: str) -> str:
    """Map a label-file name (``train_00001.jpg``) to its aligned crop name
    (``train_00001_aligned.jpg``)."""
    stem, dot, ext = original.rpartition(".")
    if not dot:  # no extension present
        return f"{original}_aligned"
    return f"{stem}_aligned.{ext}"


def _parse_label_file(label_file: Path) -> List[Tuple[str, int]]:
    """Parse ``list_patition_label.txt`` into (filename, 1-indexed label) tuples."""
    if not label_file.is_file():
        raise FileNotFoundError(f"Label file not found: {label_file}")
    entries: List[Tuple[str, int]] = []
    with label_file.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                logger.warning("Skipping malformed label line %d: %r", lineno, line)
                continue
            filename, label_str = parts
            entries.append((filename, int(label_str)))
    return entries


class RAFDBDataset(Dataset):
    """RAF-DB basic-emotion dataset backed by the pre-aligned face crops.

    Args:
        root: Dataset root (contains ``basic/``).
        aligned_dir: Relative path to aligned crops under ``root``.
        label_file: Relative path to the partition/label file under ``root``.
        classes: Config class list (defines the 0-indexed output order).
        transform: Transform applied to each PIL image.
        samples: Optional pre-filtered list of (aligned_path, label) pairs. When
            provided, ``root``/``label_file`` parsing is skipped (used by the
            train/val split helper).
    """

    def __init__(
        self,
        root: str | Path | None = None,
        aligned_dir: str = "basic/Image/aligned",
        label_file: str = "basic/EmoLabel/list_patition_label.txt",
        classes: Sequence[str] | None = None,
        transform: Callable | None = None,
        samples: List[Tuple[Path, int]] | None = None,
    ) -> None:
        self.classes = list(classes) if classes is not None else [
            RAFDB_ID_TO_NAME[i] for i in range(1, 8)
        ]
        self.label_map = build_label_map(self.classes)
        self.transform = transform

        if samples is not None:
            self.samples = samples
            return

        if root is None:
            raise ValueError("Either `root` or `samples` must be provided.")

        self.root = Path(root)
        self.aligned_path = self.root / aligned_dir
        entries = _parse_label_file(self.root / label_file)

        self.samples = []
        missing = 0
        for filename, rafdb_id in entries:
            img_path = self.aligned_path / _aligned_filename(filename)
            if not img_path.is_file():
                missing += 1
                continue
            if rafdb_id not in self.label_map:
                logger.warning("Unknown RAF-DB label id %s for %s", rafdb_id, filename)
                continue
            self.samples.append((img_path, self.label_map[rafdb_id]))
        if missing:
            logger.warning("%d labeled images not found under %s", missing, self.aligned_path)
        if not self.samples:
            raise RuntimeError(f"No usable samples found under {self.aligned_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[index]
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def labels(self) -> List[int]:
        """The 0-indexed label for every sample, in order."""
        return [label for _, label in self.samples]

    def compute_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights over this split's label distribution.

        Returns:
            Float tensor of length ``num_classes`` (normalized to mean 1.0).
            Classes absent from the split receive weight 0.
        """
        num_classes = len(self.classes)
        counts = Counter(self.labels)
        total = len(self.samples)
        weights = torch.zeros(num_classes, dtype=torch.float32)
        for cls in range(num_classes):
            n = counts.get(cls, 0)
            weights[cls] = total / (num_classes * n) if n > 0 else 0.0
        present = weights[weights > 0]
        if present.numel() > 0:
            weights[weights > 0] = present / present.mean()
        return weights

    def sample_weights(self) -> torch.Tensor:
        """Per-sample weights (inverse class frequency) for a balanced sampler.

        Feeding these to ``WeightedRandomSampler`` makes every class appear
        roughly equally often per epoch, oversampling minority classes.

        Returns:
            Double tensor of length ``len(self)`` — one weight per sample.
        """
        counts = Counter(self.labels)
        return torch.tensor(
            [1.0 / counts[label] for label in self.labels], dtype=torch.double
        )


def _split_by_prefix(
    root: Path, aligned_dir: str, label_file: str, label_map: Dict[int, int]
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """Split all labeled samples into (train, test) by filename prefix."""
    aligned_path = root / aligned_dir
    entries = _parse_label_file(root / label_file)
    train: List[Tuple[Path, int]] = []
    test: List[Tuple[Path, int]] = []
    missing = 0
    for filename, rafdb_id in entries:
        img_path = aligned_path / _aligned_filename(filename)
        if not img_path.is_file():
            missing += 1
            continue
        if rafdb_id not in label_map:
            logger.warning("Unknown RAF-DB label id %s for %s", rafdb_id, filename)
            continue
        sample = (img_path, label_map[rafdb_id])
        if filename.startswith("train_"):
            train.append(sample)
        elif filename.startswith("test_"):
            test.append(sample)
        else:
            logger.warning("Filename %s has no train_/test_ prefix; skipping", filename)
    if missing:
        logger.warning("%d labeled images not found under %s", missing, aligned_path)
    return train, test


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_class_folder(
    folder_name: str, label_map: Dict[int, int], name_to_index: Dict[str, int]
) -> int | None:
    """Resolve an ImageFolder class-directory name to a 0-indexed model label.

    Folder names may be RAF-DB 1-indexed IDs (``"1"``..``"7"``) or emotion names
    (``"happy"``). Returns None if the name matches neither.
    """
    key = folder_name.strip()
    if key.isdigit():
        return label_map.get(int(key))
    return name_to_index.get(key.lower())


def _scan_imagefolder(
    split_root: Path, label_map: Dict[int, int], name_to_index: Dict[str, int]
) -> List[Tuple[Path, int]]:
    """Collect (image_path, 0-indexed label) from an ImageFolder-style split.

    Expects ``split_root/<class>/<image>`` where ``<class>`` is a RAF-DB id
    (1..7) or an emotion name. Used for the Kaggle mirror layout.
    """
    if not split_root.is_dir():
        raise FileNotFoundError(f"ImageFolder split directory not found: {split_root}")
    samples: List[Tuple[Path, int]] = []
    for class_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        label = _resolve_class_folder(class_dir.name, label_map, name_to_index)
        if label is None:
            logger.warning("Unrecognized class folder '%s' under %s; skipping", class_dir.name, split_root)
            continue
        for img in class_dir.iterdir():
            if img.is_file() and img.suffix.lower() in _IMAGE_EXTS:
                samples.append((img, label))
    if not samples:
        raise RuntimeError(f"No images found under {split_root}")
    return samples


def _scan_extra_imagefolder(
    root: str | Path,
    train_dir: str,
    classes: Sequence[str],
    class_alias: Dict[str, str] | None,
) -> List[Tuple[Path, int]]:
    """Scan an auxiliary ImageFolder dataset, mapping its classes to ours.

    Args:
        root: Auxiliary dataset root.
        train_dir: Split subdirectory holding class folders.
        classes: Our canonical class list (0-indexed output order).
        class_alias: Maps a source folder name (case-insensitive) to one of our
            class names, or to ""/"drop"/"none"/"ignore" to exclude it. Folders
            not in the alias map are matched by name (or RAF-DB id) directly.

    Returns:
        List of (image_path, 0-indexed label) for classes present in ``classes``.
    """
    split_root = Path(root) / train_dir
    if not split_root.is_dir():
        raise FileNotFoundError(f"Extra dataset split directory not found: {split_root}")
    name_to_index = {name.lower(): idx for idx, name in enumerate(classes)}
    alias = {k.strip().lower(): (v or "").strip().lower() for k, v in (class_alias or {}).items()}
    _DROP = {"", "drop", "none", "ignore"}

    samples: List[Tuple[Path, int]] = []
    dropped_folders: List[str] = []
    for class_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        key = class_dir.name.strip().lower()
        if key in alias:
            target = alias[key]
            if target in _DROP:
                dropped_folders.append(class_dir.name)
                continue
            key = target
        # Resolve to a model index: emotion name, or RAF-DB numeric id.
        if key.isdigit():
            name = RAFDB_ID_TO_NAME.get(int(key))
            idx = name_to_index.get(name) if name else None
        else:
            idx = name_to_index.get(key)
        if idx is None:
            logger.warning("Extra dataset: unmapped class folder '%s' (%s); skipping",
                           class_dir.name, split_root)
            continue
        for img in class_dir.iterdir():
            if img.is_file() and img.suffix.lower() in _IMAGE_EXTS:
                samples.append((img, idx))
    if dropped_folders:
        logger.info("Extra dataset: dropped folders %s (aliased out)", dropped_folders)
    return samples


def build_datasets(
    root: str | Path,
    aligned_dir: str,
    label_file: str,
    classes: Sequence[str],
    image_size: int,
    val_split: float,
    seed: int = 42,
    horizontal_flip: bool = True,
    rotation_deg: float = 10.0,
    brightness: float = 0.1,
    contrast: float = 0.0,
    saturation: float = 0.0,
    rrc_scale_min: float = 1.0,
    random_erasing: float = 0.0,
    data_format: str = "rafdb_official",
    train_dir: str = "train",
    test_dir: str = "test",
    extra_datasets: List[Dict[str, object]] | None = None,
) -> Tuple[RAFDBDataset, RAFDBDataset, RAFDBDataset]:
    """Build stratified train/val/test datasets from RAF-DB.

    Two on-disk layouts are supported via ``data_format``:

    * ``rafdb_official``: the official layout — aligned crops under
      ``aligned_dir`` + a ``label_file`` partition list. Train/test come from
      the ``train_``/``test_`` filename prefixes.
    * ``imagefolder``: the Kaggle-mirror layout —
      ``root/<train_dir>/<class>/*.jpg`` and ``root/<test_dir>/<class>/*.jpg``,
      where ``<class>`` is a RAF-DB id (1..7) or an emotion name.

    In both cases a validation set is carved out of the train partition,
    stratified by class (never a random test split).

    TODO: When later training on real *application* images (not RAF-DB), switch
    to a subject-level split so that images of the same person never appear in
    both train and val/test — otherwise identity leakage inflates metrics.

    Args:
        root: Dataset root.
        aligned_dir: Relative path to aligned crops (rafdb_official).
        label_file: Relative path to the label/partition file (rafdb_official).
        classes: Config class list (0-indexed output order).
        image_size: Square image size.
        val_split: Fraction of the train partition held out for validation.
        seed: RNG seed for the stratified split.
        horizontal_flip, rotation_deg, brightness: Train augmentation params.
        data_format: ``rafdb_official`` or ``imagefolder``.
        train_dir, test_dir: Split subdirectories (imagefolder only).
        extra_datasets: Optional list of auxiliary training datasets to pool
            into the train split ONLY (val/test stay RAF-DB). Each entry is a
            dict with keys ``root``, ``train_dir``, ``class_alias`` (and
            ``name`` for logging). See :func:`_scan_extra_imagefolder`.

    Returns:
        (train_ds, val_ds, test_ds) with appropriate transforms attached.
    """
    root = Path(root)
    label_map = build_label_map(classes)
    if data_format == "imagefolder":
        name_to_index = {name.lower(): idx for idx, name in enumerate(classes)}
        train_samples = _scan_imagefolder(root / train_dir, label_map, name_to_index)
        test_samples = _scan_imagefolder(root / test_dir, label_map, name_to_index)
    elif data_format == "rafdb_official":
        train_samples, test_samples = _split_by_prefix(root, aligned_dir, label_file, label_map)
    else:
        raise ValueError(
            f"Unsupported data_format '{data_format}' "
            "(supported: rafdb_official, imagefolder)"
        )

    train_tf = build_transforms(
        image_size, train=True,
        horizontal_flip=horizontal_flip, rotation_deg=rotation_deg, brightness=brightness,
        contrast=contrast, saturation=saturation,
        rrc_scale_min=rrc_scale_min, random_erasing=random_erasing,
    )
    eval_tf = build_transforms(image_size, train=False)

    if val_split > 0 and len(train_samples) > 1:
        labels = [lbl for _, lbl in train_samples]
        num_present = len(set(labels))
        n_val = max(1, round(val_split * len(train_samples)))
        # sklearn requires the val set to be >= number of classes to stratify,
        # and every class to have >= 2 samples. Fall back to a random split
        # otherwise (only relevant for very small datasets).
        can_stratify = min(Counter(labels).values()) >= 2 and n_val >= num_present
        stratify = labels if can_stratify else None
        tr_idx, val_idx = train_test_split(
            range(len(train_samples)),
            test_size=val_split,
            random_state=seed,
            stratify=stratify,
        )
        tr = [train_samples[i] for i in tr_idx]
        val = [train_samples[i] for i in val_idx]
    else:
        tr, val = train_samples, []

    # Pool auxiliary datasets into the TRAIN split only (val/test stay RAF-DB
    # so metrics remain comparable to the RAF-DB-only baselines).
    n_rafdb_train = len(tr)
    for ds in extra_datasets or []:
        extra = _scan_extra_imagefolder(
            root=ds["root"],
            train_dir=str(ds.get("train_dir", "train")),
            classes=classes,
            class_alias=ds.get("class_alias"),  # type: ignore[arg-type]
        )
        logger.info("Extra dataset '%s': +%d train samples", ds.get("name", "extra"), len(extra))
        tr = tr + extra

    train_ds = RAFDBDataset(classes=classes, transform=train_tf, samples=tr)
    val_ds = RAFDBDataset(classes=classes, transform=eval_tf, samples=val)
    test_ds = RAFDBDataset(classes=classes, transform=eval_tf, samples=test_samples)

    if extra_datasets:
        logger.info(
            "Datasets: train=%d (%d RAF-DB + %d extra) val=%d test=%d",
            len(train_ds), n_rafdb_train, len(train_ds) - n_rafdb_train, len(val_ds), len(test_ds),
        )
    else:
        logger.info(
            "Datasets: train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds)
        )
    return train_ds, val_ds, test_ds
