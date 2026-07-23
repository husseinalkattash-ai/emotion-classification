"""Tests for the RAF-DB dataset, label map, and class weights.

Uses a small synthetic fixture (dummy aligned images + a mock label file) so
the tests run without the real, license-gated RAF-DB dataset.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data.dataset import (
    RAFDB_ID_TO_NAME,
    RAFDB_LABEL_MAP,
    RAFDBDataset,
    build_datasets,
    build_label_map,
    build_transforms,
)

CLASSES = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]


@pytest.fixture()
def rafdb_root(tmp_path: Path) -> Path:
    """Create a minimal RAF-DB-like directory with dummy aligned images."""
    root = tmp_path / "rafdb"
    aligned = root / "basic" / "Image" / "aligned"
    emolabel = root / "basic" / "EmoLabel"
    aligned.mkdir(parents=True)
    emolabel.mkdir(parents=True)

    lines = []
    # Two samples per class (14 train), so stratified val split works; plus 7 test.
    idx = 1
    for rafdb_id in range(1, 8):
        for _ in range(2):
            name = f"train_{idx:05d}.jpg"
            Image.new("RGB", (100, 100), color=(rafdb_id * 30 % 255, 10, 10)).save(
                aligned / f"train_{idx:05d}_aligned.jpg"
            )
            lines.append(f"{name} {rafdb_id}")
            idx += 1
    for rafdb_id in range(1, 8):
        name = f"test_{rafdb_id:05d}.jpg"
        Image.new("RGB", (100, 100), color=(5, rafdb_id * 30 % 255, 5)).save(
            aligned / f"test_{rafdb_id:05d}_aligned.jpg"
        )
        lines.append(f"{name} {rafdb_id}")

    (emolabel / "list_patition_label.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def test_label_map_covers_all_seven_ids() -> None:
    assert set(RAFDB_LABEL_MAP.keys()) == set(range(1, 8))
    assert sorted(RAFDB_LABEL_MAP.values()) == list(range(7))
    # Canonical order: RAF-DB id i maps to model index i-1.
    for rafdb_id, name in RAFDB_ID_TO_NAME.items():
        assert CLASSES[RAFDB_LABEL_MAP[rafdb_id]] == name


def test_build_label_map_respects_config_order() -> None:
    reordered = ["neutral", "angry", "sad", "happy", "disgust", "fear", "surprise"]
    m = build_label_map(reordered)
    # RAF-DB id 7 == neutral, now at index 0.
    assert reordered[m[7]] == "neutral"
    assert reordered[m[1]] == "surprise"


def test_build_label_map_rejects_missing_class() -> None:
    with pytest.raises(ValueError):
        build_label_map(["happy", "sad"])  # incomplete


def test_dataset_item_shape_and_label(rafdb_root: Path) -> None:
    tf = build_transforms(224, train=False)
    ds = RAFDBDataset(
        root=rafdb_root,
        aligned_dir="basic/Image/aligned",
        label_file="basic/EmoLabel/list_patition_label.txt",
        classes=CLASSES,
        transform=tf,
    )
    assert len(ds) == 21  # 14 train + 7 test entries all present on disk
    image, label = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert isinstance(label, int)
    assert 0 <= label < 7


def test_strong_augment_transform_shape() -> None:
    """Full augmentation stack (RRC + jitter + erasing) still yields (3,224,224)."""
    tf = build_transforms(
        224, train=True, horizontal_flip=True, rotation_deg=12,
        brightness=0.2, contrast=0.2, saturation=0.2,
        rrc_scale_min=0.8, random_erasing=0.25,
    )
    out = tf(Image.new("RGB", (100, 100), (120, 80, 40)))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)


def test_compute_class_weights_returns_seven(rafdb_root: Path) -> None:
    ds = RAFDBDataset(
        root=rafdb_root,
        classes=CLASSES,
        transform=build_transforms(224, train=False),
    )
    weights = ds.compute_class_weights()
    assert weights.shape == (7,)
    assert torch.all(weights >= 0)


@pytest.fixture()
def imagefolder_root(tmp_path: Path) -> Path:
    """Create a Kaggle-style ImageFolder layout (train/<id>/ and test/<id>/)."""
    root = tmp_path / "rafdb_kaggle"
    for split, per_class in (("DATASET/train", 3), ("DATASET/test", 1)):
        for rafdb_id in range(1, 8):
            cls_dir = root / split / str(rafdb_id)
            cls_dir.mkdir(parents=True)
            for k in range(per_class):
                Image.new("RGB", (80, 80), (rafdb_id * 30 % 255, k * 10, 20)).save(
                    cls_dir / f"img_{rafdb_id}_{k}.jpg"
                )
    return root


def test_imagefolder_layout(imagefolder_root: Path) -> None:
    train_ds, val_ds, test_ds = build_datasets(
        root=imagefolder_root,
        aligned_dir="",  # unused for imagefolder
        label_file="",
        classes=CLASSES,
        image_size=224,
        val_split=0.25,
        seed=42,
        data_format="imagefolder",
        train_dir="DATASET/train",
        test_dir="DATASET/test",
    )
    # 7 classes * 3 train = 21 train samples, split into train/val; 7 test.
    assert len(train_ds) + len(val_ds) == 21
    assert len(test_ds) == 7
    img, label = train_ds[0]
    assert img.shape == (3, 224, 224)
    assert 0 <= label < 7
    # Folder id 4 == "happy" -> model index 3 in canonical CLASSES.
    labels_present = set(train_ds.labels + val_ds.labels + test_ds.labels)
    assert labels_present == set(range(7))


def test_imagefolder_name_folders(tmp_path: Path) -> None:
    """Class folders named by emotion (not id) resolve correctly."""
    root = tmp_path / "named"
    for split in ("train", "test"):
        for name in CLASSES:
            d = root / split / name
            d.mkdir(parents=True)
            Image.new("RGB", (60, 60), (50, 50, 50)).save(d / "a.jpg")
    _, _, test_ds = build_datasets(
        root=root, aligned_dir="", label_file="", classes=CLASSES,
        image_size=224, val_split=0.0, data_format="imagefolder",
        train_dir="train", test_dir="test",
    )
    assert len(test_ds) == 7
    assert set(test_ds.labels) == set(range(7))


def test_sample_weights_inverse_frequency(rafdb_root: Path) -> None:
    """sample_weights returns one positive weight per sample, inverse to freq."""
    # rafdb_root fixture has 2 train samples per class -> all weights equal.
    tr, _, _ = build_datasets(
        root=rafdb_root, aligned_dir="basic/Image/aligned",
        label_file="basic/EmoLabel/list_patition_label.txt", classes=CLASSES,
        image_size=224, val_split=0.0, seed=42,
    )
    w = tr.sample_weights()
    assert w.shape == (len(tr),)
    assert torch.all(w > 0)
    assert torch.allclose(w, w[0].expand_as(w))  # balanced fixture -> equal weights


def test_extra_dataset_pooling_and_alias(rafdb_root: Path, tmp_path: Path) -> None:
    """Auxiliary dataset is pooled into TRAIN only, with alias/drop applied."""
    # Synthetic AffectNet-like aux set: emotion-name folders incl. alias + drop.
    aux = tmp_path / "affectnet"
    folders = {
        "happy": 3, "sad": 2, "anger": 4, "contempt": 5,  # anger->angry, contempt->drop
    }
    for name, n in folders.items():
        d = aux / "train" / name
        d.mkdir(parents=True)
        for k in range(n):
            Image.new("RGB", (50, 50), (30, 30, 30)).save(d / f"{name}_{k}.jpg")

    # Baseline RAF-DB-only train/test counts.
    base_tr, base_val, base_test = build_datasets(
        root=rafdb_root, aligned_dir="basic/Image/aligned",
        label_file="basic/EmoLabel/list_patition_label.txt", classes=CLASSES,
        image_size=224, val_split=0.0, seed=42,
    )
    with_aux_tr, _, with_aux_test = build_datasets(
        root=rafdb_root, aligned_dir="basic/Image/aligned",
        label_file="basic/EmoLabel/list_patition_label.txt", classes=CLASSES,
        image_size=224, val_split=0.0, seed=42,
        extra_datasets=[{
            "name": "affectnet", "root": str(aux), "train_dir": "train",
            "class_alias": {"anger": "angry", "contempt": "drop"},
        }],
    )
    # happy(3)+sad(2)+anger(4) = 9 added; contempt(5) dropped.
    assert len(with_aux_tr) == len(base_tr) + 9
    # Test split is untouched by aux data.
    assert len(with_aux_test) == len(base_test)
    # "anger" mapped to our "angry" index (5).
    angry_idx = CLASSES.index("angry")
    assert with_aux_tr.labels.count(angry_idx) == base_tr.labels.count(angry_idx) + 4


def test_build_datasets_split_by_prefix(rafdb_root: Path) -> None:
    train_ds, val_ds, test_ds = build_datasets(
        root=rafdb_root,
        aligned_dir="basic/Image/aligned",
        label_file="basic/EmoLabel/list_patition_label.txt",
        classes=CLASSES,
        image_size=224,
        val_split=0.5,  # 14 train -> 7 train / 7 val
        seed=42,
    )
    assert len(test_ds) == 7
    assert len(train_ds) + len(val_ds) == 14
    assert len(val_ds) > 0
    img, _ = train_ds[0]
    assert img.shape == (3, 224, 224)
