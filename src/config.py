"""Load and validate the YAML configuration."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ProjectCfg:
    name: str = "emotion-classifier"
    seed: int = 42
    device: str = "cuda"


@dataclass
class ExtraDatasetCfg:
    """An auxiliary training dataset pooled into the RAF-DB train split.

    Used only for training (validation/test stay RAF-DB-only so metrics remain
    comparable). Images are read ImageFolder-style; ``class_alias`` maps each
    source class-folder name to one of our canonical class names, or to an
    empty/"drop" value to exclude it (e.g. AffectNet's ``contempt``).
    """
    name: str = "extra"
    root: str = ""
    format: str = "imagefolder"
    train_dir: str = "train"
    class_alias: Dict[str, str] = field(default_factory=dict)


@dataclass
class DataCfg:
    root: str = "data/rafdb"
    # Layout of the dataset on disk: "rafdb_official" or "imagefolder" (Kaggle mirror).
    format: str = "rafdb_official"
    # rafdb_official layout:
    aligned_dir: str = "basic/Image/aligned"
    label_file: str = "basic/EmoLabel/list_patition_label.txt"
    # imagefolder layout (Kaggle): root/<train_dir|test_dir>/<class>/*.jpg
    train_dir: str = "train"
    test_dir: str = "test"
    num_classes: int = 7
    classes: List[str] = field(
        default_factory=lambda: [
            "surprise",
            "fear",
            "disgust",
            "happy",
            "sad",
            "angry",
            "neutral",
        ]
    )
    image_size: int = 224
    val_split: float = 0.1
    extra_train_datasets: List[ExtraDatasetCfg] = field(default_factory=list)


@dataclass
class ModelCfg:
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    freeze_ratio: float = 0.75
    dropout: float = 0.5


@dataclass
class TrainCfg:
    epochs: int = 40
    batch_size: int = 64
    num_workers: int = 4
    optimizer: str = "adamw"
    lr: float = 1e-4
    backbone_lr_mult: float = 1.0  # backbone LR = lr * this (discriminative fine-tuning)
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    use_class_weights: bool = True
    balanced_sampler: bool = False  # oversample minority classes (alternative to class weights)
    samples_per_epoch: int | None = None  # cap sampler draws/epoch (useful with large aux data)
    early_stopping_patience: int = 7
    amp: bool = True
    grad_clip: float | None = 1.0
    label_smoothing: float = 0.0


@dataclass
class AugmentCfg:
    horizontal_flip: bool = True
    rotation_deg: float = 10.0
    brightness: float = 0.1
    contrast: float = 0.0
    saturation: float = 0.0
    random_resized_crop_scale: float = 1.0  # min area scale; 1.0 disables RRC
    random_erasing: float = 0.0             # probability; 0 disables


@dataclass
class OutputCfg:
    dir: str = "outputs"
    checkpoint_name: str = "best_model.pth"
    tensorboard: bool = True


@dataclass
class Config:
    project: ProjectCfg = field(default_factory=ProjectCfg)
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    augment: AugmentCfg = field(default_factory=AugmentCfg)
    output: OutputCfg = field(default_factory=OutputCfg)

    @property
    def checkpoint_path(self) -> Path:
        """Absolute path to the best-model checkpoint."""
        return Path(self.output.dir) / self.output.checkpoint_name


def _build_section(cls: type, data: Dict[str, Any] | None) -> Any:
    """Instantiate a dataclass section, ignoring unknown keys with a warning."""
    data = data or {}
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(data) - known
    if unknown:
        logger.warning("Ignoring unknown config keys for %s: %s", cls.__name__, sorted(unknown))
    return cls(**{k: v for k, v in data.items() if k in known})


def _validate(cfg: Config) -> None:
    """Validate cross-field invariants; raise ValueError on failure."""
    if len(cfg.data.classes) != cfg.data.num_classes:
        raise ValueError(
            f"num_classes ({cfg.data.num_classes}) does not match len(classes) "
            f"({len(cfg.data.classes)}): {cfg.data.classes}"
        )
    if not 0.0 <= cfg.data.val_split < 1.0:
        raise ValueError(f"val_split must be in [0, 1), got {cfg.data.val_split}")
    if cfg.data.format not in {"rafdb_official", "imagefolder"}:
        raise ValueError(
            f"Unsupported data.format '{cfg.data.format}' "
            "(supported: rafdb_official, imagefolder)"
        )
    if not 0.0 <= cfg.model.freeze_ratio <= 1.0:
        raise ValueError(f"freeze_ratio must be in [0, 1], got {cfg.model.freeze_ratio}")
    if cfg.model.backbone not in {"efficientnet_b0", "resnet50"}:
        raise ValueError(
            f"Unsupported backbone '{cfg.model.backbone}' "
            "(supported: efficientnet_b0, resnet50)"
        )


def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a validated :class:`Config`.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A validated :class:`Config` instance.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    # extra_train_datasets is a list of nested objects -> parse separately.
    data_raw = dict(raw.get("data") or {})
    extra_raw = data_raw.pop("extra_train_datasets", None) or []
    data = _build_section(DataCfg, data_raw)
    data.extra_train_datasets = [_build_section(ExtraDatasetCfg, d) for d in extra_raw]

    cfg = Config(
        project=_build_section(ProjectCfg, raw.get("project")),
        data=data,
        model=_build_section(ModelCfg, raw.get("model")),
        train=_build_section(TrainCfg, raw.get("train")),
        augment=_build_section(AugmentCfg, raw.get("augment")),
        output=_build_section(OutputCfg, raw.get("output")),
    )
    _validate(cfg)
    logger.info("Loaded config from %s", path)
    return cfg
