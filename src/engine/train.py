"""Training loop: AMP, cosine schedule, class-weighted loss, early stopping."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import Config
from src.data.dataset import build_datasets
from src.engine.evaluate import collect_predictions, compute_metrics
from src.engine.utils import AverageMeter, save_checkpoint, set_seed
from src.models.classifier import build_model

logger = logging.getLogger(__name__)


def _build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Build the optimizer over trainable params, with discriminative LRs.

    The head trains at ``lr``; unfrozen backbone params train at
    ``lr * backbone_lr_mult`` so pretrained features adapt gently during full
    fine-tuning. When ``backbone_lr_mult == 1.0`` this reduces to a single LR.
    """
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if name.startswith("head") else backbone_params).append(p)

    lr = cfg.train.lr
    groups = [{"params": head_params, "lr": lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr * cfg.train.backbone_lr_mult})
    logger.info(
        "Optimizer param groups: head lr=%.2e (%d tensors), backbone lr=%.2e (%d tensors)",
        lr, len(head_params), lr * cfg.train.backbone_lr_mult, len(backbone_params),
    )

    opt = cfg.train.optimizer.lower()
    if opt == "adamw":
        return torch.optim.AdamW(groups, lr=lr, weight_decay=cfg.train.weight_decay)
    if opt == "adam":
        return torch.optim.Adam(groups, lr=lr, weight_decay=cfg.train.weight_decay)
    if opt == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9, weight_decay=cfg.train.weight_decay)
    raise ValueError(f"Unsupported optimizer: {cfg.train.optimizer}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Config
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Build the LR scheduler (cosine or none)."""
    sched = cfg.train.scheduler.lower()
    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    if sched in ("none", "", "off"):
        return None
    raise ValueError(f"Unsupported scheduler: {cfg.train.scheduler}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    grad_clip: Optional[float],
    epoch: int,
) -> float:
    """Run one training epoch and return the average loss."""
    model.train()
    loss_meter = AverageMeter()
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), images.size(0))
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")
    return loss_meter.avg


def train(config: Config) -> Path:
    """Train the emotion classifier end to end.

    Saves the best checkpoint (by val macro-F1) to ``config.checkpoint_path``.

    Args:
        config: Validated configuration.

    Returns:
        Path to the best-model checkpoint.
    """
    set_seed(config.project.seed)
    device = torch.device(
        config.project.device if torch.cuda.is_available() or config.project.device == "cpu"
        else "cpu"
    )
    if config.project.device == "cuda" and device.type == "cpu":
        logger.warning("CUDA requested but not available; falling back to CPU.")
    logger.info("Training on device: %s", device)

    # Data
    train_ds, val_ds, _ = build_datasets(
        root=config.data.root,
        aligned_dir=config.data.aligned_dir,
        label_file=config.data.label_file,
        classes=config.data.classes,
        image_size=config.data.image_size,
        val_split=config.data.val_split,
        seed=config.project.seed,
        horizontal_flip=config.augment.horizontal_flip,
        rotation_deg=config.augment.rotation_deg,
        brightness=config.augment.brightness,
        contrast=config.augment.contrast,
        saturation=config.augment.saturation,
        rrc_scale_min=config.augment.random_resized_crop_scale,
        random_erasing=config.augment.random_erasing,
        data_format=config.data.format,
        train_dir=config.data.train_dir,
        test_dir=config.data.test_dir,
        extra_datasets=[
            {
                "name": e.name,
                "root": e.root,
                "train_dir": e.train_dir,
                "class_alias": e.class_alias,
            }
            for e in config.data.extra_train_datasets
        ],
    )
    if config.train.balanced_sampler:
        from torch.utils.data import WeightedRandomSampler

        weights = train_ds.sample_weights()
        num_samples = config.train.samples_per_epoch or len(weights)
        sampler = WeightedRandomSampler(
            weights, num_samples=num_samples, replacement=True
        )
        logger.info(
            "Using class-balanced WeightedRandomSampler (minority oversampling), "
            "%d draws/epoch.", num_samples,
        )
        if config.train.use_class_weights:
            logger.warning(
                "Both balanced_sampler and use_class_weights are on — this "
                "double-corrects for imbalance; consider disabling one."
            )
        train_loader = DataLoader(
            train_ds, batch_size=config.train.batch_size, sampler=sampler,
            num_workers=config.train.num_workers, pin_memory=True, drop_last=False,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=config.train.batch_size, shuffle=True,
            num_workers=config.train.num_workers, pin_memory=True, drop_last=False,
        )
    val_loader = DataLoader(
        val_ds, batch_size=config.train.batch_size, shuffle=False,
        num_workers=config.train.num_workers, pin_memory=True,
    ) if len(val_ds) > 0 else None

    # Model
    model = build_model(config).to(device)

    # Loss (optionally class-weighted)
    class_weights = None
    if config.train.use_class_weights:
        class_weights = train_ds.compute_class_weights().to(device)
        logger.info("Using class weights: %s", class_weights.tolist())
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=config.train.label_smoothing
    )

    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)
    use_amp = config.train.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    writer: Optional[SummaryWriter] = (
        SummaryWriter(log_dir=str(Path(config.output.dir) / "tensorboard"))
        if config.output.tensorboard else None
    )

    best_f1 = -1.0
    epochs_no_improve = 0
    ckpt_path = config.checkpoint_path

    for epoch in range(1, config.train.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            use_amp, config.train.grad_clip, epoch,
        )

        # Validation
        if val_loader is not None:
            y_true, y_pred = collect_predictions(model, val_loader, device)
            metrics = compute_metrics(y_true, y_pred, config.data.classes)
            val_acc, val_f1 = metrics["accuracy"], metrics["macro_f1"]
        else:
            val_acc, val_f1 = 0.0, 0.0

        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_acc=%.4f | val_macroF1=%.4f | lr=%.2e",
            epoch, config.train.epochs, train_loss, val_acc, val_f1, lr,
        )
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("acc/val", val_acc, epoch)
            writer.add_scalar("macro_f1/val", val_f1, epoch)
            writer.add_scalar("lr", lr, epoch)

        if scheduler is not None:
            scheduler.step()

        # Checkpoint on best val macro-F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            save_checkpoint(
                {
                    "model_state": model.state_dict(),
                    "config_backbone": config.model.backbone,
                    "classes": config.data.classes,
                    "num_classes": config.data.num_classes,
                    "image_size": config.data.image_size,
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                },
                ckpt_path,
            )
            logger.info("New best val macro-F1=%.4f (epoch %d) -> saved", best_f1, epoch)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.train.early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d (no val improvement for %d epochs).",
                    epoch, epochs_no_improve,
                )
                break

    if writer is not None:
        writer.close()
    logger.info("Training complete. Best val macro-F1=%.4f | checkpoint=%s", best_f1, ckpt_path)
    return ckpt_path
