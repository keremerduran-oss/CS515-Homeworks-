"""Training utilities for HW3.

Extends HW2 train.py with AugMix augmentation support.
All other logic (early stopping, scheduler, loss curves) is unchanged.
"""

import copy
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple

from parameters import DataConfig, ModelConfig, TrainingConfig
from augmix import AugMixTransform

os.makedirs("plots", exist_ok=True)
os.makedirs("models/saved", exist_ok=True)


def get_transforms(
    data_cfg: DataConfig,
    train: bool = True,
    resize: bool = False,
    use_augmix: bool = False,
    augmix_severity: int = 3,
    augmix_width: int = 3,
    augmix_depth: int = -1,
    augmix_alpha: float = 1.0,
) -> transforms.Compose:
    """Build the transform pipeline, optionally inserting AugMix before normalisation.

    Args:
        data_cfg: Dataset configuration (mean/std).
        train: If True, apply training augmentations.
        resize: Resize to 224x224 (transfer learning option 1).
        use_augmix: Whether to inject AugMix into the pipeline.
        augmix_severity: Severity of AugMix operations.
        augmix_width: Number of augmentation chains.
        augmix_depth: Depth per chain (-1 = random 1-3).
        augmix_alpha: Dirichlet alpha for AugMix mixing.

    Returns:
        Composed transform pipeline.
    """
    mean, std = data_cfg.mean, data_cfg.std

    if data_cfg.dataset == "mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    base = []
    if resize:
        base.append(transforms.Resize(224))

    if train:
        if not resize:
            base.append(transforms.RandomCrop(32, padding=4))
        base.append(transforms.RandomHorizontalFlip())

        if use_augmix:
            base.append(AugMixTransform(
                severity      = augmix_severity,
                mixture_width = augmix_width,
                chain_depth   = augmix_depth,
                alpha         = augmix_alpha,
            ))

    base += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(base)


def get_loaders(
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    resize: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders.

    Args:
        data_cfg: Dataset configuration.
        train_cfg: Training configuration.
        resize: Resize to 224x224.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_tf = get_transforms(
        data_cfg,
        train          = True,
        resize         = resize,
        use_augmix     = train_cfg.use_augmix,
        augmix_severity= train_cfg.augmix_severity,
        augmix_width   = train_cfg.augmix_mixture_width,
        augmix_depth   = train_cfg.augmix_chain_depth,
        augmix_alpha   = train_cfg.augmix_alpha,
    )
    val_tf = get_transforms(data_cfg, train=False, resize=resize)

    if data_cfg.dataset == "mnist":
        train_ds = datasets.MNIST(data_cfg.data_dir, train=True,  download=True, transform=train_tf)
        val_ds   = datasets.MNIST(data_cfg.data_dir, train=False, download=True, transform=val_tf)
    else:
        train_ds = datasets.CIFAR10(data_cfg.data_dir, train=True,  download=True, transform=train_tf)
        val_ds   = datasets.CIFAR10(data_cfg.data_dir, train=False, download=True, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size,
                              shuffle=True,  num_workers=data_cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=train_cfg.batch_size,
                              shuffle=False, num_workers=data_cfg.num_workers, pin_memory=True)
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    train_cfg: TrainingConfig,
) -> Tuple[float, float]:
    """Run one training epoch.

    Args:
        model: Network to train.
        loader: Training DataLoader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Compute device.
        train_cfg: Training config (reg_type, reg_lambda, log_interval).

    Returns:
        Tuple of (avg_loss, accuracy).
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)

        if train_cfg.reg_type == "l1":
            l1 = sum(p.abs().sum() for p in model.parameters())
            loss = loss + train_cfg.reg_lambda * l1

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % train_cfg.log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

    return total_loss / n, correct / n


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on a validation DataLoader.

    Args:
        model: Network to evaluate.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Compute device.

    Returns:
        Tuple of (avg_loss, accuracy).
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.detach().item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)

    return total_loss / n, correct / n


def build_criterion(train_cfg: TrainingConfig) -> nn.Module:
    """Build CrossEntropyLoss, optionally with label smoothing.

    Args:
        train_cfg: Training config with label_smoothing field.

    Returns:
        Loss criterion.
    """
    return nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: TrainingConfig,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build an LR scheduler.

    Args:
        optimizer: Optimizer to wrap.
        train_cfg: Training config with scheduler type and epochs.

    Returns:
        LR scheduler instance.
    """
    if train_cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.epochs)
    if train_cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


def run_training(
    model: nn.Module,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
    resize: bool = False,
) -> float:
    """Full training loop with early stopping, scheduling, and loss curve.

    Args:
        model: Network to train.
        data_cfg: Dataset configuration.
        model_cfg: Model configuration.
        train_cfg: Training hyperparameters.
        device: Compute device.
        resize: Resize images to 224x224.

    Returns:
        Best validation accuracy.
    """
    aug_str = " [AugMix]" if train_cfg.use_augmix else ""
    print(f"Starting training{aug_str} — {train_cfg.run_name}")

    train_loader, val_loader = get_loaders(data_cfg, train_cfg, resize=resize)
    criterion = build_criterion(train_cfg)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = build_scheduler(optimizer, train_cfg)

    best_acc     = 0.0
    best_weights = None
    patience     = 10
    no_improve   = 0
    train_losses: list[float] = []
    val_losses:   list[float] = []

    for epoch in range(1, train_cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{train_cfg.epochs}")
        tr_loss,  tr_acc  = train_one_epoch(model, train_loader, optimizer, criterion, device, train_cfg)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if train_cfg.scheduler == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            no_improve   = 0
            torch.save(best_weights, train_cfg.save_path)
            print(f"  ✓ Saved best model (val_acc={best_acc:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement {no_improve}/{patience}")
            if no_improve >= patience:
                print("  Early stopping triggered.")
                break

    model.load_state_dict(best_weights)
    print(f"\nTraining done. Best val accuracy: {best_acc:.4f}")

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve — {train_cfg.run_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/loss_{train_cfg.run_name}.png")
    plt.close()
    print(f"Loss curve → plots/loss_{train_cfg.run_name}.png")

    return best_acc
