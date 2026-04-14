"""
Knowledge Distillation training module.

Supports both standard KD (soft + hard targets) and soft-targets-only mode.
This is an extended version of the HW2 distillation module that accepts
any teacher checkpoint path, enabling re-use with the AugMix-trained teacher.

Reference:
    Hinton, G., Vinyals, O., & Dean, J. (2015).
    Distilling the Knowledge in a Neural Network.
    NeurIPS Deep Learning and Representation Learning Workshop.
"""

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Tuple, Optional

from parameters import DataConfig, ModelConfig, TrainingConfig
from train import get_loaders, build_scheduler, validate

os.makedirs("plots", exist_ok=True)
os.makedirs("models/saved", exist_ok=True)

try:
    from ptflops import get_model_complexity_info
    _PTFLOPS_AVAILABLE = True
except ImportError:
    _PTFLOPS_AVAILABLE = False


def count_flops(model: nn.Module, input_res: Tuple[int, int, int] = (3, 32, 32)) -> None:
    """Print GMACs and parameter count using ptflops.

    Args:
        model: PyTorch model.
        input_res: Input tensor shape (C, H, W).
    """
    if not _PTFLOPS_AVAILABLE:
        print("ptflops not installed — skipping FLOPs count. "
              "Install with: pip install ptflops")
        return

    macs, params = get_model_complexity_info(
        model, input_res, as_strings=True,
        print_per_layer_stat=False, verbose=False,
    )
    print(f"  GMACs : {macs}")
    print(f"  Params: {params}")


class KDLoss(nn.Module):
    """Combined knowledge distillation loss (soft + hard targets).

    Loss = α * T² * KL(student_soft || teacher_soft) + (1-α) * CE(student, labels)

    Args:
        temperature: Softmax temperature for soft targets.
        alpha: Weight of distillation loss. (1-alpha) goes to hard-label CE.
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7) -> None:
        super().__init__()
        self.T     = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the KD loss.

        Args:
            student_logits: Raw logits from student (N, C).
            teacher_logits: Raw logits from teacher (N, C).
            labels: Ground-truth class indices (N,).

        Returns:
            Scalar loss.
        """
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(teacher_logits  / self.T, dim=1)

        kd_loss   = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (self.T ** 2)
        hard_loss = F.cross_entropy(student_logits, labels)

        return self.alpha * kd_loss + (1 - self.alpha) * hard_loss


class SoftTargetLoss(nn.Module):
    """Dynamic label-smoothing loss using teacher confidence as soft targets.

    Assigns teacher probability p_y to the true class and distributes
    (1 - p_y) / (C - 1) equally to all other classes.

    This was used for MobileNetV2 in HW2.

    Args:
        num_classes: Number of output classes.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.C = num_classes

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute soft-target loss.

        Args:
            student_logits: Raw logits from student (N, C).
            teacher_logits: Raw logits from teacher (N, C).
            labels: Ground-truth class indices (N,).

        Returns:
            Scalar cross-entropy against soft targets.
        """
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits, dim=1)
            py = teacher_probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # (N,)

            soft_targets = (1 - py).unsqueeze(1).expand(-1, self.C) / (self.C - 1)
            soft_targets = soft_targets.clone()
            soft_targets.scatter_(1, labels.unsqueeze(1), py.unsqueeze(1))

        log_probs = F.log_softmax(student_logits, dim=1)
        return -(soft_targets * log_probs).sum(dim=1).mean()


def _distill_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    train_cfg: TrainingConfig,
) -> Tuple[float, float]:
    """Run one distillation training epoch.

    Args:
        student: Student model (being trained).
        teacher: Teacher model (frozen, eval mode).
        loader: Training DataLoader.
        optimizer: Student optimizer.
        criterion: KDLoss or SoftTargetLoss instance.
        device: Compute device.
        train_cfg: Training config.

    Returns:
        Tuple of (avg_loss, accuracy) for the student.
    """
    student.train()
    teacher.eval()
    total_loss, correct, n = 0.0, 0, 0

    with torch.no_grad():
        pass  # teacher inference happens inside loop below

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.no_grad():
            teacher_logits = teacher(imgs)

        student_logits = student(imgs)
        loss = criterion(student_logits, teacher_logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += student_logits.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % train_cfg.log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

    return total_loss / n, correct / n


def run_distillation(
    student: nn.Module,
    teacher: nn.Module,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
) -> float:
    """Full knowledge distillation training loop.

    Args:
        student: Student model to train.
        teacher: Teacher model (weights frozen).
        data_cfg: Dataset config.
        model_cfg: Model config.
        train_cfg: Training config.
        device: Compute device.

    Returns:
        Best validation accuracy.
    """
    print(f"\nKnowledge Distillation: {train_cfg.run_name}")
    print(f"  Mode        : {train_cfg.distill_mode}")
    print(f"  Temperature : {train_cfg.temperature}")
    print(f"  Alpha       : {train_cfg.alpha}")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    train_loader, val_loader = get_loaders(data_cfg, train_cfg)
    val_criterion = nn.CrossEntropyLoss()

    if train_cfg.distill_mode == "soft_targets_only":
        criterion = SoftTargetLoss(num_classes=data_cfg.num_classes)
    else:
        criterion = KDLoss(temperature=train_cfg.temperature, alpha=train_cfg.alpha)

    optimizer = torch.optim.Adam(
        student.parameters(),
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
        tr_loss, tr_acc = _distill_one_epoch(
            student, teacher, train_loader, optimizer, criterion, device, train_cfg
        )
        val_loss, val_acc = validate(student, val_loader, val_criterion, device)

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
            best_weights = copy.deepcopy(student.state_dict())
            no_improve   = 0
            torch.save(best_weights, train_cfg.save_path)
            print(f"  ✓ Saved best student (val_acc={best_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("  Early stopping triggered.")
                break

    student.load_state_dict(best_weights)
    print(f"\nDistillation done. Best val acc: {best_acc:.4f}")

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Distillation Loss — {train_cfg.run_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/loss_{train_cfg.run_name}.png")
    plt.close()

    return best_acc
