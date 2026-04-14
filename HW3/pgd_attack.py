"""
PGD (Projected Gradient Descent) adversarial attack implementation.

Implements PGD-20 attack for both L∞ and L2 threat models,
as described in Madry et al. (2018).

Reference:
    Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018).
    Towards Deep Learning Models Resistant to Adversarial Attacks.
    ICLR 2018.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from typing import Tuple, Optional

from parameters import DataConfig, TrainingConfig
from train import get_transforms


def pgd_linf(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    device: torch.device,
    random_start: bool = True,
) -> torch.Tensor:
    """PGD attack under L∞ norm.

    Args:
        model: Target model.
        images: Clean input images (N, C, H, W), normalised.
        labels: True labels (N,).
        eps: L∞ perturbation budget.
        alpha: Step size.
        steps: Number of PGD iterations.
        device: Compute device.
        random_start: If True, start from a random perturbation in the epsilon ball.

    Returns:
        Adversarial images of the same shape as input.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    if random_start:
        delta = torch.empty_like(images).uniform_(-eps, eps)
    else:
        delta = torch.zeros_like(images)

    delta = delta.to(device)
    delta.requires_grad_(True)

    for _ in range(steps):
        adv = images + delta
        loss = criterion(model(adv), labels)
        loss.backward()

        with torch.no_grad():
            grad_sign = delta.grad.sign()
            delta.data = delta.data + alpha * grad_sign
            delta.data = delta.data.clamp(-eps, eps)
            # Keep adversarial examples in valid image space [input_min, input_max]
            # (we work in normalised space so we don't hard-clip to [0,1])

        delta.grad.zero_()

    return (images + delta.detach()).detach()


def pgd_l2(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    device: torch.device,
    random_start: bool = True,
) -> torch.Tensor:
    """PGD attack under L2 norm.

    Args:
        model: Target model.
        images: Clean input images (N, C, H, W), normalised.
        labels: True labels (N,).
        eps: L2 perturbation budget.
        alpha: Step size.
        steps: Number of PGD iterations.
        device: Compute device.
        random_start: If True, initialise delta from unit sphere.

    Returns:
        Adversarial images of the same shape as input.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    batch  = images.size(0)

    if random_start:
        delta = torch.randn_like(images)
        norms = delta.view(batch, -1).norm(dim=1, keepdim=True)
        delta = delta / norms.view(batch, 1, 1, 1) * np.random.uniform(0, eps)
    else:
        delta = torch.zeros_like(images)

    delta = delta.to(device)
    delta.requires_grad_(True)

    for _ in range(steps):
        adv  = images + delta
        loss = criterion(model(adv), labels)
        loss.backward()

        with torch.no_grad():
            grad = delta.grad
            # Normalise gradient to unit L2 per sample
            gnorm = grad.view(batch, -1).norm(dim=1).view(batch, 1, 1, 1)
            grad_unit = grad / (gnorm + 1e-8)
            delta.data = delta.data + alpha * grad_unit
            # Project onto L2 ball of radius eps
            dnorm = delta.data.view(batch, -1).norm(dim=1).view(batch, 1, 1, 1)
            factor = torch.min(torch.ones_like(dnorm), eps / (dnorm + 1e-8))
            delta.data = delta.data * factor

        delta.grad.zero_()

    return (images + delta.detach()).detach()


def run_pgd_evaluation(
    model: nn.Module,
    model_label: str,
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
    num_batches: Optional[int] = None,
) -> dict:
    """Evaluate a model under PGD-20 L∞ and L2 attacks.

    Args:
        model: Model to evaluate.
        model_label: Label string for printing.
        data_cfg: Dataset config.
        train_cfg: Training config (pgd_eps_linf, pgd_eps_l2, pgd_steps, etc.).
        device: Compute device.
        num_batches: If set, evaluate only on this many batches (faster testing).

    Returns:
        Dict with keys: clean_acc, linf_acc, l2_acc.
    """
    tf = get_transforms(data_cfg, train=False)
    test_ds = datasets.CIFAR10(data_cfg.data_dir, train=False, download=True, transform=tf)
    loader  = DataLoader(test_ds, batch_size=train_cfg.batch_size,
                         shuffle=False, num_workers=data_cfg.num_workers, pin_memory=True)

    model.eval()
    clean_correct = linf_correct = l2_correct = total = 0

    print(f"\n{'='*55}")
    print(f"PGD-20 Adversarial Evaluation — {model_label}")
    print(f"  L∞ ε = {train_cfg.pgd_eps_linf:.4f}  α = {train_cfg.pgd_step_size_linf:.4f}")
    print(f"  L2  ε = {train_cfg.pgd_eps_l2:.4f}  α = {train_cfg.pgd_step_size_l2:.4f}")
    print(f"{'='*55}")

    for batch_idx, (imgs, labels) in enumerate(loader):
        if num_batches is not None and batch_idx >= num_batches:
            break

        imgs, labels = imgs.to(device), labels.to(device)

        # Clean accuracy
        with torch.no_grad():
            clean_correct += model(imgs).argmax(1).eq(labels).sum().item()

        # L∞ attack
        adv_linf = pgd_linf(
            model, imgs, labels,
            eps   = train_cfg.pgd_eps_linf,
            alpha = train_cfg.pgd_step_size_linf,
            steps = train_cfg.pgd_steps,
            device= device,
        )
        with torch.no_grad():
            linf_correct += model(adv_linf).argmax(1).eq(labels).sum().item()

        # L2 attack
        adv_l2 = pgd_l2(
            model, imgs, labels,
            eps   = train_cfg.pgd_eps_l2,
            alpha = train_cfg.pgd_step_size_l2,
            steps = train_cfg.pgd_steps,
            device= device,
        )
        with torch.no_grad():
            l2_correct += model(adv_l2).argmax(1).eq(labels).sum().item()

        total += imgs.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f"  [{batch_idx+1}] clean={clean_correct/total:.4f}  "
                  f"L∞={linf_correct/total:.4f}  L2={l2_correct/total:.4f}")

    results = {
        "clean_acc": clean_correct / total,
        "linf_acc":  linf_correct  / total,
        "l2_acc":    l2_correct    / total,
    }

    print(f"\nResults for {model_label}:")
    print(f"  Clean accuracy : {results['clean_acc']:.4f}")
    print(f"  L∞ PGD-20 acc  : {results['linf_acc']:.4f}")
    print(f"  L2 PGD-20 acc  : {results['l2_acc']:.4f}")

    return results


def generate_adversarial_samples(
    model: nn.Module,
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
    norm: str = "linf",
    n_samples: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a batch of adversarial samples for downstream use (Grad-CAM, t-SNE).

    Selects samples where the clean model is correct but the adversarial example
    causes a misclassification.

    Args:
        model: Source model for generating adversarial examples.
        data_cfg: Dataset config.
        train_cfg: Training config.
        device: Compute device.
        norm: Attack norm, 'linf' or 'l2'.
        n_samples: Number of adversarial samples to return.

    Returns:
        Tuple (clean_imgs, adv_imgs, labels, adv_preds) — all on CPU.
    """
    tf      = get_transforms(data_cfg, train=False)
    test_ds = datasets.CIFAR10(data_cfg.data_dir, train=False, download=True, transform=tf)
    loader  = DataLoader(test_ds, batch_size=train_cfg.batch_size,
                         shuffle=False, num_workers=data_cfg.num_workers, pin_memory=True)

    clean_list, adv_list, label_list, adv_pred_list = [], [], [], []

    model.eval()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.no_grad():
            clean_preds = model(imgs).argmax(1)

        # Only use samples the clean model gets right
        mask = clean_preds.eq(labels)
        imgs_ok   = imgs[mask]
        labels_ok = labels[mask]
        if imgs_ok.size(0) == 0:
            continue

        if norm == "linf":
            adv = pgd_linf(model, imgs_ok, labels_ok,
                           eps=train_cfg.pgd_eps_linf,
                           alpha=train_cfg.pgd_step_size_linf,
                           steps=train_cfg.pgd_steps, device=device)
        else:
            adv = pgd_l2(model, imgs_ok, labels_ok,
                         eps=train_cfg.pgd_eps_l2,
                         alpha=train_cfg.pgd_step_size_l2,
                         steps=train_cfg.pgd_steps, device=device)

        with torch.no_grad():
            adv_preds = model(adv).argmax(1)

        # Keep only misclassified adversarial examples
        fooled = adv_preds.ne(labels_ok)
        clean_list.append(imgs_ok[fooled].cpu())
        adv_list.append(adv[fooled].cpu())
        label_list.append(labels_ok[fooled].cpu())
        adv_pred_list.append(adv_preds[fooled].cpu())

        total = sum(x.size(0) for x in clean_list)
        if total >= n_samples:
            break

    clean_all    = torch.cat(clean_list)[:n_samples]
    adv_all      = torch.cat(adv_list)[:n_samples]
    labels_all   = torch.cat(label_list)[:n_samples]
    adv_pred_all = torch.cat(adv_pred_list)[:n_samples]

    print(f"Generated {clean_all.size(0)} adversarial ({norm}) misclassified samples.")
    return clean_all, adv_all, labels_all, adv_pred_all


def run_transfer_attack(
    teacher: nn.Module,
    student: nn.Module,
    student_label: str,
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
    num_batches: Optional[int] = None,
) -> dict:
    """Generate PGD-20 L∞ adversarial examples on teacher, evaluate on student.

    Args:
        teacher: Model used to craft adversarial examples.
        student: Model to evaluate transferability against.
        student_label: Label for the student model.
        data_cfg: Dataset config.
        train_cfg: Training config.
        device: Compute device.
        num_batches: If set, limit evaluation to this many batches.

    Returns:
        Dict with keys: clean_acc, transfer_acc.
    """
    tf = get_transforms(data_cfg, train=False)
    test_ds = datasets.CIFAR10(data_cfg.data_dir, train=False, download=True, transform=tf)
    loader  = DataLoader(test_ds, batch_size=train_cfg.batch_size,
                         shuffle=False, num_workers=data_cfg.num_workers, pin_memory=True)

    teacher.eval()
    student.eval()
    clean_correct = transfer_correct = total = 0

    print(f"\n{'='*55}")
    print(f"Transfer Attack (teacher → {student_label})")
    print(f"  PGD-20 L∞ ε = {train_cfg.pgd_eps_linf:.4f}")
    print(f"{'='*55}")

    for batch_idx, (imgs, labels) in enumerate(loader):
        if num_batches is not None and batch_idx >= num_batches:
            break

        imgs, labels = imgs.to(device), labels.to(device)

        # Clean accuracy on student
        with torch.no_grad():
            clean_correct += student(imgs).argmax(1).eq(labels).sum().item()

        # Generate adversarial examples using teacher
        adv = pgd_linf(
            teacher, imgs, labels,
            eps   = train_cfg.pgd_eps_linf,
            alpha = train_cfg.pgd_step_size_linf,
            steps = train_cfg.pgd_steps,
            device= device,
        )

        # Evaluate on student
        with torch.no_grad():
            transfer_correct += student(adv).argmax(1).eq(labels).sum().item()

        total += imgs.size(0)

    results = {
        "clean_acc":    clean_correct    / total,
        "transfer_acc": transfer_correct / total,
    }

    print(f"\nTransfer Attack Results ({student_label}):")
    print(f"  Student clean acc   : {results['clean_acc']:.4f}")
    print(f"  Student transfer acc: {results['transfer_acc']:.4f}")
    print(f"  Accuracy drop       : {results['clean_acc'] - results['transfer_acc']:+.4f}")

    return results
