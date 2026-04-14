"""
Robustness evaluation on CIFAR-10-C.

CIFAR-10-C contains 15 corruption types, each at 5 severity levels.
Each corruption file is a numpy array of shape (50000, 32, 32, 3),
where the first 10000 samples are severity 1, next 10000 are severity 2, etc.

Reference:
    Hendrycks, D. & Dietterich, T. (2019).
    Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.
    ICLR 2019. https://openreview.net/forum?id=HJz6tiCqYm

Download:
    https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from typing import Dict, List, Tuple

from parameters import DataConfig, TrainingConfig

# All 15 corruption types in CIFAR-10-C
CORRUPTIONS: List[str] = [
    "brightness", "contrast", "defocus_blur", "elastic_transform",
    "fog", "frost", "gaussian_blur", "gaussian_noise",
    "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur",
    "pixelate", "saturate", "shot_noise", "snow",
    "spatter", "speckle_noise", "zoom_blur",
]

# The 15 used in the original benchmark paper
BENCHMARK_CORRUPTIONS: List[str] = [
    "brightness", "contrast", "defocus_blur", "elastic_transform",
    "fog", "frost", "gaussian_noise", "glass_blur",
    "impulse_noise", "jpeg_compression", "motion_blur",
    "pixelate", "shot_noise", "snow", "zoom_blur",
]


def _get_normalize(data_cfg: DataConfig) -> transforms.Normalize:
    """Return normalisation transform for CIFAR-10."""
    return transforms.Normalize(data_cfg.mean, data_cfg.std)


def evaluate_corruption(
    model: nn.Module,
    corruption: str,
    severity: int,
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
) -> float:
    """Evaluate model accuracy on a single corruption type at a given severity.

    Args:
        model: Model to evaluate (should already be on device).
        corruption: Name of the corruption (e.g. 'gaussian_noise').
        severity: Severity level 1-5.
        data_cfg: Dataset config for normalization.
        train_cfg: Training config (batch_size, cifar10c_dir).
        device: Compute device.

    Returns:
        Top-1 accuracy as a float in [0, 1].
    """
    data_path  = os.path.join(train_cfg.cifar10c_dir, f"{corruption}.npy")
    label_path = os.path.join(train_cfg.cifar10c_dir, "labels.npy")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"CIFAR-10-C file not found: {data_path}\n"
            f"Download from: https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
        )

    data   = np.load(data_path)    # (50000, 32, 32, 3) uint8
    labels = np.load(label_path)   # (50000,) int64

    # Severity index: each 10000 samples = one severity level
    start = (severity - 1) * 10000
    end   = severity * 10000
    data_sev   = data[start:end]
    labels_sev = labels[start:end]

    # Normalise: uint8 → float32 → [0,1] → normalised
    normalize = _get_normalize(data_cfg)
    imgs = torch.from_numpy(data_sev).permute(0, 3, 1, 2).float() / 255.0
    imgs = torch.stack([normalize(img) for img in imgs])
    lbls = torch.from_numpy(labels_sev).long()

    dataset = TensorDataset(imgs, lbls)
    loader  = DataLoader(dataset, batch_size=train_cfg.batch_size,
                         shuffle=False, num_workers=data_cfg.num_workers,
                         pin_memory=True)

    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += model(x).argmax(1).eq(y).sum().item()
            n       += y.size(0)

    return correct / n


def run_corrupted_evaluation(
    model: nn.Module,
    model_label: str,
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
    severities: List[int] = [1, 2, 3, 4, 5],
    corruptions: List[str] = None,
) -> Dict[str, Dict[int, float]]:
    """Evaluate robustness across all corruptions and severities.

    Prints a summary table and saves a heatmap to plots/.

    Args:
        model: Evaluated model.
        model_label: Label string for titles/filenames.
        data_cfg: Dataset config.
        train_cfg: Training config.
        device: Compute device.
        severities: List of severity levels to evaluate.
        corruptions: List of corruption names (defaults to benchmark 15).

    Returns:
        Nested dict {corruption: {severity: accuracy}}.
    """
    if corruptions is None:
        corruptions = _get_available_corruptions(train_cfg.cifar10c_dir)

    results: Dict[str, Dict[int, float]] = {}

    print(f"\n{'='*60}")
    print(f"CIFAR-10-C Robustness Evaluation — {model_label}")
    print(f"{'='*60}")
    print(f"{'Corruption':<25}", end="")
    for s in severities:
        print(f"  Sev{s}", end="")
    print(f"  {'Mean':>7}")
    print("-" * 60)

    for corruption in corruptions:
        results[corruption] = {}
        accs = []
        for s in severities:
            try:
                acc = evaluate_corruption(model, corruption, s, data_cfg, train_cfg, device)
            except FileNotFoundError:
                acc = float("nan")
            results[corruption][s] = acc
            accs.append(acc)
        mean_acc = np.nanmean(accs)
        print(f"{corruption:<25}", end="")
        for acc in accs:
            print(f"  {acc:.3f}", end="")
        print(f"  {mean_acc:.3f}")

    # Mean Corruption Accuracy (mCA)
    all_accs = [acc for sev_dict in results.values() for acc in sev_dict.values()]
    mca = np.nanmean(all_accs)
    print(f"\nMean Corruption Accuracy (mCA): {mca:.4f}")

    # ── Save heatmap ─────────────────────────────────────────────────────────
    _save_heatmap(results, severities, model_label)

    return results


def _get_available_corruptions(cifar10c_dir: str) -> List[str]:
    """Return list of corruption names available on disk."""
    available = []
    for c in BENCHMARK_CORRUPTIONS:
        if os.path.exists(os.path.join(cifar10c_dir, f"{c}.npy")):
            available.append(c)
    if not available:
        # Fall back to anything present
        for c in CORRUPTIONS:
            if os.path.exists(os.path.join(cifar10c_dir, f"{c}.npy")):
                available.append(c)
    return available


def _save_heatmap(
    results: Dict[str, Dict[int, float]],
    severities: List[int],
    model_label: str,
) -> None:
    """Save a heatmap of accuracy per corruption × severity.

    Args:
        results: Nested dict from run_corrupted_evaluation.
        severities: Severity levels evaluated.
        model_label: Used for plot title and filename.
    """
    corruptions = list(results.keys())
    matrix = np.array([
        [results[c][s] for s in severities]
        for c in corruptions
    ])

    fig, ax = plt.subplots(figsize=(max(6, len(severities) * 1.5),
                                    max(6, len(corruptions) * 0.5)))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(severities)))
    ax.set_xticklabels([f"Sev {s}" for s in severities])
    ax.set_yticks(range(len(corruptions)))
    ax.set_yticklabels(corruptions)
    ax.set_title(f"CIFAR-10-C Accuracy — {model_label}")
    plt.colorbar(im, ax=ax, label="Accuracy")
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    out = f"plots/cifar10c_{model_label.replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Heatmap saved → {out}")


def compare_robustness(
    results_baseline: Dict[str, Dict[int, float]],
    results_augmix:   Dict[str, Dict[int, float]],
) -> None:
    """Print a side-by-side comparison of baseline vs AugMix robustness.

    Args:
        results_baseline: Results from baseline model.
        results_augmix: Results from AugMix-trained model.
    """
    corruptions = list(results_baseline.keys())

    print(f"\n{'='*65}")
    print("Robustness Comparison: Baseline vs AugMix")
    print(f"{'='*65}")
    print(f"{'Corruption':<25}  {'Baseline':>10}  {'AugMix':>10}  {'Delta':>8}")
    print("-" * 65)

    deltas = []
    for c in corruptions:
        base_mean = np.nanmean(list(results_baseline[c].values()))
        augm_mean = np.nanmean(list(results_augmix[c].values()))
        delta = augm_mean - base_mean
        deltas.append(delta)
        print(f"{c:<25}  {base_mean:>10.4f}  {augm_mean:>10.4f}  {delta:>+8.4f}")

    print("-" * 65)
    overall_base = np.nanmean([np.nanmean(list(v.values())) for v in results_baseline.values()])
    overall_augm = np.nanmean([np.nanmean(list(v.values())) for v in results_augmix.values()])
    print(f"{'OVERALL mCA':<25}  {overall_base:>10.4f}  {overall_augm:>10.4f}  "
          f"{overall_augm - overall_base:>+8.4f}")
