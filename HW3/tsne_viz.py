"""
t-SNE visualisation of model feature embeddings for clean vs adversarial samples.

Extracts penultimate-layer features from the model, then projects them
to 2D using t-SNE and colours points by class and sample type.

Reference:
    van der Maaten, L. & Hinton, G. (2008).
    Visualizing Data using t-SNE. JMLR, 9, 2579–2605.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets
from typing import Optional, Tuple

from parameters import DataConfig, TrainingConfig
from train import get_transforms
from pgd_attack import pgd_linf

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# 10 visually distinct colours for CIFAR-10 classes
CLASS_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
]


class FeatureExtractor:
    """Hook-based feature extractor for the penultimate layer of a model.

    Args:
        model: Model whose pre-classification features we want.
        layer: The layer immediately before the classification head.
    """

    def __init__(self, model: nn.Module, layer: nn.Module) -> None:
        self.features: Optional[torch.Tensor] = None
        self._hook = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, _module, _input, output: torch.Tensor) -> None:
        self.features = output.detach()

    def remove(self) -> None:
        """Remove the hook."""
        self._hook.remove()


def _extract_features(
    model: nn.Module,
    loader: DataLoader,
    feature_layer: nn.Module,
    device: torch.device,
    max_samples: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract penultimate-layer features for all clean samples.

    Args:
        model: Model to extract from.
        loader: DataLoader for the dataset.
        feature_layer: Layer to hook.
        device: Compute device.
        max_samples: Maximum number of samples to extract.

    Returns:
        Tuple of (features: (N, D), labels: (N,)).
    """
    extractor = FeatureExtractor(model, feature_layer)
    model.eval()

    feats_list, labels_list = [], []
    n_collected = 0

    with torch.no_grad():
        for imgs, labels in loader:
            if n_collected >= max_samples:
                break
            imgs = imgs.to(device)
            model(imgs)  # trigger hook
            feat = extractor.features
            if feat.dim() > 2:
                feat = feat.view(feat.size(0), -1)
            feats_list.append(feat.cpu().numpy())
            labels_list.append(labels.numpy())
            n_collected += feat.size(0)

    extractor.remove()
    feats  = np.concatenate(feats_list, axis=0)[:max_samples]
    labels = np.concatenate(labels_list, axis=0)[:max_samples]
    return feats, labels


def _extract_adv_features(
    model: nn.Module,
    loader: DataLoader,
    feature_layer: nn.Module,
    train_cfg: TrainingConfig,
    device: torch.device,
    max_samples: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features for adversarial examples (PGD-20 L∞).

    Args:
        model: Model to attack and extract from.
        loader: DataLoader for clean samples.
        feature_layer: Layer to hook for feature extraction.
        train_cfg: Training config (PGD parameters).
        device: Compute device.
        max_samples: Maximum adversarial samples to generate.

    Returns:
        Tuple of (adv_features: (N, D), labels: (N,)).
    """
    extractor = FeatureExtractor(model, feature_layer)
    model.eval()

    feats_list, labels_list = [], []
    n_collected = 0

    for imgs, labels in loader:
        if n_collected >= max_samples:
            break
        imgs, labels = imgs.to(device), labels.to(device)

        adv = pgd_linf(
            model, imgs, labels,
            eps   = train_cfg.pgd_eps_linf,
            alpha = train_cfg.pgd_step_size_linf,
            steps = train_cfg.pgd_steps,
            device= device,
        )

        with torch.no_grad():
            model(adv)  # trigger hook
        feat = extractor.features
        if feat.dim() > 2:
            feat = feat.view(feat.size(0), -1)

        feats_list.append(feat.cpu().numpy())
        labels_list.append(labels.cpu().numpy())
        n_collected += feat.size(0)

    extractor.remove()
    feats  = np.concatenate(feats_list, axis=0)[:max_samples]
    labels = np.concatenate(labels_list, axis=0)[:max_samples]
    return feats, labels


def run_tsne_visualization(
    model: nn.Module,
    feature_layer: nn.Module,
    model_label: str,
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    device: torch.device,
    n_clean: int = 1000,
    n_adv: int = 300,
    perplexity: float = 40.0,
    random_state: int = 42,
) -> None:
    """Generate a t-SNE plot showing clean and adversarial sample distributions.

    Saves a figure to plots/ with:
    - Clean samples as filled circles coloured by class
    - Adversarial samples as crosses (×) coloured by class

    Args:
        model: Model to evaluate.
        feature_layer: Penultimate layer for feature extraction.
        model_label: Label for plot title and filename.
        data_cfg: Dataset config.
        train_cfg: Training config.
        device: Compute device.
        n_clean: Number of clean samples to embed.
        n_adv: Number of adversarial samples to embed.
        perplexity: t-SNE perplexity parameter.
        random_state: Random seed for t-SNE reproducibility.
    """
    print(f"\nRunning t-SNE for {model_label} ...")

    tf      = get_transforms(data_cfg, train=False)
    test_ds = datasets.CIFAR10(data_cfg.data_dir, train=False, download=True, transform=tf)
    loader  = DataLoader(test_ds, batch_size=train_cfg.batch_size,
                         shuffle=False, num_workers=data_cfg.num_workers, pin_memory=True)

    # Extract clean features
    print("  Extracting clean features ...")
    clean_feats, clean_labels = _extract_features(
        model, loader, feature_layer, device, max_samples=n_clean
    )

    # Extract adversarial features
    print("  Generating adversarial examples and extracting features ...")
    adv_feats, adv_labels = _extract_adv_features(
        model, loader, feature_layer, train_cfg, device, max_samples=n_adv
    )

    # Combine and run t-SNE
    all_feats  = np.concatenate([clean_feats, adv_feats], axis=0)
    all_labels = np.concatenate([clean_labels, adv_labels], axis=0)
    is_adv     = np.array([False] * len(clean_feats) + [True] * len(adv_feats))

    print(f"  Running t-SNE on {len(all_feats)} samples (perplexity={perplexity}) ...")
    tsne   = TSNE(n_components=2, perplexity=perplexity,
                  random_state=random_state, max_iter=1000, verbose=0)
    coords = tsne.fit_transform(all_feats)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 9))

    for cls_idx in range(data_cfg.num_classes):
        color = CLASS_COLORS[cls_idx]
        cls_name = CIFAR10_CLASSES[cls_idx]

        # Clean: filled circles
        mask_clean = (~is_adv) & (all_labels == cls_idx)
        if mask_clean.any():
            ax.scatter(coords[mask_clean, 0], coords[mask_clean, 1],
                       c=color, marker="o", s=12, alpha=0.6, label=f"{cls_name} (clean)")

        # Adversarial: crosses
        mask_adv = is_adv & (all_labels == cls_idx)
        if mask_adv.any():
            ax.scatter(coords[mask_adv, 0], coords[mask_adv, 1],
                       c=color, marker="x", s=30, alpha=0.9, linewidths=1.2)

    # Legend: classes (colour) + sample type (marker)
    class_patches = [
        mpatches.Patch(color=CLASS_COLORS[i], label=CIFAR10_CLASSES[i])
        for i in range(data_cfg.num_classes)
    ]
    clean_marker = plt.Line2D([0], [0], marker="o", color="gray",
                               markersize=6, linestyle="None", label="Clean")
    adv_marker   = plt.Line2D([0], [0], marker="x", color="gray",
                               markersize=6, linestyle="None", label="Adversarial (PGD-20 L∞)",
                               markeredgewidth=1.5)
    ax.legend(
        handles=class_patches + [clean_marker, adv_marker],
        loc="upper right", fontsize=7, ncol=2,
    )

    ax.set_title(f"t-SNE: Clean vs Adversarial Embeddings — {model_label}", fontsize=13)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    out = f"plots/tsne_{model_label.replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"t-SNE plot saved → {out}")
