"""
GradCAM — Gradient-weighted Class Activation Mapping
======================================================
GradCAM answers the question: "Which parts of the image made the model
predict class C?"

Key idea
--------
After a forward pass, we:
  1. Pick a convolutional layer (usually the last one — it has the richest
     spatial features while still being high-resolution enough to be useful).
  2. Compute the gradient of the class score  y^c  w.r.t. every activation
     map  A^k  in that layer.
  3. Average those gradients spatially to get importance weights  α_k^c.
  4. Take a weighted sum of the activation maps, then ReLU it.

     GradCAM(x) = ReLU( Σ_k  α_k^c · A^k )

     The ReLU keeps only the activations that *increase* the class score
     (negatively-contributing regions are ignored — they'd correspond to a
     different class).

  5. Resize the heatmap back to input resolution and overlay it.

References
----------
Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization," ICCV 2017.
https://arxiv.org/abs/1610.02391
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from typing import Tuple, Optional


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


# ── GradCAM core ──────────────────────────────────────────────────────────────

class GradCAM:
    """
    Computes GradCAM heatmaps for a given model and target layer.

    How the hooks work
    ------------------
    PyTorch's hook system lets us intercept:
      • forward_hook       — captures activations A^k after the forward pass.
      • full_backward_hook — captures gradients dY/dA^k after the backward pass.

    We register both on the target conv layer, then read them off after calling
    loss.backward() to compute the weighted heatmap.

    Args:
        model: The neural network to explain.
        target_layer: The convolutional layer to hook (e.g. model.layer4[-1]).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model       = model
        self.activations = None
        self.gradients   = None
        self._fwd_hook   = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach()))
        self._bwd_hook   = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach()))

    def __call__(
        self,
        x: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """Generate a GradCAM heatmap for a single image.

        Args:
            x: Input image tensor of shape (1, C, H, W).
            class_idx: Class index to explain. Uses argmax if None.

        Returns:
            Tuple of (heatmap, class_idx) where heatmap is a (H, W) float
            array in [0, 1] at the same resolution as the input image.
        """
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(x)                                     # (1, num_classes)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        logits[0, class_idx].backward()

        # alpha_k^c = global-average-pool of gradients → weighted sum → ReLU
        # ex. for resnet18 layer4[-1]: weights shape is (1, 512, 1, 1)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        # ex. for resnet18 layer4[-1]: cam shape is (1, 1, 4, 4) for CIFAR-10 32x32 input
        cam     = torch.relu(
            (weights * self.activations).sum(dim=1, keepdim=True)
        )

        # Normalise and upsample to input resolution
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        h, w    = x.shape[2], x.shape[3]
        heatmap = np.array(
            Image.fromarray(np.uint8(cam * 255)).resize((w, h), Image.BILINEAR)
        ) / 255.0                                                   # (H, W) in [0,1]
        return heatmap, class_idx

    def remove_hooks(self) -> None:
        """Remove forward and backward hooks from the target layer."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ── Helpers ───────────────────────────────────────────────────────────────────

def denormalize(
    t: torch.Tensor,
    mean: Tuple = CIFAR10_MEAN,
    std:  Tuple = CIFAR10_STD,
) -> np.ndarray:
    """Convert a normalised CIFAR-10 tensor to a displayable numpy array.

    Args:
        t: Image tensor of shape (C, H, W) or (1, C, H, W).
        mean: Per-channel mean used during normalisation.
        std: Per-channel std used during normalisation.

    Returns:
        (H, W, 3) float array clipped to [0, 1].
    """
    t = t.clone().squeeze(0) * torch.tensor(std).view(3, 1, 1) \
        + torch.tensor(mean).view(3, 1, 1)
    return np.clip(t.permute(1, 2, 0).numpy(), 0, 1)


def overlay(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a jet-colourmap heatmap on a RGB image.

    Args:
        img: (H, W, 3) float array in [0, 1].
        heatmap: (H, W) float array in [0, 1].
        alpha: Heatmap blending weight.

    Returns:
        (H, W, 3) uint8 blended array.
    """
    rgb = cm.get_cmap("jet")(heatmap)[:, :, :3]
    return (np.clip((1 - alpha) * img + alpha * rgb, 0, 1) * 255).astype(np.uint8)


# ── HW3: adversarial vs clean visualisation ───────────────────────────────────

def visualize_gradcam_pairs(
    model: nn.Module,
    clean_imgs: torch.Tensor,
    adv_imgs:   torch.Tensor,
    true_labels:  torch.Tensor,
    adv_preds:    torch.Tensor,
    target_layer: nn.Module,
    device: torch.device,
    n_samples: int = 2,
    model_label: str = "model",
    mean: Tuple = CIFAR10_MEAN,
    std:  Tuple = CIFAR10_STD,
) -> None:
    """Visualise GradCAM for clean vs adversarial image pairs side-by-side.

    For each sample shows: original | GradCAM (clean) | adversarial |
    GradCAM (adv) | perturbation magnitude.

    Args:
        model: The neural network.
        clean_imgs: (N, C, H, W) clean image tensors (CPU).
        adv_imgs: (N, C, H, W) adversarial image tensors (CPU).
        true_labels: (N,) ground-truth label indices.
        adv_preds: (N,) model predictions on adversarial images.
        target_layer: Layer to hook (e.g. model.layer4[-1]).
        device: Compute device.
        n_samples: Number of sample pairs to plot.
        model_label: Used in plot title and filename.
        mean: CIFAR-10 normalisation mean.
        std: CIFAR-10 normalisation std.
    """
    os.makedirs("plots", exist_ok=True)
    n = min(n_samples, clean_imgs.size(0))

    fig, axes = plt.subplots(n, 5, figsize=(18, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "Clean Image", "GradCAM (Clean)",
        "Adv Image",   "GradCAM (Adv)",
        "Perturbation",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for i in range(n):
        true_cls = CIFAR10_CLASSES[true_labels[i].item()]
        adv_cls  = CIFAR10_CLASSES[adv_preds[i].item()]

        clean_t = clean_imgs[i].unsqueeze(0).to(device)
        adv_t   = adv_imgs[i].unsqueeze(0).to(device)

        # GradCAM on clean image — explain the true class
        cam_clean = GradCAM(model, target_layer)
        heatmap_clean, _ = cam_clean(clean_t, class_idx=true_labels[i].item())
        cam_clean.remove_hooks()

        # GradCAM on adversarial image — explain the (wrong) predicted class
        cam_adv = GradCAM(model, target_layer)
        heatmap_adv, _ = cam_adv(adv_t, class_idx=adv_preds[i].item())
        cam_adv.remove_hooks()

        clean_np = denormalize(clean_imgs[i], mean, std)
        adv_np   = denormalize(adv_imgs[i],  mean, std)
        diff_np  = np.abs(adv_np - clean_np)
        diff_np  = diff_np / (diff_np.max() + 1e-8)

        axes[i, 0].imshow(clean_np)
        axes[i, 0].set_ylabel(f"Sample {i+1}\nTrue: {true_cls}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(overlay(clean_np, heatmap_clean))
        axes[i, 1].axis("off")

        axes[i, 2].imshow(adv_np)
        axes[i, 2].set_xlabel(f"Pred: {adv_cls}", fontsize=9)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(overlay(adv_np, heatmap_adv))
        axes[i, 3].axis("off")

        axes[i, 4].imshow(diff_np)
        axes[i, 4].axis("off")

    fig.suptitle(
        f"GradCAM: Clean vs Adversarial — {model_label}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = f"plots/gradcam_{model_label.replace(' ', '_')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"GradCAM figure saved → {out}")
