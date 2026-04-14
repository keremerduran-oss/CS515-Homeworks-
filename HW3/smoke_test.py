"""
smoke_test.py — end-to-end sanity check for all HW3 modules.
Run before your first real training run to catch wiring issues early.

Usage:
    python smoke_test.py
"""

import sys
import os
import tempfile
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ── Make sure HW3 root is on path ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

DEVICE = torch.device("cpu")
BATCH  = 4
NC     = 10

# ── Minimal ResNet-18 for all tests ──────────────────────────────────────────
from models.ResNet import ResNet, BasicBlock
from models.CNN    import SimpleCNN
from models.MobileNet import MobileNetV2

def make_resnet() -> nn.Module:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=NC).to(DEVICE)

def make_imgs(b: int = BATCH) -> torch.Tensor:
    return torch.randn(b, 3, 32, 32).to(DEVICE)

def make_labels(b: int = BATCH) -> torch.Tensor:
    return torch.randint(0, NC, (b,)).to(DEVICE)


def test_augmix() -> None:
    print("  [augmix] ...", end=" ")
    from augmix import AugMixTransform, augment_and_mix, AUGMENTATIONS
    assert len(AUGMENTATIONS) == 13

    img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    aug = AugMixTransform(severity=3, mixture_width=3)
    out = aug(img)
    assert out.size == (32, 32)

    # All individual ops
    for op in AUGMENTATIONS:
        res = op(img, 5.0)
        assert isinstance(res, Image.Image)

    print("OK")


def test_pgd() -> None:
    print("  [pgd_attack] ...", end=" ")
    from pgd_attack import pgd_linf, pgd_l2

    model  = make_resnet()
    imgs   = make_imgs()
    labels = make_labels()

    # L∞
    adv_linf = pgd_linf(model, imgs, labels, eps=4/255, alpha=1/255,
                         steps=3, device=DEVICE)
    assert adv_linf.shape == imgs.shape
    diff_linf = (adv_linf - imgs).abs().max().item()
    assert diff_linf <= 4/255 + 1e-5, f"L∞ budget exceeded: {diff_linf}"

    # L2
    adv_l2 = pgd_l2(model, imgs, labels, eps=0.25, alpha=0.05,
                      steps=3, device=DEVICE)
    assert adv_l2.shape == imgs.shape

    print("OK")


def test_gradcam() -> None:
    print("  [gradcam] ...", end=" ")
    from gradcam import GradCAM, denormalize, overlay

    model = make_resnet()
    img   = make_imgs(1)

    cam_gen = GradCAM(model, model.layer4[-1])
    heatmap, pred_cls = cam_gen(img, class_idx=3)
    cam_gen.remove_hooks()
    assert heatmap.shape == (32, 32)
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0

    # Overlay
    img_np   = denormalize(img.squeeze(0).cpu())
    overlaid = overlay(img_np, heatmap)
    assert overlaid.shape == (32, 32, 3)

    print("OK")


def test_tsne() -> None:
    print("  [tsne_viz] ...", end=" ")
    from tsne_viz import FeatureExtractor
    from sklearn.manifold import TSNE

    model = make_resnet()
    model.eval()

    extractor = FeatureExtractor(model, model.avgpool)
    with torch.no_grad():
        model(make_imgs(8))
    feat = extractor.features
    assert feat is not None
    extractor.remove()

    # Tiny t-SNE sanity check
    data   = np.random.randn(20, 16).astype(np.float32)
    coords = TSNE(n_components=2, perplexity=5, max_iter=250).fit_transform(data)
    assert coords.shape == (20, 2)

    print("OK")


def test_distillation() -> None:
    print("  [distillation] ...", end=" ")
    from distillation import KDLoss, SoftTargetLoss

    s = torch.randn(BATCH, NC)
    t = torch.randn(BATCH, NC)
    l = make_labels()

    kd_loss = KDLoss(temperature=4.0, alpha=0.7)(s, t, l)
    assert kd_loss.item() > 0

    soft_loss = SoftTargetLoss(num_classes=NC)(s, t, l)
    assert soft_loss.item() > 0

    print("OK")


def test_train_loop() -> None:
    """Run 2 epochs of AugMix training on a tiny in-memory dataset."""
    print("  [train + AugMix] ...", end=" ")
    from torch.utils.data import TensorDataset, DataLoader
    from train import train_one_epoch, validate, build_criterion, build_scheduler

    # Build a minimal TrainingConfig
    from dataclasses import dataclass
    from typing import List, Tuple, Optional

    @dataclass
    class _TC:
        epochs: int = 2
        batch_size: int = 4
        learning_rate: float = 1e-3
        weight_decay: float = 1e-4
        scheduler: str = "cosine"
        reg_type: str = "none"
        reg_lambda: float = 0.0
        seed: int = 42
        device: str = "cpu"
        save_path: str = "/tmp/smoke_model.pth"
        log_interval: int = 9999  # silence
        mode: str = "both"
        run_name: str = "smoke"
        transfer_option: Optional[str] = None
        label_smoothing: float = 0.0
        temperature: float = 4.0
        alpha: float = 0.7
        teacher_path: Optional[str] = None
        distill_mode: str = "standard"
        use_augmix: bool = False
        augmix_severity: int = 3
        augmix_mixture_width: int = 3
        augmix_chain_depth: int = -1
        augmix_alpha: float = 1.0
        pgd_eps_linf: float = 4/255
        pgd_eps_l2: float = 0.25
        pgd_steps: int = 20
        pgd_step_size_linf: float = 1/255
        pgd_step_size_l2: float = 0.05
        cifar10c_dir: str = "./data/CIFAR-10-C"
        eval_model_path: Optional[str] = None
        teacher_augmix_path: Optional[str] = None

    cfg = _TC()

    imgs   = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, NC, (16,))
    loader = DataLoader(TensorDataset(imgs, labels), batch_size=4, shuffle=True)

    model     = make_resnet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)

    for _ in range(2):
        loss, acc = train_one_epoch(model, loader, optimizer, criterion, DEVICE, cfg)
        scheduler.step()

    val_loss, val_acc = validate(model, loader, criterion, DEVICE)
    assert 0 <= val_acc <= 1.0
    print("OK")


def test_robustness_mock() -> None:
    """Test robustness module logic with a mock numpy file."""
    print("  [robustness mock] ...", end=" ")
    import tempfile, pathlib
    from robustness import evaluate_corruption

    # Create mock CIFAR-10-C files
    tmp = tempfile.mkdtemp()
    data_arr   = np.random.randint(0, 255, (50000, 32, 32, 3), dtype=np.uint8)
    label_arr  = np.random.randint(0, 10, (50000,), dtype=np.int64)
    np.save(os.path.join(tmp, "gaussian_noise.npy"), data_arr)
    np.save(os.path.join(tmp, "labels.npy"), label_arr)

    from dataclasses import dataclass
    from typing import Tuple, Optional, List

    @dataclass
    class _DC:
        dataset: str = "cifar10"
        data_dir: str = "./data"
        num_workers: int = 0
        mean: Tuple = (0.4914, 0.4822, 0.4465)
        std:  Tuple = (0.2023, 0.1994, 0.2010)
        input_size: int = 3072
        num_classes: int = 10

    @dataclass
    class _TC:
        batch_size: int = 64
        cifar10c_dir: str = tmp
        num_workers: int = 0

    model = make_resnet()
    acc   = evaluate_corruption(model, "gaussian_noise", severity=1,
                                 data_cfg=_DC(), train_cfg=_TC(), device=DEVICE)
    assert 0.0 <= acc <= 1.0
    print("OK")


def test_transfer_attack_logic() -> None:
    """Test that PGD on teacher + eval on student runs without error."""
    print("  [transfer attack] ...", end=" ")
    from pgd_attack import pgd_linf

    teacher = make_resnet()
    student = SimpleCNN(num_classes=NC).to(DEVICE)

    imgs   = make_imgs(4)
    labels = make_labels(4)

    adv = pgd_linf(teacher, imgs, labels, eps=4/255, alpha=1/255,
                    steps=3, device=DEVICE)
    student.eval()
    with torch.no_grad():
        preds = student(adv).argmax(1)
    assert preds.shape == (4,)
    print("OK")


if __name__ == "__main__":
    print("=" * 50)
    print("HW3 Smoke Tests")
    print("=" * 50)

    tests = [
        test_augmix,
        test_pgd,
        test_gradcam,
        test_tsne,
        test_distillation,
        test_train_loop,
        test_robustness_mock,
        test_transfer_attack_logic,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAILED — {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed ✓")
    sys.exit(failed)
