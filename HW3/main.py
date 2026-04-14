"""
CS515 HW3 — Robustness, AugMix, Adversarial Attacks & Knowledge Distillation.

Entry point that routes to all HW3 tasks based on --mode argument.

Usage examples:
    # 1. Evaluate baseline model on CIFAR-10-C
    python main.py --mode eval_corrupted --eval_model_path models/saved/resnet18_baseline.pth --cifar10c_dir ./data/CIFAR-10-C

    # 2. Fine-tune with AugMix
    python main.py --mode train --use_augmix --model resnet --epochs 50 --run_name resnet18_augmix

    # 3. PGD adversarial evaluation
    python main.py --mode pgd_attack --eval_model_path models/saved/resnet18_baseline.pth
    python main.py --mode pgd_attack --eval_model_path models/saved/resnet18_augmix.pth

    # 4. Grad-CAM visualisation
    python main.py --mode gradcam --eval_model_path models/saved/resnet18_baseline.pth

    # 5. t-SNE visualisation
    python main.py --mode tsne --eval_model_path models/saved/resnet18_baseline.pth

    # 6. Distillation with AugMix teacher
    python main.py --mode distill --model mobilenet --teacher_augmix_path models/saved/resnet18_augmix.pth --run_name mobilenet_kd_augmix_teacher

    # 7. Transfer attack (teacher adversarial examples → student)
    python main.py --mode transfer_attack --eval_model_path models/saved/mobilenet_kd_augmix_teacher.pth --teacher_augmix_path models/saved/resnet18_augmix.pth
"""

import random
import ssl
import numpy as np
import torch

from parameters import get_params, DataConfig, ModelConfig, TrainingConfig
from models.MLP import MLP
from models.CNN import MNIST_CNN, SimpleCNN
from models.VGG import VGG
from models.ResNet import ResNet, BasicBlock
from models.MobileNet import MobileNetV2
from train import run_training
from test import run_test
from robustness import run_corrupted_evaluation, compare_robustness
from pgd_attack import run_pgd_evaluation, generate_adversarial_samples, run_transfer_attack
from gradcam import visualize_gradcam_pairs
from tsne_viz import run_tsne_visualization
from distillation import run_distillation, count_flops

ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def build_model(data_cfg: DataConfig, model_cfg: ModelConfig) -> torch.nn.Module:
    """Instantiate the model specified in model_cfg.

    Args:
        data_cfg: Dataset config (num_classes, input_size).
        model_cfg: Architecture config.

    Returns:
        Instantiated PyTorch model.
    """
    name = model_cfg.model
    nc   = data_cfg.num_classes

    if name == "mlp":
        return MLP(
            input_size   = data_cfg.input_size,
            hidden_sizes = model_cfg.hidden_sizes,
            num_classes  = nc,
            dropout      = model_cfg.dropout,
            activation   = model_cfg.activation,
            batch_norm   = model_cfg.batch_norm,
        )
    if name == "cnn":
        return SimpleCNN(num_classes=nc)
    if name == "vgg":
        return VGG(dept=model_cfg.vgg_depth, num_class=nc)
    if name == "resnet":
        return ResNet(BasicBlock, model_cfg.resnet_layers, num_classes=nc)
    if name == "mobilenet":
        return MobileNetV2(num_classes=nc)

    raise ValueError(f"Unknown model: {name}")


def _load_resnet18(path: str, data_cfg: DataConfig, device: torch.device) -> torch.nn.Module:
    """Helper: load a ResNet-18 from disk.

    Args:
        path: Path to .pth checkpoint.
        data_cfg: Dataset config.
        device: Compute device.

    Returns:
        Loaded ResNet-18.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=data_cfg.num_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Loaded ResNet-18 from: {path}")
    return model


def main() -> None:
    """Parse args and route to the appropriate HW3 task."""
    data_cfg, model_cfg, train_cfg = get_params()
    set_seed(train_cfg.seed)

    device = torch.device(
        train_cfg.device if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device : {device}")
    print(f"Mode   : {train_cfg.mode}")
    print(f"Model  : {model_cfg.model}")
    if train_cfg.use_augmix:
        print(f"AugMix : severity={train_cfg.augmix_severity} "
              f"width={train_cfg.augmix_mixture_width} "
              f"depth={train_cfg.augmix_chain_depth}")

    mode = train_cfg.mode

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Standard training / fine-tuning (with optional AugMix)
    # ─────────────────────────────────────────────────────────────────────────
    if mode in ("train", "both"):
        model = build_model(data_cfg, model_cfg).to(device)
        run_training(model, data_cfg, model_cfg, train_cfg, device)
        if mode == "both":
            run_test(model, data_cfg, train_cfg, device, load_weights=False)
        return

    if mode == "test":
        model = build_model(data_cfg, model_cfg).to(device)
        run_test(model, data_cfg, train_cfg, device)
        return

    # ─────────────────────────────────────────────────────────────────────────
    # 2. CIFAR-10-C robustness evaluation
    # ─────────────────────────────────────────────────────────────────────────
    if mode == "eval_corrupted":
        if train_cfg.eval_model_path is None:
            raise ValueError("--eval_model_path is required for eval_corrupted mode.")

        baseline = _load_resnet18(train_cfg.eval_model_path, data_cfg, device)
        results_baseline = run_corrupted_evaluation(
            baseline, "ResNet18-Baseline", data_cfg, train_cfg, device
        )

        # If AugMix model is also provided, compare
        if train_cfg.teacher_augmix_path is not None:
            augmix_model = _load_resnet18(train_cfg.teacher_augmix_path, data_cfg, device)
            results_augmix = run_corrupted_evaluation(
                augmix_model, "ResNet18-AugMix", data_cfg, train_cfg, device
            )
            compare_robustness(results_baseline, results_augmix)
        return

    # ─────────────────────────────────────────────────────────────────────────
    # 3. PGD adversarial robustness evaluation
    # ─────────────────────────────────────────────────────────────────────────
    if mode == "pgd_attack":
        if train_cfg.eval_model_path is None:
            raise ValueError("--eval_model_path is required for pgd_attack mode.")

        model = _load_resnet18(train_cfg.eval_model_path, data_cfg, device)
        label = ("ResNet18-AugMix"
                 if "augmix" in train_cfg.eval_model_path.lower()
                 else "ResNet18-Baseline")
        run_pgd_evaluation(model, label, data_cfg, train_cfg, device)
        return

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Grad-CAM on adversarial vs clean samples
    # ─────────────────────────────────────────────────────────────────────────
    if mode == "gradcam":
        if train_cfg.eval_model_path is None:
            raise ValueError("--eval_model_path is required for gradcam mode.")

        model = _load_resnet18(train_cfg.eval_model_path, data_cfg, device)
        label = ("ResNet18-AugMix"
                 if "augmix" in train_cfg.eval_model_path.lower()
                 else "ResNet18-Baseline")

        # Generate adversarial samples where clean model is correct but adv fools it
        clean_imgs, adv_imgs, true_labels, adv_preds = generate_adversarial_samples(
            model, data_cfg, train_cfg, device, norm="linf", n_samples=10
        )

        visualize_gradcam_pairs(
            model         = model,
            clean_imgs    = clean_imgs,
            adv_imgs      = adv_imgs,
            true_labels   = true_labels,
            adv_preds     = adv_preds,
            target_layer  = model.layer4[-1],   # last BasicBlock of layer4
            device        = device,
            n_samples     = 2,
            model_label   = label,
            mean          = data_cfg.mean,
            std           = data_cfg.std,
        )
        return

    # ─────────────────────────────────────────────────────────────────────────
    # 5. t-SNE visualisation
    # ─────────────────────────────────────────────────────────────────────────
    if mode == "tsne":
        if train_cfg.eval_model_path is None:
            raise ValueError("--eval_model_path is required for tsne mode.")

        model = _load_resnet18(train_cfg.eval_model_path, data_cfg, device)
        label = ("ResNet18-AugMix"
                 if "augmix" in train_cfg.eval_model_path.lower()
                 else "ResNet18-Baseline")

        # Hook the avgpool output (512-d) as the embedding
        run_tsne_visualization(
            model         = model,
            feature_layer = model.avgpool,
            model_label   = label,
            data_cfg      = data_cfg,
            train_cfg     = train_cfg,
            device        = device,
        )
        return

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Knowledge distillation with AugMix teacher
    # ─────────────────────────────────────────────────────────────────────────
    if mode == "distill":
        teacher_path = train_cfg.teacher_augmix_path or train_cfg.teacher_path
        if teacher_path is None:
            raise ValueError(
                "Provide --teacher_augmix_path (AugMix teacher) or --teacher_path."
            )

        teacher = _load_resnet18(teacher_path, data_cfg, device)
        student = build_model(data_cfg, model_cfg).to(device)

        print("\n── Teacher FLOPs ──")
        count_flops(teacher)
        print("── Student FLOPs ──")
        count_flops(student)

        run_distillation(student, teacher, data_cfg, model_cfg, train_cfg, device)

        # Clean test accuracy after distillation
        run_test(student, data_cfg, train_cfg, device, load_weights=False)
        return

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Transferability: teacher adversarial examples → student
    # ─────────────────────────────────────────────────────────────────────────
    if mode == "transfer_attack":
        if train_cfg.eval_model_path is None or train_cfg.teacher_augmix_path is None:
            raise ValueError(
                "--eval_model_path (student) and --teacher_augmix_path (teacher) "
                "are both required for transfer_attack mode."
            )

        teacher = _load_resnet18(train_cfg.teacher_augmix_path, data_cfg, device)
        student = build_model(data_cfg, model_cfg).to(device)
        student.load_state_dict(
            torch.load(train_cfg.eval_model_path, map_location=device)
        )
        print(f"Student loaded from: {train_cfg.eval_model_path}")

        run_transfer_attack(
            teacher       = teacher,
            student       = student,
            student_label = model_cfg.model,
            data_cfg      = data_cfg,
            train_cfg     = train_cfg,
            device        = device,
        )
        return

    raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
