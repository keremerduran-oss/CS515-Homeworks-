import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class DataConfig:
    """Dataset and normalization configuration."""
    dataset: str
    data_dir: str
    num_workers: int
    mean: Tuple
    std: Tuple
    input_size: int
    num_classes: int


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model: str
    hidden_sizes: List[int]
    dropout: float
    activation: str
    batch_norm: bool
    vgg_depth: str
    resnet_layers: List[int]


@dataclass
class TrainingConfig:
    """Training hyperparameters and run settings."""
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    scheduler: str
    reg_type: str
    reg_lambda: float
    seed: int
    device: str
    save_path: str
    log_interval: int
    mode: str
    run_name: str
    # Transfer learning
    transfer_option: Optional[str]
    # Knowledge distillation / label smoothing
    label_smoothing: float
    temperature: float
    alpha: float
    teacher_path: Optional[str]
    distill_mode: str
    # HW3 additions
    use_augmix: bool
    augmix_severity: int
    augmix_mixture_width: int
    augmix_chain_depth: int
    augmix_alpha: float
    # PGD attack
    pgd_eps_linf: float
    pgd_eps_l2: float
    pgd_steps: int
    pgd_step_size_linf: float
    pgd_step_size_l2: float
    # Evaluation paths
    cifar10c_dir: str
    eval_model_path: Optional[str]
    teacher_augmix_path: Optional[str]


def get_params() -> Tuple[DataConfig, ModelConfig, TrainingConfig]:
    """Parse command-line arguments and return structured config dataclasses."""
    parser = argparse.ArgumentParser(
        description="CS515 HW3 - Robustness, AugMix, Adversarial Attacks, KD"
    )

    # ── Mode & data ──────────────────────────────────────────────────────────
    parser.add_argument("--mode", choices=["train", "test", "both", "eval_corrupted",
                                           "pgd_attack", "gradcam", "tsne",
                                           "distill", "transfer_attack"], default="both")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="cifar10")
    parser.add_argument("--model", choices=["mlp", "cnn", "vgg", "resnet", "mobilenet"],
                        default="resnet")

    # ── Training hyperparameters ─────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--activation", choices=["relu", "gelu"], default="relu")
    parser.add_argument("--batch_norm", type=lambda x: x.lower() != "false", default=True)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--reg_type", choices=["l1", "l2", "none"], default="none")
    parser.add_argument("--reg_lambda", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["step", "cosine", "plateau"], default="cosine")
    parser.add_argument("--vgg_depth", choices=["11", "13", "16", "19"], default="16")
    parser.add_argument("--resnet_layers", type=int, nargs=4, default=[2, 2, 2, 2],
                        metavar=("L1", "L2", "L3", "L4"))

    # ── Transfer learning ────────────────────────────────────────────────────
    parser.add_argument("--transfer_option", choices=["option1", "option2"], default=None)

    # ── Label smoothing & KD ─────────────────────────────────────────────────
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--teacher_path", type=str, default=None)
    parser.add_argument("--distill_mode", choices=["standard", "soft_targets_only"],
                        default="standard")

    # ── AugMix ───────────────────────────────────────────────────────────────
    parser.add_argument("--use_augmix", action="store_true",
                        help="Enable AugMix data augmentation during training.")
    parser.add_argument("--augmix_severity", type=int, default=3,
                        help="Severity of AugMix operations (1-10).")
    parser.add_argument("--augmix_mixture_width", type=int, default=3,
                        help="Number of augmentation chains to mix.")
    parser.add_argument("--augmix_chain_depth", type=int, default=-1,
                        help="Depth of augmentation chains (-1 = random 1-3).")
    parser.add_argument("--augmix_alpha", type=float, default=1.0,
                        help="Dirichlet distribution alpha for AugMix mixing.")

    # ── PGD attack ───────────────────────────────────────────────────────────
    parser.add_argument("--pgd_eps_linf", type=float, default=4/255,
                        help="L-inf epsilon for PGD attack.")
    parser.add_argument("--pgd_eps_l2", type=float, default=0.25,
                        help="L2 epsilon for PGD attack.")
    parser.add_argument("--pgd_steps", type=int, default=20,
                        help="Number of PGD steps (PGD-20).")
    parser.add_argument("--pgd_step_size_linf", type=float, default=1/255,
                        help="Step size for L-inf PGD.")
    parser.add_argument("--pgd_step_size_l2", type=float, default=0.05,
                        help="Step size for L2 PGD.")

    # ── Evaluation paths ─────────────────────────────────────────────────────
    parser.add_argument("--cifar10c_dir", type=str, default="./data/CIFAR-10-C",
                        help="Path to the CIFAR-10-C directory.")
    parser.add_argument("--eval_model_path", type=str, default=None,
                        help="Path to a saved model to evaluate (robustness / PGD).")
    parser.add_argument("--teacher_augmix_path", type=str, default=None,
                        help="Path to AugMix-trained teacher for distillation/transfer.")

    args = parser.parse_args()

    # ── Derived dataset config ───────────────────────────────────────────────
    if args.dataset == "mnist":
        input_size = 784
        mean, std = (0.1307,), (0.3081,)
    else:
        input_size = 3072
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)

    # ── Run name ─────────────────────────────────────────────────────────────
    if args.run_name:
        run_name = args.run_name
    elif args.use_augmix:
        run_name = f"{args.model}_augmix_sev{args.augmix_severity}"
    elif args.transfer_option:
        run_name = f"transfer_{args.transfer_option}_resnet18"
    else:
        run_name = (
            f"{args.model}_act{args.activation}_do{args.dropout}"
            f"_bn{args.batch_norm}_reg{args.reg_type}_sch{args.scheduler}"
            f"_ls{args.label_smoothing}"
        )

    save_path = f"models/saved/{run_name}.pth"

    data_cfg = DataConfig(
        dataset     = args.dataset,
        data_dir    = "./data",
        num_workers = 4,
        mean        = mean,
        std         = std,
        input_size  = input_size,
        num_classes = 10,
    )

    model_cfg = ModelConfig(
        model         = args.model,
        hidden_sizes  = args.hidden_sizes,
        dropout       = args.dropout,
        activation    = args.activation,
        batch_norm    = args.batch_norm,
        vgg_depth     = args.vgg_depth,
        resnet_layers = args.resnet_layers,
    )

    train_cfg = TrainingConfig(
        epochs               = args.epochs,
        batch_size           = args.batch_size,
        learning_rate        = args.lr,
        weight_decay         = 1e-4,
        scheduler            = args.scheduler,
        reg_type             = args.reg_type,
        reg_lambda           = args.reg_lambda,
        seed                 = 42,
        device               = args.device,
        save_path            = save_path,
        log_interval         = 100,
        mode                 = args.mode,
        run_name             = run_name,
        transfer_option      = args.transfer_option,
        label_smoothing      = args.label_smoothing,
        temperature          = args.temperature,
        alpha                = args.alpha,
        teacher_path         = args.teacher_path,
        distill_mode         = args.distill_mode,
        use_augmix           = args.use_augmix,
        augmix_severity      = args.augmix_severity,
        augmix_mixture_width = args.augmix_mixture_width,
        augmix_chain_depth   = args.augmix_chain_depth,
        augmix_alpha         = args.augmix_alpha,
        pgd_eps_linf         = args.pgd_eps_linf,
        pgd_eps_l2           = args.pgd_eps_l2,
        pgd_steps            = args.pgd_steps,
        pgd_step_size_linf   = args.pgd_step_size_linf,
        pgd_step_size_l2     = args.pgd_step_size_l2,
        cifar10c_dir         = args.cifar10c_dir,
        eval_model_path      = args.eval_model_path,
        teacher_augmix_path  = args.teacher_augmix_path,
    )

    return data_cfg, model_cfg, train_cfg
