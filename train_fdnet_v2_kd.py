from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vit_b_16

from fdnet_v2_kd import FDNetV2, distillation_loss, freeze_module


EXPECTED_CLASSES = ["Cut", "Hole", "Stain", "ThreadError"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def assert_expected_classes(classes: list[str]) -> None:
    if list(classes) != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected ImageFolder class order {EXPECTED_CLASSES}, but found {list(classes)}. "
            "Keep the train/valid/test folder names identical and do not remap labels between teacher and student."
        )


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if isinstance(checkpoint, Mapping):
        for key in (
            "model_state_dict",
            "state_dict",
            "best_model_state_dict",
            "model_state",
            "model",
            "net",
        ):
            value = checkpoint.get(key)
            if isinstance(value, nn.Module):
                return value.state_dict()
            if isinstance(value, Mapping) and value and all(torch.is_tensor(v) for v in value.values()):
                return value

        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint

    raise ValueError(
        "Could not find a model state_dict in the checkpoint. "
        "Expected a plain state_dict or a dictionary containing model_state_dict/state_dict/model."
    )


def strip_prefix_if_present(
    state_dict: Mapping[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    keys = list(state_dict.keys())
    if keys and all(key.startswith(prefix) for key in keys):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return dict(state_dict)


def clean_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = dict(state_dict)
    for prefix in ("module.", "_orig_mod.", "model."):
        cleaned = strip_prefix_if_present(cleaned, prefix)
    return cleaned


def compute_classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


class PairedImageFolder(Dataset):
    """One geometric/colour augmentation, then separate teacher/student normalization."""

    def __init__(self, root: str | Path, base_transform: transforms.Compose) -> None:
        self.dataset = ImageFolder(str(root), transform=None)
        self.base_transform = base_transform
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.targets = self.dataset.targets
        self.samples = self.dataset.samples
        self.student_normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.teacher_normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        assert_expected_classes(self.classes)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        tensor = self.base_transform(image)
        student_image = self.student_normalize(tensor.clone())
        teacher_image = self.teacher_normalize(tensor.clone())
        return student_image, teacher_image, target


def build_base_transform(training: bool) -> transforms.Compose:
    # IMPORTANT FOR A FAIR PAPER COMPARISON:
    # If your original train_fdnet_v2.py used a different augmentation recipe,
    # copy that exact augmentation here and keep KD as the only experimental change.
    if training:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.05),
                transforms.ToTensor(),
            ]
        )
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


def resolve_split(data_root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = data_root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find any of {names} under {data_root}. "
        "Expected classification_dataset/train, valid (or val), and test."
    )


def make_weighted_sampler(dataset: PairedImageFolder) -> WeightedRandomSampler:
    targets = torch.tensor(dataset.targets, dtype=torch.long)
    counts = torch.bincount(targets, minlength=len(dataset.classes)).float()
    if torch.any(counts == 0):
        raise ValueError(f"At least one class has zero training samples: counts={counts.tolist()}")
    class_weights = 1.0 / counts
    sample_weights = class_weights[targets]
    return WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(sample_weights),
        replacement=True,
    )


def worker_seed_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_loaders(data_root: Path, batch_size: int, workers: int, seed: int):
    train_dir = resolve_split(data_root, ("train",))
    val_dir = resolve_split(data_root, ("valid", "val", "validation"))
    test_dir = resolve_split(data_root, ("test",))

    train_ds = PairedImageFolder(train_dir, build_base_transform(training=True))
    val_ds = PairedImageFolder(val_dir, build_base_transform(training=False))
    test_ds = PairedImageFolder(test_dir, build_base_transform(training=False))

    if train_ds.class_to_idx != val_ds.class_to_idx or train_ds.class_to_idx != test_ds.class_to_idx:
        raise ValueError("Class-to-index mapping differs across train/validation/test splits.")

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=make_weighted_sampler(train_ds),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=worker_seed_fn,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_seed_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_seed_fn,
    )
    return train_loader, val_loader, test_loader, train_ds.classes


def build_teacher(checkpoint_path: Path, device: torch.device) -> nn.Module:
    teacher = vit_b_16(weights=None)
    teacher.heads.head = nn.Linear(teacher.heads.head.in_features, len(EXPECTED_CLASSES))

    # Only load checkpoints you created or trust; PyTorch checkpoints may use pickle.
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = clean_state_dict(extract_state_dict(checkpoint))
    try:
        teacher.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "The teacher checkpoint does not match torchvision.models.vit_b_16 with a 4-class head. "
            "Confirm that this is the best ViT-B/16 checkpoint used for the 95.08% baseline."
        ) from exc

    teacher.to(device)
    freeze_module(teacher)
    return teacher


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_f1: float,
    args: argparse.Namespace,
    classes: list[str],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_f1": best_val_f1,
        "classes": classes,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "seed": args.seed,
        "model_name": "FD-Net V2-KD",
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
    }
    torch.save(payload, path)


def load_student_checkpoint(path: Path, device: torch.device) -> FDNetV2:
    model = FDNetV2(num_classes=len(EXPECTED_CLASSES))
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state_dict = clean_state_dict(extract_state_dict(checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def train_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    temperature: float,
    alpha: float,
    amp_enabled: bool,
) -> dict[str, float]:
    student.train()
    teacher.eval()
    running_total = 0.0
    running_ce = 0.0
    running_kd = 0.0
    seen = 0

    for student_images, teacher_images, targets in loader:
        student_images = student_images.to(device, non_blocking=True)
        teacher_images = teacher_images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                teacher_logits = teacher(teacher_images)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            student_logits = student(student_images)
            loss, ce, kd = distillation_loss(
                student_logits,
                teacher_logits,
                targets,
                temperature=temperature,
                alpha=alpha,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch = targets.size(0)
        seen += batch
        running_total += loss.item() * batch
        running_ce += ce.item() * batch
        running_kd += kd.item() * batch

    return {
        "train_loss": running_total / seen,
        "train_ce": running_ce / seen,
        "train_kd": running_kd / seen,
    }


@torch.no_grad()
def evaluate_student(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict[str, float], list[int], list[int]]:
    model.eval()
    loss_sum = 0.0
    seen = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for student_images, _teacher_images, targets in loader:
        student_images = student_images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(student_images)
        loss = nn.functional.cross_entropy(logits, targets)
        preds = logits.argmax(dim=1)

        batch = targets.size(0)
        seen += batch
        loss_sum += loss.item() * batch
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    return loss_sum / seen, compute_classification_metrics(y_true, y_pred), y_true, y_pred


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def train_command(args: argparse.Namespace) -> None:
    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(torch.cuda.is_available() and not args.no_amp)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _test_loader, classes = build_loaders(
        Path(args.data_root), args.batch_size, args.workers, args.seed
    )

    teacher = build_teacher(Path(args.teacher_checkpoint), device)
    student = FDNetV2(num_classes=len(classes)).to(device)

    parameter_count = sum(p.numel() for p in student.parameters())
    if parameter_count != 3_068_448:
        raise RuntimeError(f"FD-Net V2 parameter count changed unexpectedly: {parameter_count:,}")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val_f1 = -1.0
    patience_counter = 0
    best_path = output_dir / "fdnet_v2_kd_best.pth"
    history: list[dict[str, float]] = []

    print(f"Device: {device}")
    print(f"AMP training: {amp_enabled}")
    print(f"Teacher: ViT-B/16 (frozen)")
    print(f"Student: FD-Net V2 ({parameter_count:,} parameters)")
    print(f"KD settings: T={args.temperature}, alpha(CE)={args.alpha}")
    print("IMPORTANT: test split is NOT evaluated by the train command.")

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            student,
            teacher,
            train_loader,
            optimizer,
            scaler,
            device,
            args.temperature,
            args.alpha,
            amp_enabled,
        )
        val_loss, val_metrics, _, _ = evaluate_student(student, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": lr,
            **train_stats,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
        }
        history.append(row)
        write_history(output_dir / "history.csv", history)

        print(
            f"Epoch {epoch:03d} | train={train_stats['train_loss']:.4f} "
            f"(CE={train_stats['train_ce']:.4f}, KD={train_stats['train_kd']:.4f}) | "
            f"val_loss={val_loss:.4f} | val_acc={val_metrics['accuracy']*100:.2f}% | "
            f"val_F1={val_metrics['f1_macro']*100:.2f}%"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            patience_counter = 0
            save_checkpoint(best_path, student, optimizer, epoch, best_val_f1, args, classes)
        else:
            patience_counter += 1

        scheduler.step()

        if patience_counter >= args.patience:
            print(f"Early stopping after {epoch} epochs (patience={args.patience}).")
            break

    summary = {
        "best_validation_macro_f1": best_val_f1,
        "best_checkpoint": str(best_path),
        "temperature": args.temperature,
        "alpha_ce": args.alpha,
        "seed": args.seed,
        "student_parameters": parameter_count,
        "test_evaluated": False,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Best validation macro-F1: {best_val_f1*100:.2f}%")
    print(f"Saved: {best_path}")
    print("Choose KD hyperparameters using VALIDATION results only. Run the 'test' command once after selection.")


def save_confusion_matrix(y_true: list[int], y_pred: list[int], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(EXPECTED_CLASSES))))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EXPECTED_CLASSES)
    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, values_format="d", colorbar=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def test_command(args: argparse.Namespace) -> None:
    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _train_loader, _val_loader, test_loader, classes = build_loaders(
        Path(args.data_root), args.batch_size, args.workers, args.seed
    )
    assert_expected_classes(classes)
    model = load_student_checkpoint(Path(args.student_checkpoint), device)

    test_loss, metrics, y_true, y_pred = evaluate_student(model, test_loader, device)
    report = {
        "test_loss": test_loss,
        **metrics,
        "samples": len(y_true),
        "checkpoint": str(args.student_checkpoint),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix.png")

    print("Independent test result")
    print(f"Samples:   {len(y_true)}")
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision_macro']*100:.2f}% (macro)")
    print(f"Recall:    {metrics['recall_macro']*100:.2f}% (macro)")
    print(f"F1:        {metrics['f1_macro']*100:.2f}% (macro)")
    print(f"Saved metrics to {output_dir / 'test_metrics.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ViT-B/16 -> FD-Net V2 knowledge distillation without test-set leakage"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train KD student; validation only, no test evaluation")
    train.add_argument("--data-root", default="classification_dataset")
    train.add_argument("--teacher-checkpoint", required=True)
    train.add_argument("--output-dir", default="results/fdnet_v2_kd_T4_A05")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--patience", type=int, default=20)
    train.add_argument("--temperature", type=float, default=4.0)
    train.add_argument("--alpha", type=float, default=0.5, help="Weight on ground-truth CE; KD weight is 1-alpha")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--no-amp", action="store_true")
    train.add_argument("--deterministic", action="store_true")
    train.set_defaults(func=train_command)

    test = subparsers.add_parser("test", help="Evaluate one already-selected KD checkpoint on the test set")
    test.add_argument("--data-root", default="classification_dataset")
    test.add_argument("--student-checkpoint", required=True)
    test.add_argument("--output-dir", default="results/fdnet_v2_kd_final_test")
    test.add_argument("--batch-size", type=int, default=64)
    test.add_argument("--workers", type=int, default=4)
    test.add_argument("--seed", type=int, default=42)
    test.add_argument("--deterministic", action="store_true")
    test.set_defaults(func=test_command)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "temperature", 1.0) <= 0:
        raise ValueError("--temperature must be > 0")
    if hasattr(args, "alpha") and not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")
    args.func(args)


if __name__ == "__main__":
    main()
