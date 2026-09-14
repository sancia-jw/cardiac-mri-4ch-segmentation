#!/usr/bin/env python3
"""
Train a simple 2D multiclass UNet on CINE 4CH_TR slices.

Requires outputs/splits_4ch.csv from create_splits_4ch.py.
"""

from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cine_4ch.config import DEFAULT_IMAGE_SIZE, LABEL_NAMES, NUM_CLASSES, OUTPUTS_DIR
from cine_4ch.dataset import Cine4CHSliceDataset, load_split_cases
from cine_4ch.metrics import combined_loss, multiclass_dice
from cine_4ch.model import UNet2D


def resolve_device(device_arg: str) -> torch.device:
    """Resolve --device auto|cpu|cuda without crashing when CUDA is unavailable."""
    choice = device_arg.lower()
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("WARNING: --device cuda requested but CUDA is not available. Using CPU.")
        return torch.device("cpu")
    raise ValueError(f"Unknown device '{device_arg}'. Use auto, cpu, or cuda.")


def parse_args() -> argparse.Namespace:
    # CPU-friendly defaults; increase batch size on GPU if desired.
    default_batch_size = 4
    default_num_workers = 0 if platform.system() == "Windows" else 0

    parser = argparse.ArgumentParser(description="Train baseline 2D UNet on CINE 4CH.")
    parser.add_argument(
        "--splits-csv",
        type=Path,
        default=OUTPUTS_DIR / "splits_4ch.csv",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=default_num_workers)
    parser.add_argument("--checkpoint-dir", type=Path, default=OUTPUTS_DIR / "checkpoints")
    parser.add_argument("--log-csv", type=Path, default=OUTPUTS_DIR / "training_log.csv")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="auto: CUDA if available, else CPU (default)",
    )
    return parser.parse_args()


@torch.no_grad()
def run_sanity_check(model: UNet2D, loader: DataLoader, device: torch.device) -> None:
    """Load one batch and verify tensor shapes before full training."""
    model.eval()
    images, labels, stems, frames = next(iter(loader))

    print("Sanity check (one batch):")
    print(f"  image tensor shape: {tuple(images.shape)}")
    print(f"  mask tensor shape:  {tuple(labels.shape)}")
    print(f"  unique mask labels: {sorted(labels.unique().tolist())}")
    print(f"  sample stems:       {list(stems)[:min(3, len(stems))]}")

    images = images.to(device)
    logits = model(images)
    print(f"  model output shape: {tuple(logits.shape)}")
    print("Sanity check passed.")
    print()


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for images, labels, _, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        batch_metrics = multiclass_dice(logits, labels)
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    for images, labels, _, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = combined_loss(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / max(len(loader), 1)


def main() -> int:
    args = parse_args()
    if not args.splits_csv.exists():
        print(f"Split file not found: {args.splits_csv}", file=sys.stderr)
        print("Run scripts/create_splits_4ch.py first.", file=sys.stderr)
        return 1

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_csv.parent.mkdir(parents=True, exist_ok=True)

    train_cases = load_split_cases(args.splits_csv, "train")
    val_cases = load_split_cases(args.splits_csv, "val")
    print(f"Train cases: {len(train_cases)}  Val cases: {len(val_cases)}")
    print("Preloading volumes into memory...")

    train_ds = Cine4CHSliceDataset(train_cases, image_size=DEFAULT_IMAGE_SIZE, augment=True)
    val_ds = Cine4CHSliceDataset(val_cases, image_size=DEFAULT_IMAGE_SIZE, augment=False)
    print(f"Train slices: {len(train_ds)}  Val slices: {len(val_ds)}")

    device = resolve_device(args.device)
    use_cuda = device.type == "cuda"
    print(f"Using device: {device}")
    if not use_cuda:
        print("CUDA not in use - training will run on CPU (slower but supported).")
    print()

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )

    model = UNet2D(in_channels=1, num_classes=NUM_CLASSES).to(device)
    run_sanity_check(model, train_loader, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    log_fields = ["epoch", "train_loss", "val_dice_mean"] + [
        f"val_dice_class_{i}" for i in range(NUM_CLASSES)
    ]
    best_dice = -1.0

    with open(args.log_csv, "w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=log_fields)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)

            row = {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_dice_mean": f"{val_metrics['dice_mean']:.6f}",
            }
            for i in range(NUM_CLASSES):
                row[f"val_dice_class_{i}"] = f"{val_metrics[f'dice_class_{i}']:.6f}"

            writer.writerow(row)
            log_file.flush()

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:03d}/{args.epochs}  "
                f"loss={train_loss:.4f}  val_dice={val_metrics['dice_mean']:.4f}  "
                f"({elapsed:.1f}s)"
            )
            for i in range(NUM_CLASSES):
                print(f"  class {i} ({LABEL_NAMES[i]}): {val_metrics[f'dice_class_{i}']:.4f}")

            ckpt_path = args.checkpoint_dir / f"unet_4ch_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice_mean": val_metrics["dice_mean"],
                },
                ckpt_path,
            )

            if val_metrics["dice_mean"] > best_dice:
                best_dice = val_metrics["dice_mean"]
                best_path = args.checkpoint_dir / "unet_4ch_best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_dice_mean": val_metrics["dice_mean"],
                    },
                    best_path,
                )
                print(f"  -> new best checkpoint: {best_path}")

    print(f"Training log saved to {args.log_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
