#!/usr/bin/env python3
"""
Evaluate the baseline 4CH UNet on the held-out test split.

Loads outputs/checkpoints/unet_4ch_best.pt, computes per-class Dice on the
test set, saves metrics CSV, random overlays, and worst-case overlays.
"""

from __future__ import annotations

import argparse
import csv
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cine_4ch.config import DEFAULT_IMAGE_SIZE, LABEL_COLORS, LABEL_NAMES, NUM_CLASSES, OUTPUTS_DIR
from cine_4ch.dataset import Cine4CHSliceDataset, load_split_cases
from cine_4ch.io import choose_representative_frame, load_pair, normalize_image_slice
from cine_4ch.metrics import multiclass_dice
from cine_4ch.model import UNet2D
from cine_4ch.viz import overlay_label


@dataclass
class SlicePrediction:
    stem: str
    frame_idx: int
    dice_foreground: float
    image_2d: np.ndarray
    gt_2d: np.ndarray
    pred_2d: np.ndarray


def resolve_device(device_arg: str) -> torch.device:
    choice = device_arg.lower()
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("WARNING: CUDA not available. Using CPU.")
        return torch.device("cpu")
    raise ValueError(f"Unknown device '{device_arg}'. Use auto, cpu, or cuda.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline UNet on 4CH test split.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=OUTPUTS_DIR / "checkpoints" / "unet_4ch_best.pt",
    )
    parser.add_argument(
        "--splits-csv",
        type=Path,
        default=OUTPUTS_DIR / "splits_4ch.csv",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=OUTPUTS_DIR / "test_metrics_4ch.csv",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=OUTPUTS_DIR / "test_predictions",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=OUTPUTS_DIR / "test_evaluation_summary.md",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0 if platform.system() == "Windows" else 0)
    parser.add_argument("--num-overlays", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def verify_splits(splits_csv: Path) -> dict:
    """Confirm case-level splits with no case ID leakage across splits."""
    df = pd.read_csv(splits_csv)
    df["case_id"] = df["case_id"].astype(str).str.zfill(3)

    split_counts = df["split"].value_counts().to_dict()
    duplicate_cases = df[df.duplicated("case_id", keep=False)]
    multi_split_cases = (
        df.groupby("case_id")["split"].nunique().loc[lambda s: s > 1].index.tolist()
    )

    return {
        "total_rows": len(df),
        "unique_cases": df["case_id"].nunique(),
        "split_counts": split_counts,
        "case_level": len(df) == df["case_id"].nunique(),
        "duplicate_case_rows": len(duplicate_cases),
        "cases_in_multiple_splits": multi_split_cases,
    }


@torch.no_grad()
def evaluate_test_set(model: UNet2D, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for images, labels, _, _ in tqdm(loader, desc="test"):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        batch_metrics = multiclass_dice(logits, labels)
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def collect_slice_predictions(
    model: UNet2D,
    loader: DataLoader,
    device: torch.device,
) -> list[SlicePrediction]:
    """Collect per-slice predictions and foreground Dice for ranking."""
    model.eval()
    results: list[SlicePrediction] = []

    for images, labels, stems, frames in tqdm(loader, desc="slice preds"):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)

        for i in range(images.shape[0]):
            pred = preds[i].cpu().numpy().astype(np.int64)
            gt = labels[i].cpu().numpy().astype(np.int64)
            image_2d = images[i].squeeze(0).cpu().numpy()

            fg_dices = []
            for class_id in range(1, NUM_CLASSES):
                pred_mask = pred == class_id
                gt_mask = gt == class_id
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = pred_mask.sum() + gt_mask.sum()
                dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
                fg_dices.append(float(dice))

            results.append(
                SlicePrediction(
                    stem=stems[i],
                    frame_idx=int(frames[i]),
                    dice_foreground=float(np.mean(fg_dices)),
                    image_2d=image_2d,
                    gt_2d=gt,
                    pred_2d=pred,
                )
            )
    return results


def save_metrics_csv(metrics: dict[str, float], checkpoint: Path, checkpoint_meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"metric": "checkpoint", "value": str(checkpoint)},
        {"metric": "checkpoint_epoch", "value": checkpoint_meta.get("epoch", "")},
        {"metric": "checkpoint_val_dice_mean", "value": checkpoint_meta.get("val_dice_mean", "")},
        {"metric": "test_dice_mean_all_classes", "value": f"{metrics['dice_mean']:.6f}"},
        {"metric": "test_dice_mean_foreground", "value": f"{metrics['dice_mean_foreground']:.6f}"},
    ]
    for class_id in range(NUM_CLASSES):
        rows.append(
            {
                "metric": f"test_dice_class_{class_id}",
                "value": f"{metrics[f'dice_class_{class_id}']:.6f}",
            }
        )
        rows.append(
            {
                "metric": f"test_dice_{LABEL_NAMES[class_id]}",
                "value": f"{metrics[f'dice_{LABEL_NAMES[class_id]}']:.6f}",
            }
        )

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def save_prediction_figure(
    image_2d: np.ndarray,
    gt_2d: np.ndarray,
    pred_2d: np.ndarray,
    stem: str,
    frame_idx: int,
    save_path: Path,
    dice_fg: float | None = None,
) -> None:
    base_gray = normalize_image_slice(image_2d)
    gt_overlay = overlay_label(base_gray, gt_2d)
    pred_overlay = overlay_label(base_gray, pred_2d)

    title = f"Test prediction - {stem} (frame {frame_idx})"
    if dice_fg is not None:
        title += f" | fg Dice={dice_fg:.3f}"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)

    axes[0].imshow(base_gray, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("CINE image")
    axes[0].axis("off")

    axes[1].imshow(gt_overlay)
    axes[1].set_title("GT overlay")
    axes[1].axis("off")

    axes[2].imshow(pred_overlay)
    axes[2].set_title("Prediction overlay")
    axes[2].axis("off")

    labels_present = sorted(set(np.unique(gt_2d).tolist() + np.unique(pred_2d).tolist()))
    patches = [
        mpatches.Patch(
            color=tuple(c / 255.0 for c in LABEL_COLORS[label_id]),
            label=f"{label_id}: {LABEL_NAMES[label_id]}",
        )
        for label_id in labels_present
        if label_id in LABEL_NAMES
    ]
    if patches:
        fig.legend(handles=patches, loc="lower center", ncol=min(len(patches), 3), frameon=False)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_overlay_set(
    items: list[SlicePrediction],
    output_dir: Path,
    prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        out_path = output_dir / f"{prefix}_{item.stem}_frame{item.frame_idx:03d}_fgdice{item.dice_foreground:.3f}.png"
        save_prediction_figure(
            item.image_2d,
            item.gt_2d,
            item.pred_2d,
            item.stem,
            item.frame_idx,
            out_path,
            dice_fg=item.dice_foreground,
        )
        print(f"  saved {out_path.name}")


@torch.no_grad()
def save_random_case_overlays(
    model: UNet2D,
    test_cases,
    device: torch.device,
    output_dir: Path,
    num_overlays: int,
    seed: int,
) -> list[SlicePrediction]:
    """Save one representative-frame overlay per randomly selected test case."""
    model.eval()
    rng = random.Random(seed)
    chosen = rng.sample(test_cases, k=min(num_overlays, len(test_cases)))
    saved: list[SlicePrediction] = []

    for case in sorted(chosen, key=lambda c: c.stem):
        _, label, _ = load_pair(case)
        frame_idx = choose_representative_frame(label)

        ds = Cine4CHSliceDataset([case], image_size=DEFAULT_IMAGE_SIZE, augment=False)
        frame_indices = [idx for idx, (_, fidx) in enumerate(ds.index) if fidx == frame_idx]
        if not frame_indices:
            frame_indices = [0]
        image_t, label_t, _, fidx = ds[frame_indices[0]]

        logits = model(image_t.unsqueeze(0).to(device))
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)
        gt = label_t.numpy().astype(np.int64)
        base_gray = image_t.squeeze(0).numpy()

        fg_dices = []
        for class_id in range(1, NUM_CLASSES):
            pred_mask = pred == class_id
            gt_mask = gt == class_id
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = pred_mask.sum() + gt_mask.sum()
            fg_dices.append((2.0 * intersection + 1e-6) / (union + 1e-6))

        item = SlicePrediction(
            stem=case.stem,
            frame_idx=fidx,
            dice_foreground=float(np.mean(fg_dices)),
            image_2d=base_gray,
            gt_2d=gt,
            pred_2d=pred,
        )
        out_path = output_dir / f"random_{case.stem}_frame{fidx:03d}_fgdice{item.dice_foreground:.3f}.png"
        save_prediction_figure(base_gray, gt, pred, case.stem, fidx, out_path, dice_fg=item.dice_foreground)
        print(f"  saved {out_path.name}")
        saved.append(item)
    return saved


def write_summary(
    path: Path,
    split_info: dict,
    metrics: dict[str, float],
    checkpoint_meta: dict,
    val_dice_best: float,
) -> None:
    test_all = metrics["dice_mean"]
    test_fg = metrics["dice_mean_foreground"]
    gap = val_dice_best - test_all

    lines = [
        "# 4CH Test Evaluation Summary",
        "",
        "## Split integrity",
        f"- Rows in splits CSV: {split_info['total_rows']}",
        f"- Unique case IDs: {split_info['unique_cases']}",
        f"- Case-level split (one row per case): **{split_info['case_level']}**",
        f"- Cases appearing in multiple splits: **{split_info['cases_in_multiple_splits'] or 'none'}**",
        f"- Split counts: {split_info['split_counts']}",
        "",
        "## Test metrics (held-out cases only)",
        f"- Mean Dice (all classes, incl. background): **{test_all:.4f}**",
        f"- Mean Dice (foreground only, classes 1-5): **{test_fg:.4f}**",
        f"- Best checkpoint validation Dice: **{val_dice_best:.4f}** (epoch {checkpoint_meta.get('epoch', 'n/a')})",
        f"- Val-test gap (all-class mean): **{gap:.4f}**",
        "",
        "### Per-class test Dice",
    ]
    for class_id in range(NUM_CLASSES):
        lines.append(
            f"- Class {class_id} ({LABEL_NAMES[class_id]}): {metrics[f'dice_class_{class_id}']:.4f}"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
        ]
    )

    if split_info["case_level"] and not split_info["cases_in_multiple_splits"]:
        lines.append("- Split assignment is case-level with no case ID overlap across train/val/test.")
    else:
        lines.append("- **Warning:** split integrity issue detected; metrics may be inflated.")

    if gap > 0.03:
        lines.append(
            "- Validation Dice is noticeably higher than test Dice, suggesting mild overfitting or split difficulty differences."
        )
    elif gap < -0.01:
        lines.append(
            "- Test Dice is similar to or better than validation Dice; no strong overfitting signal from this gap alone."
        )
    else:
        lines.append(
            "- Validation and test Dice are close; metrics appear consistent with reasonable generalization."
        )

    if metrics["dice_class_0"] > 0.98:
        lines.append(
            "- Background Dice is very high and inflates the all-class mean; foreground-only Dice is the more informative headline metric."
        )

    if test_fg < 0.80:
        lines.append("- Foreground performance is moderate; review worst-case overlays for failure modes.")
    else:
        lines.append("- Foreground performance is solid for a simple baseline UNet on held-out cases.")

    lines.append(
        "- Temporal frames from the same case stay within one split, so frame-level training slices do not leak test cases across splits."
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    if not args.splits_csv.exists():
        print(f"Split file not found: {args.splits_csv}", file=sys.stderr)
        return 1

    split_info = verify_splits(args.splits_csv)
    print("Split verification:")
    print(f"  case-level split: {split_info['case_level']}")
    print(f"  unique cases: {split_info['unique_cases']}")
    print(f"  split counts: {split_info['split_counts']}")
    print(f"  cases in multiple splits: {split_info['cases_in_multiple_splits'] or 'none'}")
    print()

    device = resolve_device(args.device)
    use_cuda = device.type == "cuda"
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    val_dice_best = float(checkpoint.get("val_dice_mean", 0.0))
    print(
        f"Loaded checkpoint: {args.checkpoint.name} "
        f"(epoch={checkpoint.get('epoch')}, val_dice={val_dice_best:.4f})"
    )

    test_cases = load_split_cases(args.splits_csv, "test")
    print(f"Test cases: {len(test_cases)}")
    test_ds = Cine4CHSliceDataset(test_cases, image_size=DEFAULT_IMAGE_SIZE, augment=False)
    print(f"Test slices: {len(test_ds)}")

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )

    model = UNet2D(in_channels=1, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_test_set(model, test_loader, device)
    save_metrics_csv(metrics, args.checkpoint, checkpoint, args.metrics_csv)

    print()
    print(f"Test Dice (all classes): {metrics['dice_mean']:.4f}")
    print(f"Test Dice (foreground):    {metrics['dice_mean_foreground']:.4f}")
    for class_id in range(NUM_CLASSES):
        print(f"  class {class_id} ({LABEL_NAMES[class_id]}): {metrics[f'dice_class_{class_id}']:.4f}")
    print(f"Saved metrics to {args.metrics_csv}")

    slice_preds = collect_slice_predictions(model, test_loader, device)
    worst_items = sorted(slice_preds, key=lambda x: x.dice_foreground)[: args.num_overlays]

    random_dir = args.pred_dir / "random"
    worst_dir = args.pred_dir / "worst"

    print()
    print(f"Saving {args.num_overlays} random test overlays to {random_dir}:")
    save_random_case_overlays(
        model, test_cases, device, random_dir, args.num_overlays, args.seed
    )

    print()
    print(f"Saving {args.num_overlays} worst test overlays to {worst_dir}:")
    save_overlay_set(worst_items, worst_dir, prefix="worst")

    write_summary(args.summary_path, split_info, metrics, checkpoint, val_dice_best)
    print()
    print(f"Saved summary to {args.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
