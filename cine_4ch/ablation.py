"""Training and evaluation utilities for EMD ablation experiments."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cine_4ch.config import DEFAULT_IMAGE_SIZE, LABEL_COLORS, LABEL_NAMES, NUM_CLASSES, OUTPUTS_DIR
from cine_4ch.dataset import Cine4CHSliceDataset, load_split_cases
from cine_4ch.io import choose_representative_frame, load_pair
from cine_4ch.metrics import combined_loss, multiclass_dice
from cine_4ch.model import UNet2D
from cine_4ch.viz import overlay_label
from src.preprocessing.emd_enhancement import EMEnhancementConfig, output_channels, safe_minmax_normalize


@dataclass
class AblationHyperparams:
    """Fixed across all ablation runs."""

    epochs: int = 15
    batch_size: int = 4
    lr: float = 1e-3
    seed: int = 42
    num_workers: int = 0


@dataclass
class AblationRunSpec:
    run_id: str
    emd_config: EMEnhancementConfig
    description: str = ""


def default_ablation_runs() -> List[AblationRunSpec]:
    """Preprocessing modes compared against the original baseline."""
    return [
        AblationRunSpec(
            run_id="original",
            description="Normalized original MRI only (no EMD).",
            emd_config=EMEnhancementConfig(mode="original", imf_indices=[]),
        ),
        AblationRunSpec(
            run_id="subtract_imf_0",
            description="Original minus IMF [0] (highest-frequency detail).",
            emd_config=EMEnhancementConfig(mode="subtract", imf_indices=[0]),
        ),
        AblationRunSpec(
            run_id="subtract_imf_1",
            description="Original minus IMF [1] (mid-scale structure).",
            emd_config=EMEnhancementConfig(mode="subtract", imf_indices=[1]),
        ),
        AblationRunSpec(
            run_id="subtract_imf_0_1",
            description="Original minus IMFs [0, 1].",
            emd_config=EMEnhancementConfig(mode="subtract", imf_indices=[0, 1]),
        ),
        AblationRunSpec(
            run_id="subtract_imf_1_2",
            description="Original minus IMFs [1, 2].",
            emd_config=EMEnhancementConfig(mode="subtract", imf_indices=[1, 2]),
        ),
        AblationRunSpec(
            run_id="subtract_imf_2",
            description="Original minus IMF [2] (lower mid-frequency structure).",
            emd_config=EMEnhancementConfig(mode="subtract", imf_indices=[2]),
        ),
        AblationRunSpec(
            run_id="imf_only_1",
            description="IMF [1] only.",
            emd_config=EMEnhancementConfig(mode="imf_only", imf_indices=[1]),
        ),
        AblationRunSpec(
            run_id="imf_only_0_1",
            description="Sum of IMFs [0, 1].",
            emd_config=EMEnhancementConfig(mode="imf_only", imf_indices=[0, 1]),
        ),
        AblationRunSpec(
            run_id="concat_imf_1",
            description="Two-channel input: original + IMF [1].",
            emd_config=EMEnhancementConfig(mode="concat", imf_indices=[1], concat_sum_imfs=True),
        ),
        AblationRunSpec(
            run_id="subtract_imf_neg2_3",
            description="Original minus IMFs [-2, -3] (slow components, keep residual).",
            emd_config=EMEnhancementConfig(mode="subtract", imf_indices=[-2, -3]),
        ),
    ]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    raise ValueError(f"Unknown device: {device_arg}")


def config_to_dict(cfg: EMEnhancementConfig) -> Dict[str, Any]:
    return asdict(cfg)


def save_run_config(
    run_dir: Path,
    spec: AblationRunSpec,
    hyperparams: AblationHyperparams,
    in_channels: int,
) -> None:
    payload = {
        "run_id": spec.run_id,
        "description": spec.description,
        "emd_config": config_to_dict(spec.emd_config),
        "hyperparams": asdict(hyperparams),
        "in_channels": in_channels,
        "num_classes": NUM_CLASSES,
    }
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


@torch.no_grad()
def evaluate_loader(model: UNet2D, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
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
        loss = combined_loss(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / max(len(loader), 1)


def _default_preprocess_workers() -> int:
    import os

    cpu = os.cpu_count() or 2
    # Leave one core free; EMD benefits from process parallelism on CPU.
    return max(1, min(8, cpu - 1))


def train_run(
    spec: AblationRunSpec,
    hyperparams: AblationHyperparams,
    splits_csv: Path,
    run_dir: Path,
    device: torch.device,
    resume: bool = False,
    preprocess_workers: int | None = None,
) -> Dict[str, Any]:
    set_seed(hyperparams.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    workers = _default_preprocess_workers() if preprocess_workers is None else preprocess_workers

    in_channels = output_channels(spec.emd_config)
    save_run_config(run_dir, spec, hyperparams, in_channels)

    train_cases = load_split_cases(splits_csv, "train")
    val_cases = load_split_cases(splits_csv, "val")

    print(f"\n=== Run: {spec.run_id} ===")
    print(f"Preprocessing: {spec.emd_config.mode}  IMF={spec.emd_config.imf_indices}")
    print(f"EMD preprocess workers: {workers}")
    print("Precomputing training preprocessing...")
    train_ds = Cine4CHSliceDataset(
        train_cases,
        image_size=DEFAULT_IMAGE_SIZE,
        augment=True,
        emd_config=spec.emd_config,
        precompute_desc=f"{spec.run_id} train",
        use_disk_cache=True,
        num_preprocess_workers=workers,
    )
    print("Precomputing validation preprocessing...")
    val_ds = Cine4CHSliceDataset(
        val_cases,
        image_size=DEFAULT_IMAGE_SIZE,
        augment=False,
        emd_config=spec.emd_config,
        precompute_desc=f"{spec.run_id} val",
        use_disk_cache=True,
        num_preprocess_workers=workers,
    )

    # Defensive channel check before training (especially for concat).
    sample_img, sample_mask, _, _ = train_ds[0]
    expected_channels = output_channels(spec.emd_config)
    print(f"  sample image shape: {tuple(sample_img.shape)}")
    print(f"  sample mask shape:  {tuple(sample_mask.shape)}")
    print(f"  model in_channels:  {expected_channels}")
    if sample_img.ndim != 3 or sample_img.shape[0] != expected_channels:
        raise RuntimeError(
            f"Concat/channel check failed for run '{spec.run_id}': "
            f"sample image shape {tuple(sample_img.shape)}, expected "
            f"({expected_channels}, H, W)."
        )
    if spec.emd_config.mode == "concat" and expected_channels != 2:
        raise RuntimeError(
            f"concat mode must produce 2 channels, got output_channels={expected_channels}"
        )
    if sample_mask.ndim != 2:
        raise RuntimeError(f"Mask must be 2D integer labels, got shape {tuple(sample_mask.shape)}")

    generator = torch.Generator()
    generator.manual_seed(hyperparams.seed)
    use_cuda = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=hyperparams.batch_size,
        shuffle=True,
        num_workers=hyperparams.num_workers,
        pin_memory=use_cuda,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=hyperparams.batch_size,
        shuffle=False,
        num_workers=hyperparams.num_workers,
        pin_memory=use_cuda,
    )

    model = UNet2D(in_channels=in_channels, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.lr)

    batch_images, batch_masks, _, _ = next(iter(train_loader))
    print(f"  train batch image shape: {tuple(batch_images.shape)}")
    print(f"  train batch mask shape:  {tuple(batch_masks.shape)}")
    print(f"  model in_channels:       {in_channels}")
    if batch_images.shape[1] != in_channels:
        raise RuntimeError(
            f"Batch channel mismatch: images have {batch_images.shape[1]} channels, "
            f"model expects {in_channels}."
        )

    log_path = run_dir / "training_log.csv"
    log_fields = (
        ["epoch", "train_loss", "val_dice_mean", "val_dice_mean_foreground"]
        + [f"val_dice_class_{i}" for i in range(NUM_CLASSES)]
    )

    start_epoch = 1
    best_fg = -1.0
    best_epoch = 0
    best_metrics: Dict[str, float] = {}
    existing_rows: List[Dict[str, str]] = []

    if resume and log_path.exists():
        with open(log_path, newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))
        if existing_rows:
            start_epoch = int(existing_rows[-1]["epoch"]) + 1
            for row in existing_rows:
                fg = float(row["val_dice_mean_foreground"])
                if fg > best_fg:
                    best_fg = fg
                    best_epoch = int(row["epoch"])
                    best_metrics = {
                        "dice_mean": float(row["val_dice_mean"]),
                        "dice_mean_foreground": fg,
                        **{
                            f"dice_class_{i}": float(row[f"val_dice_class_{i}"])
                            for i in range(NUM_CLASSES)
                        },
                    }
            latest_ckpt = ckpt_dir / f"epoch_{start_epoch - 1:03d}.pt"
            if not latest_ckpt.exists():
                latest_ckpt = ckpt_dir / "best.pt"
            if latest_ckpt.exists() and start_epoch <= hyperparams.epochs:
                checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"  Resuming from epoch {start_epoch} using {latest_ckpt.name}")
            elif start_epoch > hyperparams.epochs:
                print(f"  Resume: already completed {hyperparams.epochs} epochs.")

    if start_epoch > hyperparams.epochs:
        plot_validation_curves(log_path, run_dir / "val_curves.png")
        save_metrics_csv(run_dir / "val_metrics_best.csv", "val", best_metrics)
        return {
            "run_id": spec.run_id,
            "best_epoch": best_epoch,
            "val_dice_mean": best_metrics.get("dice_mean", 0.0),
            "val_dice_mean_foreground": best_metrics.get("dice_mean_foreground", 0.0),
            **{f"val_dice_class_{i}": best_metrics.get(f"dice_class_{i}", 0.0) for i in range(NUM_CLASSES)},
            "checkpoint": str(ckpt_dir / "best.pt"),
        }

    mode = "a" if (resume and existing_rows) else "w"
    with open(log_path, mode, newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=log_fields)
        if mode == "w":
            writer.writeheader()

        for epoch in range(start_epoch, hyperparams.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate_loader(model, val_loader, device)

            row = {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_dice_mean": f"{val_metrics['dice_mean']:.6f}",
                "val_dice_mean_foreground": f"{val_metrics['dice_mean_foreground']:.6f}",
            }
            for i in range(NUM_CLASSES):
                row[f"val_dice_class_{i}"] = f"{val_metrics[f'dice_class_{i}']:.6f}"
            writer.writerow(row)
            log_file.flush()

            print(
                f"  epoch {epoch:02d}/{hyperparams.epochs}  loss={train_loss:.4f}  "
                f"val_fg={val_metrics['dice_mean_foreground']:.4f}"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_dice_mean_foreground": val_metrics["dice_mean_foreground"],
                    "run_id": spec.run_id,
                    "emd_config": config_to_dict(spec.emd_config),
                },
                ckpt_dir / f"epoch_{epoch:03d}.pt",
            )

            if val_metrics["dice_mean_foreground"] > best_fg:
                best_fg = val_metrics["dice_mean_foreground"]
                best_epoch = epoch
                best_metrics = val_metrics.copy()
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_dice_mean_foreground": val_metrics["dice_mean_foreground"],
                        "run_id": spec.run_id,
                    },
                    ckpt_dir / "best.pt",
                )

    plot_validation_curves(log_path, run_dir / "val_curves.png")
    save_metrics_csv(run_dir / "val_metrics_best.csv", "val", best_metrics)

    return {
        "run_id": spec.run_id,
        "best_epoch": best_epoch,
        "val_dice_mean": best_metrics.get("dice_mean", 0.0),
        "val_dice_mean_foreground": best_metrics.get("dice_mean_foreground", 0.0),
        **{f"val_dice_class_{i}": best_metrics.get(f"dice_class_{i}", 0.0) for i in range(NUM_CLASSES)},
        "checkpoint": str(ckpt_dir / "best.pt"),
    }


def save_metrics_csv(path: Path, prefix: str, metrics: Dict[str, float]) -> None:
    """Write mean / foreground / per-class Dice to a two-column CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow([f"{prefix}_dice_mean", f"{metrics.get('dice_mean', 0):.6f}"])
        writer.writerow(
            [f"{prefix}_dice_mean_foreground", f"{metrics.get('dice_mean_foreground', 0):.6f}"]
        )
        for i in range(NUM_CLASSES):
            writer.writerow([f"{prefix}_dice_class_{i}", f"{metrics.get(f'dice_class_{i}', 0):.6f}"])
            writer.writerow(
                [f"{prefix}_dice_{LABEL_NAMES[i]}", f"{metrics.get(f'dice_{LABEL_NAMES[i]}', 0):.6f}"]
            )


def plot_validation_curves(log_csv: Path, save_path: Path) -> None:
    epochs: List[int] = []
    dice_all: List[float] = []
    dice_fg: List[float] = []
    with open(log_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            dice_all.append(float(row["val_dice_mean"]))
            dice_fg.append(float(row["val_dice_mean_foreground"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, dice_all, marker="o", label="Val Dice (all classes)")
    ax.plot(epochs, dice_fg, marker="s", label="Val Dice (foreground)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.set_title("Validation curves")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate_test_and_visualize(
    spec: AblationRunSpec,
    hyperparams: AblationHyperparams,
    splits_csv: Path,
    run_dir: Path,
    device: torch.device,
    num_overlays: int = 10,
    seed: int = 42,
    preprocess_workers: int | None = None,
) -> Dict[str, float]:
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    in_channels = output_channels(spec.emd_config)
    workers = _default_preprocess_workers() if preprocess_workers is None else preprocess_workers

    print(
        f"Evaluating {spec.run_id}: checkpoint epoch={checkpoint.get('epoch')} "
        f"val_fg={checkpoint.get('val_dice_mean_foreground')} in_channels={in_channels}"
    )

    test_cases = load_split_cases(splits_csv, "test")
    test_ds = Cine4CHSliceDataset(
        test_cases,
        image_size=DEFAULT_IMAGE_SIZE,
        augment=False,
        emd_config=spec.emd_config,
        precompute_desc=f"{spec.run_id} test",
        use_disk_cache=True,
        num_preprocess_workers=workers,
    )
    # Channel check on test sample
    t0, m0, _, _ = test_ds[0]
    print(f"  test sample image shape: {tuple(t0.shape)}")
    print(f"  test sample mask shape:  {tuple(m0.shape)}")
    if t0.shape[0] != in_channels:
        raise RuntimeError(
            f"Test sample has {t0.shape[0]} channels, model expects {in_channels}"
        )

    test_loader = DataLoader(
        test_ds,
        batch_size=hyperparams.batch_size,
        shuffle=False,
        num_workers=hyperparams.num_workers,
    )

    model = UNet2D(in_channels=in_channels, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_loader(model, test_loader, device)

    metrics_path = run_dir / "test_metrics.csv"
    save_metrics_csv(metrics_path, "test", test_metrics)

    pred_dir = run_dir / "test_predictions"
    overlay_dir = run_dir / "qualitative_overlays"
    pred_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    chosen_cases = rng.sample(test_cases, k=min(num_overlays, len(test_cases)))

    for case in sorted(chosen_cases, key=lambda c: c.stem):
        image, label, _ = load_pair(case)
        frame_idx = choose_representative_frame(label)

        ds = Cine4CHSliceDataset(
            [case],
            image_size=DEFAULT_IMAGE_SIZE,
            augment=False,
            emd_config=spec.emd_config,
            precompute_desc=None,
            use_disk_cache=True,
            num_preprocess_workers=0,
        )
        frame_indices = [idx for idx, (_, fidx) in enumerate(ds.index) if fidx == frame_idx]
        if not frame_indices:
            frame_indices = [0]
        image_t, label_t, _, fidx = ds[frame_indices[0]]

        logits = model(image_t.unsqueeze(0).to(device))
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)
        gt = label_t.numpy().astype(np.int64)

        if image_t.shape[0] == 1:
            display = image_t.squeeze(0).numpy()
        else:
            display = image_t[0].numpy()

        base = safe_minmax_normalize(display, clip=True)
        gt_overlay = overlay_label(base, gt)
        pred_overlay = overlay_label(base, pred)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"{spec.run_id} — {case.stem} frame {fidx}")
        axes[0].imshow(base, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Input (ch0)")
        axes[0].axis("off")
        axes[1].imshow(gt_overlay)
        axes[1].set_title("GT overlay")
        axes[1].axis("off")
        axes[2].imshow(pred_overlay)
        axes[2].set_title("Prediction overlay")
        axes[2].axis("off")

        out_path = overlay_dir / f"{case.stem}_frame{fidx:03d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        np.savez_compressed(
            pred_dir / f"{case.stem}_frame{fidx:03d}.npz",
            prediction=pred,
            ground_truth=gt,
            frame_index=fidx,
        )

    return test_metrics


def load_val_summary_from_run(run_dir: Path) -> Dict[str, Any]:
    """Load best validation metrics from an existing run directory."""
    log_path = run_dir / "training_log.csv"
    if log_path.exists():
        best_row: Dict[str, str] | None = None
        best_fg = -1.0
        with open(log_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("epoch"):
                    continue
                fg = float(row["val_dice_mean_foreground"])
                if fg > best_fg:
                    best_fg = fg
                    best_row = row
        if best_row is not None:
            summary: Dict[str, Any] = {
                "run_id": run_dir.name,
                "best_epoch": int(best_row["epoch"]),
                "val_dice_mean": float(best_row["val_dice_mean"]),
                "val_dice_mean_foreground": float(best_row["val_dice_mean_foreground"]),
            }
            for i in range(NUM_CLASSES):
                summary[f"val_dice_class_{i}"] = float(best_row[f"val_dice_class_{i}"])
            # Keep val_metrics_best.csv / curves in sync with the log.
            save_metrics_csv(
                run_dir / "val_metrics_best.csv",
                "val",
                {
                    "dice_mean": summary["val_dice_mean"],
                    "dice_mean_foreground": summary["val_dice_mean_foreground"],
                    **{f"dice_class_{i}": summary[f"val_dice_class_{i}"] for i in range(NUM_CLASSES)},
                    **{
                        f"dice_{LABEL_NAMES[i]}": summary[f"val_dice_class_{i}"]
                        for i in range(NUM_CLASSES)
                    },
                },
            )
            plot_validation_curves(log_path, run_dir / "val_curves.png")
            return summary

    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return {
            "run_id": checkpoint.get("run_id", run_dir.name),
            "best_epoch": checkpoint.get("epoch", ""),
            "val_dice_mean_foreground": checkpoint.get("val_dice_mean_foreground", 0.0),
        }

    return {"run_id": run_dir.name}


def write_ablation_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
