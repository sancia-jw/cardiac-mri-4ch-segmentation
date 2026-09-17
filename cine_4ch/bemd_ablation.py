"""Training / evaluation for the BEMD (square-pad) ablation experiment.

Reuses the same hyperparams, U-Net, loss, metrics, and checkpoint selection as
``cine_4ch.ablation``, but consumes ``BEMDSliceDataset`` (no BEMD at train time).
"""

from __future__ import annotations

import csv
import json
import platform
import random
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from cine_4ch.ablation import (
    AblationHyperparams,
    evaluate_loader,
    load_val_summary_from_run,
    plot_validation_curves,
    resolve_device,
    save_metrics_csv,
    set_seed,
    train_one_epoch,
)
from cine_4ch.bemd_cache import BEMD_CACHE_ROOT, METHOD_ID
from cine_4ch.bemd_dataset import (
    ENHANCED_CACHE_ROOT,
    BEMDEnhanceSpec,
    BEMDSliceDataset,
    default_bemd_ablation_specs,
    required_bimf_count,
    spec_to_dict,
)
from cine_4ch.config import DATA_ROOT, DEFAULT_IMAGE_SIZE, LABEL_NAMES, NUM_CLASSES, OUTPUTS_DIR, PROJECT_ROOT
from cine_4ch.dataset import load_split_cases
from cine_4ch.io import choose_representative_frame, load_pair
from cine_4ch.model import UNet2D
from cine_4ch.viz import overlay_label
from src.preprocessing.bemd_square_pad import BEMDConfig, pyemd_version

# Re-export for scripts
__all__ = [
    "AblationHyperparams",
    "BEMDEnhanceSpec",
    "default_bemd_ablation_specs",
    "collect_provenance",
    "train_bemd_run",
    "evaluate_bemd_test",
    "write_bemd_summary",
    "resolve_device",
    "load_val_summary_from_run",
    "fixed_subset",
]


BEMD_ABLATION_ROOT = OUTPUTS_DIR / "bemd_ablation"


def fixed_subset(cases: Sequence, n: int, seed: int) -> list:
    """Deterministic order-preserving subset (same helper as IMF screen)."""
    cases = list(cases)
    if n >= len(cases):
        return cases
    rng = np.random.default_rng(seed)
    idxs = sorted(rng.choice(len(cases), size=n, replace=False).tolist())
    return [cases[i] for i in idxs]


def _git_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"git_commit": None, "git_dirty": None, "git_available": False}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["git_available"] = True
        info["git_commit"] = commit
        info["git_dirty"] = bool(dirty)
    except Exception:
        pass
    return info


def collect_provenance(
    *,
    spec: BEMDEnhanceSpec,
    hyperparams: AblationHyperparams,
    device: torch.device,
    bemd_cache_root: Path,
    splits_csv: Path,
    bemd_cfg: Optional[BEMDConfig] = None,
    data_root: Path = DATA_ROOT,
    enhanced_cache_root: Path = ENHANCED_CACHE_ROOT,
    excluded_frames: frozenset[tuple[str, int]] = frozenset(),
) -> Dict[str, Any]:
    cfg = bemd_cfg or BEMDConfig()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "pyemd": pyemd_version(),
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": hyperparams.seed,
        "splits_csv": str(splits_csv),
        "split_counts_expected": {"train": 74, "val": 16, "test": 15},
        "decomposition_method": METHOD_ID,
        "bemd_cache_root": str(bemd_cache_root),
        "data_root": str(data_root),
        "enhanced_cache_root": str(enhanced_cache_root),
        "excluded_frames": [{"case_stem": s, "frame_idx": f} for s, f in sorted(excluded_frames)],
        "bemd_settings": asdict(cfg),
        "padding_method": "zero_bottom_right_to_max_hw",
        "reconstruction_convention": (
            "gastro_style: per-BIMF min-max (normalize_bimfs=True) then "
            "original - sum(selected); finalize with per-slice min-max to [0,1]"
        ),
        "hyperparams": asdict(hyperparams),
        "model": "UNet2D",
        "image_size": list(DEFAULT_IMAGE_SIZE),
        "num_classes": NUM_CLASSES,
        "label_names": LABEL_NAMES,
        "enhance_spec": spec_to_dict(spec),
        **_git_info(),
    }


def save_run_artifacts(
    run_dir: Path,
    spec: BEMDEnhanceSpec,
    hyperparams: AblationHyperparams,
    provenance: Dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": spec.run_id,
        "description": spec.description,
        "enhance_spec": spec_to_dict(spec),
        "hyperparams": asdict(hyperparams),
        "in_channels": 1,
        "num_classes": NUM_CLASSES,
        "experiment": "bemd_ablation",
        "decomposition_method": METHOD_ID,
    }
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def train_bemd_run(
    spec: BEMDEnhanceSpec,
    hyperparams: AblationHyperparams,
    splits_csv: Path,
    run_dir: Path,
    device: torch.device,
    *,
    resume: bool = False,
    bemd_cache_root: Path = BEMD_CACHE_ROOT,
    train_cases: Optional[list] = None,
    val_cases: Optional[list] = None,
    data_root: Path = DATA_ROOT,
    enhanced_cache_root: Path = ENHANCED_CACHE_ROOT,
    excluded_frames: frozenset[tuple[str, int]] = frozenset(),
) -> Dict[str, Any]:
    set_seed(hyperparams.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    need = required_bimf_count([spec])
    provenance = collect_provenance(
        spec=spec,
        hyperparams=hyperparams,
        device=device,
        bemd_cache_root=bemd_cache_root,
        splits_csv=splits_csv,
        data_root=data_root,
        enhanced_cache_root=enhanced_cache_root,
        excluded_frames=excluded_frames,
    )
    save_run_artifacts(run_dir, spec, hyperparams, provenance)

    if train_cases is None:
        train_cases = load_split_cases(splits_csv, "train", data_root=data_root)
    if val_cases is None:
        val_cases = load_split_cases(splits_csv, "val", data_root=data_root)

    print(f"\n=== BEMD run: {spec.run_id} ===")
    print(f"Enhance: mode={spec.mode} bimf_indices={list(spec.bimf_indices)}")
    print(f"BEMD cache: {bemd_cache_root}")

    train_ds = BEMDSliceDataset(
        train_cases,
        spec,
        image_size=DEFAULT_IMAGE_SIZE,
        augment=True,
        bemd_cache_root=bemd_cache_root,
        require_n_bimf=need or None,
        precompute_desc=f"{spec.run_id} train",
        enhanced_cache_root=enhanced_cache_root,
        excluded_frames=excluded_frames,
    )
    val_ds = BEMDSliceDataset(
        val_cases,
        spec,
        image_size=DEFAULT_IMAGE_SIZE,
        augment=False,
        bemd_cache_root=bemd_cache_root,
        require_n_bimf=need or None,
        precompute_desc=f"{spec.run_id} val",
        enhanced_cache_root=enhanced_cache_root,
        excluded_frames=excluded_frames,
    )

    sample_img, sample_mask, _, _ = train_ds[0]
    print(f"  sample image shape: {tuple(sample_img.shape)}")
    print(f"  sample mask shape:  {tuple(sample_mask.shape)}")
    if sample_img.ndim != 3 or sample_img.shape[0] != 1:
        raise RuntimeError(f"Expected 1-channel input, got {tuple(sample_img.shape)}")

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

    model = UNet2D(in_channels=1, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.lr)

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
                    "enhance_spec": spec_to_dict(spec),
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
                        "enhance_spec": spec_to_dict(spec),
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


@torch.no_grad()
def evaluate_bemd_test(
    spec: BEMDEnhanceSpec,
    hyperparams: AblationHyperparams,
    splits_csv: Path,
    run_dir: Path,
    device: torch.device,
    *,
    num_overlays: int = 10,
    seed: int = 42,
    bemd_cache_root: Path = BEMD_CACHE_ROOT,
    test_cases: Optional[list] = None,
    data_root: Path = DATA_ROOT,
    enhanced_cache_root: Path = ENHANCED_CACHE_ROOT,
    excluded_frames: frozenset[tuple[str, int]] = frozenset(),
) -> Dict[str, float]:
    from src.preprocessing.emd_enhancement import safe_minmax_normalize

    ckpt_path = run_dir / "checkpoints" / "best.pt"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    need = required_bimf_count([spec])

    if test_cases is None:
        test_cases = load_split_cases(splits_csv, "test", data_root=data_root)

    test_ds = BEMDSliceDataset(
        test_cases,
        spec,
        image_size=DEFAULT_IMAGE_SIZE,
        augment=False,
        bemd_cache_root=bemd_cache_root,
        require_n_bimf=need or None,
        precompute_desc=f"{spec.run_id} test",
        enhanced_cache_root=enhanced_cache_root,
        excluded_frames=excluded_frames,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=hyperparams.batch_size,
        shuffle=False,
        num_workers=hyperparams.num_workers,
    )
    model = UNet2D(in_channels=1, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_loader(model, test_loader, device)
    save_metrics_csv(run_dir / "test_metrics.csv", "test", test_metrics)

    pred_dir = run_dir / "test_predictions"
    overlay_dir = run_dir / "qualitative_overlays"
    pred_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    chosen = rng.sample(test_cases, k=min(num_overlays, len(test_cases)))
    for case in sorted(chosen, key=lambda c: c.stem):
        _, label, _ = load_pair(case)
        frame_idx = choose_representative_frame(label)
        ds = BEMDSliceDataset(
            [case],
            spec,
            image_size=DEFAULT_IMAGE_SIZE,
            augment=False,
            bemd_cache_root=bemd_cache_root,
            require_n_bimf=need or None,
            enhanced_cache_root=enhanced_cache_root,
            excluded_frames=excluded_frames,
        )
        if not len(ds):
            continue
        frame_indices = [i for i, (_, fidx) in enumerate(ds.index) if fidx == frame_idx] or [0]
        image_t, label_t, _, fidx = ds[frame_indices[0]]
        logits = model(image_t.unsqueeze(0).to(device))
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)
        gt = label_t.numpy().astype(np.int64)
        display = image_t.squeeze(0).numpy()
        base = safe_minmax_normalize(display, clip=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"{spec.run_id} — {case.stem} frame {fidx}")
        axes[0].imshow(base, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Input")
        axes[0].axis("off")
        axes[1].imshow(overlay_label(base, gt))
        axes[1].set_title("GT")
        axes[1].axis("off")
        axes[2].imshow(overlay_label(base, pred))
        axes[2].set_title("Pred")
        axes[2].axis("off")
        fig.savefig(overlay_dir / f"{case.stem}_frame{fidx:03d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        np.savez_compressed(
            pred_dir / f"{case.stem}_frame{fidx:03d}.npz",
            prediction=pred,
            ground_truth=gt,
            frame_index=fidx,
        )
    return test_metrics


def write_bemd_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
