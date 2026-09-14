#!/usr/bin/env python3
"""
Fast staged screening for lower-frequency IMF subtraction settings.

Stages
------
1. Inspect how many IMFs exist on a small train/val sample.
2. Visual side-by-side comparison on a tiny case set.
3. 3-epoch training screen on a fixed small subset (no test set).
4. Rank candidates and recommend at most one full run (does NOT launch it).

Outputs land under outputs/emd_lower_imf_screen/ and reuse outputs/emd_cache/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cine_4ch.ablation import (
    AblationHyperparams,
    evaluate_loader,
    resolve_device,
    set_seed,
    train_one_epoch,
)
from cine_4ch.config import DEFAULT_IMAGE_SIZE, LABEL_NAMES, NUM_CLASSES, OUTPUTS_DIR
from cine_4ch.dataset import Cine4CHSliceDataset, load_split_cases
from cine_4ch.io import CasePair, choose_representative_frame, extract_frame, load_pair
from cine_4ch.metrics import multiclass_dice
from cine_4ch.model import UNet2D
from cine_4ch.viz import overlay_label
from src.preprocessing.emd_enhancement import (
    EMEnhancementConfig,
    compute_imfs,
    enhance_mri_slice,
    output_channels,
    safe_minmax_normalize,
)


SCREEN_ROOT = OUTPUTS_DIR / "emd_lower_imf_screen"
VISUALS_DIR = SCREEN_ROOT / "visuals"
METRICS_DIR = SCREEN_ROOT / "metrics"
CKPT_DIR = SCREEN_ROOT / "checkpoints"


@dataclass
class ScreenCandidate:
    run_id: str
    imf_indices: List[int]
    description: str
    optional: bool = False  # skipped unless Stage 1 confirms availability


def candidate_catalog_mid() -> List[ScreenCandidate]:
    """Mid/lower positive IMF indices (previous screen)."""
    return [
        ScreenCandidate("original", [], "Subset baseline: normalized original (no EMD)."),
        ScreenCandidate("subtract_imf_2", [2], "Original minus IMF [2]."),
        ScreenCandidate("subtract_imf_3", [3], "Original minus IMF [3]."),
        ScreenCandidate("subtract_imf_2_3", [2, 3], "Original minus IMFs [2, 3]."),
        ScreenCandidate(
            "subtract_imf_3_4",
            [3, 4],
            "Original minus IMFs [3, 4] (only if IMF 4 exists consistently).",
            optional=True,
        ),
        ScreenCandidate(
            "subtract_residual",
            [-1],
            "Original minus final/lowest-frequency IMF (residual trend).",
            optional=True,
        ),
    ]


def candidate_catalog_lowfreq() -> List[ScreenCandidate]:
    """
    True low-frequency / trend removal (negative IMF indices).

    IMF[-1] is the slowest residual; [-1,-2,-3] matches Gastro-style detrending.
    """
    return [
        ScreenCandidate("original", [], "Subset baseline: normalized original (no EMD)."),
        ScreenCandidate(
            "subtract_imf_neg1",
            [-1],
            "Original minus IMF [-1] (lowest-frequency residual only).",
        ),
        ScreenCandidate(
            "subtract_imf_neg1_2",
            [-1, -2],
            "Original minus IMFs [-1, -2] (two slowest components).",
        ),
        ScreenCandidate(
            "subtract_imf_neg2_3",
            [-2, -3],
            "Original minus IMFs [-2, -3] (slow components, keep residual).",
        ),
        ScreenCandidate(
            "subtract_imf_neg1_2_3",
            [-1, -2, -3],
            "Original minus IMFs [-1, -2, -3] (Gastro-style low-frequency trend removal).",
        ),
    ]


def candidate_catalog(name: str = "mid") -> List[ScreenCandidate]:
    if name == "mid":
        return candidate_catalog_mid()
    if name == "lowfreq":
        return candidate_catalog_lowfreq()
    raise ValueError(f"Unknown catalog '{name}'. Use mid or lowfreq.")


def emd_config_for(candidate: ScreenCandidate) -> EMEnhancementConfig:
    if candidate.run_id == "original" or not candidate.imf_indices:
        return EMEnhancementConfig(mode="original", imf_indices=[])
    return EMEnhancementConfig(mode="subtract", imf_indices=list(candidate.imf_indices))


def fixed_subset(
    cases: Sequence[CasePair],
    n: int,
    seed: int,
) -> List[CasePair]:
    """Deterministic subset from the existing split list (order-preserving sample)."""
    if n >= len(cases):
        return list(cases)
    rng = np.random.default_rng(seed)
    idxs = sorted(rng.choice(len(cases), size=n, replace=False).tolist())
    return [cases[i] for i in idxs]


def elapsed_str(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}m"


def is_nearly_constant(arr: np.ndarray, eps: float = 1e-6) -> bool:
    return float(np.nanmax(arr) - np.nanmin(arr)) <= eps


# ---------------------------------------------------------------------------
# Stage 1: IMF availability
# ---------------------------------------------------------------------------


def stage1_inspect_imfs(
    train_cases: Sequence[CasePair],
    val_cases: Sequence[CasePair],
    *,
    n_train_cases: int = 6,
    n_val_cases: int = 3,
    frames_per_case: int = 2,
    seed: int = 42,
) -> Dict[str, Any]:
    print("\n=== Stage 1: IMF availability ===")
    t0 = time.perf_counter()
    sample_train = fixed_subset(train_cases, n_train_cases, seed)
    sample_val = fixed_subset(val_cases, n_val_cases, seed + 1)

    counts: List[int] = []
    index_hits: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    residual_ok = 0
    samples_checked = 0
    details: List[Dict[str, Any]] = []

    for case in list(sample_train) + list(sample_val):
        image, label, _ = load_pair(case)
        rep = choose_representative_frame(label)
        n_frames = 1 if label.ndim < 3 else label.shape[-1]
        frame_idxs = sorted({rep, max(0, min(n_frames - 1, rep // 2))})[:frames_per_case]
        for fidx in frame_idxs:
            slice_2d = extract_frame(image, fidx)
            imfs = compute_imfs(slice_2d, normalize_imfs=True)
            n = len(imfs)
            counts.append(n)
            samples_checked += 1
            for idx in index_hits:
                if n > idx:
                    index_hits[idx] += 1
            if n >= 1:
                residual_ok += 1
            details.append({"case": case.stem, "frame": fidx, "n_imfs": n})
            print(f"  {case.stem} frame {fidx:03d}: {n} IMFs")

    counts_arr = np.asarray(counts, dtype=np.int32)
    report = {
        "samples_checked": samples_checked,
        "n_imfs_min": int(counts_arr.min()) if len(counts_arr) else 0,
        "n_imfs_median": float(np.median(counts_arr)) if len(counts_arr) else 0,
        "n_imfs_max": int(counts_arr.max()) if len(counts_arr) else 0,
        "index_availability": {
            str(k): {"present": v, "fraction": v / max(samples_checked, 1)}
            for k, v in index_hits.items()
        },
        "residual_neg1_fraction": residual_ok / max(samples_checked, 1),
        "elapsed_sec": time.perf_counter() - t0,
        "details": details,
    }

    # Require >= 90% presence for an index to be considered consistent.
    threshold = 0.9
    valid_positive = [k for k, v in index_hits.items() if v / max(samples_checked, 1) >= threshold]
    residual_valid = report["residual_neg1_fraction"] >= threshold

    print(
        f"  IMF count min/median/max = "
        f"{report['n_imfs_min']}/{report['n_imfs_median']}/{report['n_imfs_max']}"
    )
    for k, info in report["index_availability"].items():
        print(f"  IMF[{k}] present in {info['present']}/{samples_checked} ({info['fraction']:.0%})")
    print(f"  residual IMF[-1] present in {residual_ok}/{samples_checked}")
    print(f"  Stage 1 elapsed: {elapsed_str(report['elapsed_sec'])}")

    report["consistent_indices"] = valid_positive
    report["residual_valid"] = residual_valid
    report["availability_threshold"] = threshold
    return report


def select_valid_candidates(
    stage1: Dict[str, Any],
    catalog: Sequence[ScreenCandidate],
) -> List[ScreenCandidate]:
    consistent = set(stage1["consistent_indices"])
    residual_valid = bool(stage1["residual_valid"])
    n_min = int(stage1.get("n_imfs_min", 0))
    selected: List[ScreenCandidate] = []
    for cand in catalog:
        if cand.run_id == "original":
            selected.append(cand)
            continue

        # Negative indices need enough IMFs (e.g. -3 requires >= 3 components).
        neg = [i for i in cand.imf_indices if i < 0]
        if neg:
            need = max(abs(i) for i in neg)
            if n_min < need or not residual_valid:
                print(
                    f"  SKIP {cand.run_id}: need >= {need} IMFs "
                    f"(have min {n_min}, residual_ok={residual_valid})"
                )
                continue
            selected.append(cand)
            continue

        if cand.run_id == "subtract_residual":
            if residual_valid:
                selected.append(cand)
            else:
                print(f"  SKIP {cand.run_id}: residual IMF[-1] not consistently available")
            continue

        needed = [i for i in cand.imf_indices if i >= 0]
        if all(i in consistent for i in needed):
            selected.append(cand)
        else:
            missing = [i for i in needed if i not in consistent]
            print(f"  SKIP {cand.run_id}: missing consistent indices {missing}")
    return selected


# ---------------------------------------------------------------------------
# Stage 2: visual screening
# ---------------------------------------------------------------------------


def stage2_visuals(
    train_cases: Sequence[CasePair],
    val_cases: Sequence[CasePair],
    candidates: Sequence[ScreenCandidate],
    *,
    seed: int = 42,
) -> Dict[str, Any]:
    print("\n=== Stage 2: visual screening ===")
    t0 = time.perf_counter()
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    vis_train = fixed_subset(train_cases, 3, seed)
    vis_val = fixed_subset(val_cases, 2, seed + 7)
    # Exclude original from figure columns (shown as reference).
    emd_cands = [c for c in candidates if c.run_id != "original"]
    notes: List[Dict[str, Any]] = []
    saved = 0

    for case in list(vis_train) + list(vis_val):
        image, label, _ = load_pair(case)
        fidx = choose_representative_frame(label)
        slice_2d = extract_frame(image, fidx)
        gt = extract_frame(label, fidx).astype(np.int64)
        original = safe_minmax_normalize(slice_2d, clip=True)

        panels: List[Tuple[str, np.ndarray]] = [("original", original)]
        anatomy_flags: Dict[str, str] = {}
        for cand in emd_cands:
            cfg = emd_config_for(cand)
            try:
                enhanced = enhance_mri_slice(slice_2d, cfg)
            except Exception as exc:  # pragma: no cover
                anatomy_flags[cand.run_id] = f"FAILED: {exc}"
                continue
            if enhanced.ndim == 3:
                enhanced = enhanced[..., 0]
            enhanced = safe_minmax_normalize(enhanced, clip=True)
            if is_nearly_constant(enhanced):
                anatomy_flags[cand.run_id] = "NEARLY_CONSTANT"
            else:
                # Heuristic: if chamber-ish bright regions flatten toward mean, flag distortion.
                fg = gt > 0
                if fg.any():
                    orig_contrast = float(original[fg].std())
                    enh_contrast = float(enhanced[fg].std())
                    if enh_contrast < 0.35 * max(orig_contrast, 1e-6):
                        anatomy_flags[cand.run_id] = "LOW_FG_CONTRAST"
                    else:
                        anatomy_flags[cand.run_id] = "OK"
                else:
                    anatomy_flags[cand.run_id] = "OK_NO_FG"
            panels.append((cand.run_id, enhanced))

        n = len(panels)
        fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.5))
        if n == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        for col, (title, img) in enumerate(panels):
            axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[0, col].set_title(title, fontsize=9)
            axes[0, col].axis("off")
            axes[1, col].imshow(overlay_label(img, gt))
            axes[1, col].set_title("overlay", fontsize=9)
            axes[1, col].axis("off")
        fig.suptitle(f"{case.stem} frame {fidx}")
        out = VISUALS_DIR / f"{case.stem}_frame{fidx:03d}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        saved += 1
        notes.append({"case": case.stem, "frame": fidx, "flags": anatomy_flags, "figure": str(out)})
        print(f"  saved {out.name}  flags={anatomy_flags}")

    report = {
        "figures_saved": saved,
        "elapsed_sec": time.perf_counter() - t0,
        "notes": notes,
    }
    print(f"  Stage 2 elapsed: {elapsed_str(report['elapsed_sec'])}")
    return report


def visuals_look_reasonable(stage2: Dict[str, Any], run_id: str) -> bool:
    bad = {"NEARLY_CONSTANT", "LOW_FG_CONTRAST"}
    flags = []
    for note in stage2.get("notes", []):
        flag = note.get("flags", {}).get(run_id)
        if flag:
            flags.append(flag)
    if not flags:
        return True
    # Require majority OK-like.
    ok = sum(1 for f in flags if f.startswith("OK"))
    return ok >= math.ceil(0.6 * len(flags)) and not all(f in bad for f in flags)


# ---------------------------------------------------------------------------
# Stage 3: short training screen
# ---------------------------------------------------------------------------


def _invalid_batch(images: torch.Tensor) -> bool:
    if not torch.isfinite(images).all():
        return True
    # Channel-wise nearly constant across batch.
    flat = images.reshape(images.shape[0], images.shape[1], -1)
    spreads = (flat.amax(dim=-1) - flat.amin(dim=-1)).mean().item()
    return spreads < 1e-5


def screen_train_one(
    candidate: ScreenCandidate,
    train_cases: Sequence[CasePair],
    val_cases: Sequence[CasePair],
    hyperparams: AblationHyperparams,
    device: torch.device,
    *,
    preprocess_workers: int,
    original_baseline_fg: Optional[float],
) -> Dict[str, Any]:
    print(f"\n--- Screen train: {candidate.run_id} ---")
    t0 = time.perf_counter()
    set_seed(hyperparams.seed)
    cfg = emd_config_for(candidate)
    in_ch = output_channels(cfg)

    t_pre = time.perf_counter()
    train_ds = Cine4CHSliceDataset(
        train_cases,
        image_size=DEFAULT_IMAGE_SIZE,
        augment=True,
        emd_config=cfg,
        precompute_desc=f"screen {candidate.run_id} train",
        use_disk_cache=True,
        num_preprocess_workers=preprocess_workers,
    )
    val_ds = Cine4CHSliceDataset(
        val_cases,
        image_size=DEFAULT_IMAGE_SIZE,
        augment=False,
        emd_config=cfg,
        precompute_desc=f"screen {candidate.run_id} val",
        use_disk_cache=True,
        num_preprocess_workers=preprocess_workers,
    )
    preprocess_sec = time.perf_counter() - t_pre

    sample_img, sample_mask, _, _ = train_ds[0]
    print(f"  sample image {tuple(sample_img.shape)}  mask {tuple(sample_mask.shape)}  in_ch={in_ch}")
    if is_nearly_constant(sample_img.numpy()):
        return {
            "run_id": candidate.run_id,
            "status": "INVALID_CONSTANT_INPUT",
            "preprocess_sec": preprocess_sec,
            "train_sec": 0.0,
            "elapsed_sec": time.perf_counter() - t0,
            "best_epoch": "",
            "val_dice_mean_foreground": "",
            **{f"val_dice_class_{i}": "" for i in range(NUM_CLASSES)},
        }

    gen = torch.Generator().manual_seed(hyperparams.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=hyperparams.batch_size,
        shuffle=True,
        num_workers=0,
        generator=gen,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=hyperparams.batch_size,
        shuffle=False,
        num_workers=0,
    )

    batch_images, batch_masks, _, _ = next(iter(train_loader))
    print(f"  batch image {tuple(batch_images.shape)}  mask {tuple(batch_masks.shape)}")
    if _invalid_batch(batch_images):
        return {
            "run_id": candidate.run_id,
            "status": "INVALID_BATCH",
            "preprocess_sec": preprocess_sec,
            "train_sec": 0.0,
            "elapsed_sec": time.perf_counter() - t0,
            "best_epoch": "",
            "val_dice_mean_foreground": "",
            **{f"val_dice_class_{i}": "" for i in range(NUM_CLASSES)},
        }

    model = UNet2D(in_channels=in_ch, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.lr)

    best_fg = -1.0
    best_epoch = 0
    best_metrics: Dict[str, float] = {}
    status = "OK"
    t_train = time.perf_counter()

    log_path = METRICS_DIR / f"{candidate.run_id}_training_log.csv"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["epoch", "train_loss", "val_dice_mean", "val_dice_mean_foreground"]
        + [f"val_dice_class_{i}" for i in range(NUM_CLASSES)]
    )

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, hyperparams.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            if not math.isfinite(train_loss):
                status = "NAN_LOSS"
                print(f"  EARLY STOP: NaN loss at epoch {epoch}")
                break
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
            f.flush()
            print(
                f"  epoch {epoch}/{hyperparams.epochs}  loss={train_loss:.4f}  "
                f"val_fg={val_metrics['dice_mean_foreground']:.4f}"
            )

            if val_metrics["dice_mean_foreground"] > best_fg:
                best_fg = val_metrics["dice_mean_foreground"]
                best_epoch = epoch
                best_metrics = val_metrics.copy()
                CKPT_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "run_id": candidate.run_id,
                        "val_dice_mean_foreground": best_fg,
                    },
                    CKPT_DIR / f"{candidate.run_id}_best.pt",
                )

            # Early stop if clearly worse than subset original after epoch >= 2.
            if (
                original_baseline_fg is not None
                and epoch >= 2
                and best_fg < (original_baseline_fg - 0.05)
            ):
                status = "EARLY_STOP_WORSE_THAN_BASELINE"
                print(
                    f"  EARLY STOP: best_fg={best_fg:.4f} << subset original "
                    f"{original_baseline_fg:.4f}"
                )
                break

    train_sec = time.perf_counter() - t_train
    result = {
        "run_id": candidate.run_id,
        "description": candidate.description,
        "imf_indices": str(candidate.imf_indices),
        "status": status,
        "preprocess_sec": round(preprocess_sec, 2),
        "train_sec": round(train_sec, 2),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "n_train_slices": len(train_ds),
        "n_val_slices": len(val_ds),
        "best_epoch": best_epoch if best_metrics else "",
        "val_dice_mean": best_metrics.get("dice_mean", ""),
        "val_dice_mean_foreground": best_metrics.get("dice_mean_foreground", ""),
    }
    for i in range(NUM_CLASSES):
        result[f"val_dice_class_{i}"] = best_metrics.get(f"dice_class_{i}", "")
        result[f"val_dice_{LABEL_NAMES[i]}"] = best_metrics.get(f"dice_{LABEL_NAMES[i]}", "")
    print(
        f"  done status={status} best_fg={result['val_dice_mean_foreground']} "
        f"preprocess={elapsed_str(preprocess_sec)} train={elapsed_str(train_sec)}"
    )
    return result


def stage3_screen(
    candidates: Sequence[ScreenCandidate],
    train_cases: Sequence[CasePair],
    val_cases: Sequence[CasePair],
    *,
    seed: int,
    device: torch.device,
    preprocess_workers: int,
    max_train_cases: int = 10,
    max_val_cases: int = 3,
    epochs: int = 3,
) -> Tuple[List[Dict[str, Any]], List[CasePair], List[CasePair]]:
    print("\n=== Stage 3: short training screen (3 epochs, small subset) ===")
    screen_train = fixed_subset(train_cases, max_train_cases, seed)
    screen_val = fixed_subset(val_cases, max_val_cases, seed + 3)
    print(f"  train cases ({len(screen_train)}): {[c.stem for c in screen_train]}")
    print(f"  val cases   ({len(screen_val)}): {[c.stem for c in screen_val]}")

    hyperparams = AblationHyperparams(epochs=epochs, batch_size=4, lr=1e-3, seed=seed)
    rows: List[Dict[str, Any]] = []
    original_fg: Optional[float] = None

    # Always run original first for subset baseline.
    ordered = sorted(candidates, key=lambda c: 0 if c.run_id == "original" else 1)
    for cand in ordered:
        row = screen_train_one(
            cand,
            screen_train,
            screen_val,
            hyperparams,
            device,
            preprocess_workers=preprocess_workers,
            original_baseline_fg=None if cand.run_id == "original" else original_fg,
        )
        rows.append(row)
        if cand.run_id == "original" and row.get("val_dice_mean_foreground") != "":
            original_fg = float(row["val_dice_mean_foreground"])
            print(f"  Subset original baseline FG Dice = {original_fg:.4f}")

    return rows, screen_train, screen_val


# ---------------------------------------------------------------------------
# Stage 4: selection (no auto full run)
# ---------------------------------------------------------------------------


def estimate_full_run_hours(
    *,
    n_train_slices: int = 5810,
    n_val_slices: int = 1245,
    epochs: int = 15,
    preprocess_sec_observed: float,
    screen_train_slices: int,
    screen_train_sec: float,
    screen_epochs: int = 3,
) -> Dict[str, float]:
    """Rough wall-clock estimate from observed screen timings."""
    # Scale preprocess by full vs screen slice counts (cache misses assumed for new config).
    screen_slices = max(screen_train_slices, 1)
    # Observed preprocess covers train+val screen; allocate proportionally.
    pre_per_slice = preprocess_sec_observed / max(screen_slices, 1)
    full_pre_sec = pre_per_slice * (n_train_slices + n_val_slices)

    train_per_epoch = screen_train_sec / max(screen_epochs, 1)
    # Full train has ~5810/screen_train_slices more steps per epoch.
    scale = n_train_slices / max(screen_train_slices, 1)
    full_train_sec = train_per_epoch * scale * epochs
    # Val cost rough: ~0.25 of train epoch cost at full size.
    full_val_sec = train_per_epoch * scale * 0.25 * epochs
    total = full_pre_sec + full_train_sec + full_val_sec
    return {
        "est_preprocess_hours": full_pre_sec / 3600,
        "est_train_hours": (full_train_sec + full_val_sec) / 3600,
        "est_total_hours": total / 3600,
    }


def stage4_select(
    rows: Sequence[Dict[str, Any]],
    stage2: Dict[str, Any],
    screen_train: Sequence[CasePair],
) -> Dict[str, Any]:
    print("\n=== Stage 4: selection ===")
    original = next((r for r in rows if r["run_id"] == "original"), None)
    if not original or original.get("val_dice_mean_foreground") == "":
        print("  No valid subset original baseline; cannot recommend a full run.")
        return {"recommend_full_run": False, "reason": "missing_original_baseline"}

    baseline_fg = float(original["val_dice_mean_foreground"])
    ranked = []
    for r in rows:
        if r["run_id"] == "original":
            continue
        if r.get("val_dice_mean_foreground") == "" or r.get("status") not in {
            "OK",
            "EARLY_STOP_WORSE_THAN_BASELINE",
        }:
            continue
        fg = float(r["val_dice_mean_foreground"])
        ranked.append((fg, r))
    ranked.sort(key=lambda x: x[0], reverse=True)

    print(f"  Subset original FG Dice: {baseline_fg:.4f}")
    print("  Ranking (validation FG Dice):")
    for fg, r in ranked:
        vis_ok = visuals_look_reasonable(stage2, r["run_id"])
        print(
            f"    {r['run_id']:22s}  fg={fg:.4f}  "
            f"delta={fg - baseline_fg:+.4f}  visual_ok={vis_ok}  status={r['status']}"
        )

    recommendation = None
    for fg, r in ranked:
        if fg <= baseline_fg:
            continue
        if not visuals_look_reasonable(stage2, r["run_id"]):
            print(f"  Reject {r['run_id']}: beats baseline numerically but visuals look poor.")
            continue
        recommendation = r
        break

    result: Dict[str, Any] = {
        "subset_original_fg": baseline_fg,
        "recommend_full_run": recommendation is not None,
        "recommended_run_id": recommendation["run_id"] if recommendation else None,
    }

    if recommendation is None:
        print("\n  RECOMMENDATION: no candidate is worth a full run.")
        print("  (None both beat the subset original baseline AND look anatomically reasonable.)")
        result["reason"] = "no_candidate_beats_baseline_with_ok_visuals"
        return result

    est = estimate_full_run_hours(
        preprocess_sec_observed=float(recommendation["preprocess_sec"]),
        screen_train_slices=int(recommendation.get("n_train_slices") or len(screen_train) * 70),
        screen_train_sec=float(recommendation["train_sec"]),
    )
    result.update(est)
    print(f"\n  RECOMMENDATION (do NOT auto-start): {recommendation['run_id']}")
    print(
        f"  Estimated full-run wall time on CPU: "
        f"~{est['est_total_hours']:.1f} hours "
        f"(preprocess ~{est['est_preprocess_hours']:.1f}h + train/val ~{est['est_train_hours']:.1f}h)"
    )
    print("  Waiting for your approval before any full experiment.")
    return result


def write_summary(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    # Stable column order.
    keys: List[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Staged EMD IMF screening (no auto full run).")
    p.add_argument("--splits-csv", type=Path, default=OUTPUTS_DIR / "splits_4ch.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--catalog",
        type=str,
        default="mid",
        choices=["mid", "lowfreq"],
        help="mid = IMF 2/3/... screen; lowfreq = Gastro-style negative IMF trend removal.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Screen output directory (default depends on --catalog).",
    )
    p.add_argument(
        "--preprocess-workers",
        type=int,
        default=2,
        help="Parallel EMD cache workers (0=serial). Default 2 for safer CPU use.",
    )
    p.add_argument("--screen-epochs", type=int, default=3)
    p.add_argument("--max-train-cases", type=int, default=10)
    p.add_argument("--max-val-cases", type=int, default=3)
    p.add_argument("--skip-train", action="store_true", help="Run stages 1-2 only.")
    return p.parse_args()


def main() -> int:
    global SCREEN_ROOT, VISUALS_DIR, METRICS_DIR, CKPT_DIR

    args = parse_args()
    if args.output_root is not None:
        SCREEN_ROOT = args.output_root
    elif args.catalog == "lowfreq":
        SCREEN_ROOT = OUTPUTS_DIR / "emd_lowfreq_imf_screen"
    else:
        SCREEN_ROOT = OUTPUTS_DIR / "emd_lower_imf_screen"
    VISUALS_DIR = SCREEN_ROOT / "visuals"
    METRICS_DIR = SCREEN_ROOT / "metrics"
    CKPT_DIR = SCREEN_ROOT / "checkpoints"

    SCREEN_ROOT.mkdir(parents=True, exist_ok=True)
    if not args.splits_csv.exists():
        print(f"Missing splits: {args.splits_csv}", file=sys.stderr)
        return 1

    catalog = candidate_catalog(args.catalog)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Catalog: {args.catalog}")
    print(f"Output root: {SCREEN_ROOT}")

    train_cases = load_split_cases(args.splits_csv, "train")
    val_cases = load_split_cases(args.splits_csv, "val")
    print(f"Split sizes: train={len(train_cases)} val={len(val_cases)} (unchanged)")

    stage1 = stage1_inspect_imfs(train_cases, val_cases, seed=args.seed)
    (SCREEN_ROOT / "stage1_imf_availability.json").write_text(
        json.dumps(stage1, indent=2), encoding="utf-8"
    )

    valid = select_valid_candidates(stage1, catalog)
    print("\nValid candidates for Stages 2–3:")
    for c in valid:
        print(f"  - {c.run_id}: {c.description}")

    stage2 = stage2_visuals(train_cases, val_cases, valid, seed=args.seed)
    (SCREEN_ROOT / "stage2_visual_notes.json").write_text(
        json.dumps(stage2, indent=2), encoding="utf-8"
    )

    rows: List[Dict[str, Any]] = []
    screen_train: List[CasePair] = []
    screen_val: List[CasePair] = []
    selection: Dict[str, Any] = {}

    if not args.skip_train:
        rows, screen_train, screen_val = stage3_screen(
            valid,
            train_cases,
            val_cases,
            seed=args.seed,
            device=device,
            preprocess_workers=args.preprocess_workers,
            max_train_cases=args.max_train_cases,
            max_val_cases=args.max_val_cases,
            epochs=args.screen_epochs,
        )
        # Attach visual OK flag.
        for row in rows:
            if row["run_id"] == "original":
                row["visual_ok"] = True
            else:
                row["visual_ok"] = visuals_look_reasonable(stage2, row["run_id"])
        selection = stage4_select(rows, stage2, screen_train)
        for row in rows:
            row["recommended_full_run"] = (
                selection.get("recommend_full_run")
                and selection.get("recommended_run_id") == row["run_id"]
            )
            if selection.get("recommended_run_id") == row["run_id"]:
                row["est_full_run_hours"] = selection.get("est_total_hours", "")
            else:
                row["est_full_run_hours"] = ""
    else:
        print("\n--skip-train set: skipping Stages 3–4 training screen.")

    summary_path = SCREEN_ROOT / "screening_summary.csv"
    write_summary(rows, summary_path)
    (SCREEN_ROOT / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    subset_meta = {
        "catalog": args.catalog,
        "train_stems": [c.stem for c in screen_train],
        "val_stems": [c.stem for c in screen_val],
        "seed": args.seed,
        "screen_epochs": args.screen_epochs,
    }
    (SCREEN_ROOT / "subset_cases.json").write_text(json.dumps(subset_meta, indent=2), encoding="utf-8")

    # Final console report
    print("\n========== SCREENING REPORT ==========")
    print(f"Catalog: {args.catalog}")
    print(f"Valid settings: {[c.run_id for c in valid if c.run_id != 'original']}")
    if rows:
        print(f"Summary CSV: {summary_path}")
        orig = next((r for r in rows if r["run_id"] == "original"), None)
        if orig:
            print(f"Subset original baseline FG Dice: {orig.get('val_dice_mean_foreground')}")
        print("Per-candidate:")
        for r in rows:
            print(
                f"  {r['run_id']:22s}  status={r.get('status')}  "
                f"pre={r.get('preprocess_sec')}s  train={r.get('train_sec')}s  "
                f"val_fg={r.get('val_dice_mean_foreground')}  "
                f"visual_ok={r.get('visual_ok')}"
            )
            pcs = [
                f"{LABEL_NAMES[i]}={r.get(f'val_dice_class_{i}')}"
                for i in range(1, NUM_CLASSES)
                if r.get(f"val_dice_class_{i}") != ""
            ]
            if pcs:
                print(f"    per-class: {', '.join(pcs)}")
        if selection.get("recommend_full_run"):
            print(
                f"\nWorth a full run: {selection['recommended_run_id']} "
                f"(est. ~{selection.get('est_total_hours', '?'):.1f} h on CPU)"
            )
            print("NOT starting full training — reply to approve if you want it.")
        else:
            print("\nNo candidate recommended for a full run.")
    print("======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
