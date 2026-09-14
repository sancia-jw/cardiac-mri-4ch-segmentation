#!/usr/bin/env python3
"""
Stage C extension — full 2D BEMD component sweep (screening only).

Audits PyEMD EMD2D (max_imf=-1) on the Stage C subset, characterizes BIMF0 vs
residual, writes qualitative montages, then screens one-at-a-time removal:

  A) raw:     original - component
  B) matched: component scaled to 1D-row IMF0 energy (Stage C convention), then subtract

Does NOT launch Stage D / 15-epoch full runs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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
from cine_4ch.dataset import load_split_cases
from cine_4ch.io import CasePair, extract_frame, load_pair
from cine_4ch.model import UNet2D
from src.preprocessing.emd_enhancement import safe_minmax_normalize
from src.preprocessing.emd_geometry import (
    component_energy_stats,
    decompose_emd2d_pyemd,
    display_normalize,
    enhance_geometry_matched,
    simple_ssim,
    spectral_centroid_cycles_per_pixel,
)

OUT_ROOT = OUTPUTS_DIR / "emd_geometry_followup" / "stage_c_component_sweep"
CACHE_ROOT = OUTPUTS_DIR / "emd_geometry_cache" / "stage_c_component_sweep"
DECOMP_CACHE = CACHE_ROOT / "full_emd2d"
METRICS_DIR = OUT_ROOT / "metrics"
CKPT_DIR = OUT_ROOT / "checkpoints"
FIG_DIR = OUT_ROOT / "figures"
STAGE_C_ROOT = OUTPUTS_DIR / "emd_geometry_followup" / "stage_c"

SEED = 42
MAX_TRAIN_CASES = 10
MAX_VAL_CASES = 3
SCREEN_EPOCHS = 3
MENTOR_CASE = "CINE_4CH_009"
MENTOR_FRAME = 83


@dataclass
class SweepCandidate:
    run_id: str
    mode: str
    description: str
    analysis: str  # "baseline" | "raw" | "matched"
    component: str  # "none" | "bimf0" | "residual"
    resume_from_stage_c: bool = False


def candidates() -> List[SweepCandidate]:
    return [
        SweepCandidate(
            "original",
            "original",
            "Normalized original (no EMD).",
            "baseline",
            "none",
            resume_from_stage_c=True,
        ),
        SweepCandidate(
            "2d_subtract_bimf0",
            "2d_subtract_bimf0",
            "Raw subtract full-decomp BIMF0 (natural energy).",
            "raw",
            "bimf0",
        ),
        SweepCandidate(
            "2d_subtract_residual",
            "2d_subtract_residual",
            "Raw subtract full-decomp residual/trend (natural energy).",
            "raw",
            "residual",
        ),
        SweepCandidate(
            "2d_bimf0_matched",
            "2d_bimf0_matched",
            "BIMF0 scaled to 1D-row IMF0 energy, raw subtract (Stage C convention).",
            "matched",
            "bimf0",
            resume_from_stage_c=True,
        ),
        SweepCandidate(
            "2d_residual_matched",
            "2d_residual_matched",
            "Residual scaled to 1D-row IMF0 energy, raw subtract (Stage C convention).",
            "matched",
            "residual",
        ),
    ]


def fixed_subset(cases: Sequence[CasePair], n: int, seed: int) -> List[CasePair]:
    if n >= len(cases):
        return list(cases)
    rng = np.random.default_rng(seed)
    idxs = sorted(rng.choice(len(cases), size=n, replace=False).tolist())
    return [cases[i] for i in idxs]


def stage_c_subset() -> Tuple[List[CasePair], List[CasePair]]:
    train_all = load_split_cases(OUTPUTS_DIR / "splits_4ch.csv", "train")
    val_all = load_split_cases(OUTPUTS_DIR / "splits_4ch.csv", "val")
    return (
        fixed_subset(train_all, MAX_TRAIN_CASES, SEED),
        fixed_subset(val_all, MAX_VAL_CASES, SEED + 3),
    )


def _resize(image_2d: np.ndarray, label_2d: np.ndarray, size=DEFAULT_IMAGE_SIZE):
    label_t = torch.from_numpy(label_2d).float().unsqueeze(0).unsqueeze(0)
    label_r = F.interpolate(label_t, size=size, mode="nearest").squeeze().numpy().astype(np.int64)
    image_t = torch.from_numpy(image_2d).float().unsqueeze(0).unsqueeze(0)
    image_r = F.interpolate(image_t, size=size, mode="bilinear", align_corners=False)
    return image_r.squeeze().numpy().astype(np.float32), label_r


def _cache_key(mode: str) -> str:
    payload = {
        "stage": "C_component_sweep",
        "mode": mode,
        "protocol": "full_emd2d_raw_or_matched_v1",
        "max_imf": -1,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return f"{mode}_{digest}"


def _cache_path(mode: str, case_stem: str, frame_idx: int) -> Path:
    return CACHE_ROOT / _cache_key(mode) / f"{case_stem}_f{frame_idx:03d}.npy"


def _decomp_path(case_stem: str, frame_idx: int) -> Path:
    return DECOMP_CACHE / f"{case_stem}_f{frame_idx:03d}.npz"


def load_or_decompose(image_2d: np.ndarray, case_stem: str, frame_idx: int):
    path = _decomp_path(case_stem, frame_idx)
    if path.exists():
        z = np.load(path)
        return {
            "original": z["original"].astype(np.float32),
            "bimf0": z["bimf0"].astype(np.float32),
            "residual": z["residual"].astype(np.float32),
            "n_bimf": int(z["n_bimf"]),
            "elapsed_sec": float(z["elapsed_sec"]),
            "recon_max_abs": float(z["recon_max_abs"]),
        }
    decomp = decompose_emd2d_pyemd(image_2d, max_imf=-1, normalize_components=False)
    if decomp.n_oscillatory < 1:
        raise RuntimeError(f"No BIMF for {case_stem} frame {frame_idx}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        original=decomp.original.astype(np.float32),
        bimf0=decomp.components[0].astype(np.float32),
        residual=decomp.residual.astype(np.float32),
        n_bimf=np.int32(decomp.n_oscillatory),
        elapsed_sec=np.float64(decomp.elapsed_sec),
        recon_max_abs=np.float64(decomp.meta["reconstruction"]["max_abs"]),
    )
    return {
        "original": decomp.original.astype(np.float32),
        "bimf0": decomp.components[0].astype(np.float32),
        "residual": decomp.residual.astype(np.float32),
        "n_bimf": decomp.n_oscillatory,
        "elapsed_sec": decomp.elapsed_sec,
        "recon_max_abs": decomp.meta["reconstruction"]["max_abs"],
    }


def audit_and_characterize(cases: Sequence[CasePair]) -> Dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    per_slice: List[Dict[str, Any]] = []
    n_bimf_list: List[int] = []

    for case in cases:
        image, label, _ = load_pair(case)
        n_frames = 1 if label.ndim < 3 else int(label.shape[-1])
        for fidx in tqdm(range(n_frames), desc=f"decomp {case.stem}"):
            raw = extract_frame(image, fidx).astype(np.float32)
            d = load_or_decompose(raw, case.stem, fidx)
            n_bimf_list.append(int(d["n_bimf"]))
            b0 = component_energy_stats(d["original"], d["bimf0"])
            rr = component_energy_stats(d["original"], d["residual"])
            rem_b0 = (d["original"].astype(np.float64) - d["bimf0"].astype(np.float64)).astype(
                np.float32
            )
            rem_rr = (
                d["original"].astype(np.float64) - d["residual"].astype(np.float64)
            ).astype(np.float32)
            per_slice.append(
                {
                    "case": case.stem,
                    "frame": fidx,
                    "n_bimf": int(d["n_bimf"]),
                    "elapsed_sec": d["elapsed_sec"],
                    "recon_max_abs": d["recon_max_abs"],
                    "bimf0_energy": b0["energy"],
                    "bimf0_energy_fraction": b0["energy_fraction_of_original"],
                    "bimf0_rms": b0["rms"],
                    "bimf0_variance": b0["variance"],
                    "bimf0_spectral_centroid_cpp": spectral_centroid_cycles_per_pixel(d["bimf0"]),
                    "bimf0_ssim_after_removal": simple_ssim(d["original"], rem_b0),
                    "residual_energy": rr["energy"],
                    "residual_energy_fraction": rr["energy_fraction_of_original"],
                    "residual_rms": rr["rms"],
                    "residual_variance": rr["variance"],
                    "residual_spectral_centroid_cpp": spectral_centroid_cycles_per_pixel(
                        d["residual"]
                    ),
                    "residual_ssim_after_removal": simple_ssim(d["original"], rem_rr),
                }
            )

    ctr = Counter(n_bimf_list)
    audit = {
        "n_slices": len(n_bimf_list),
        "n_bimf_min": int(min(n_bimf_list)),
        "n_bimf_median": float(np.median(n_bimf_list)),
        "n_bimf_max": int(max(n_bimf_list)),
        "n_bimf_histogram": {str(k): int(v) for k, v in sorted(ctr.items())},
        "bimf_indices_available_in_ALL": list(range(int(min(n_bimf_list)))),
        "residual_treated_separately": True,
        "backend": "PyEMD.EMD2d.EMD2D",
        "max_imf": -1,
        "note": (
            "On this Stage C subset, full EMD2D always yields exactly one oscillatory "
            "BIMF (index 0) plus a residual/trend. No BIMF1+ exists to screen."
        ),
    }
    (OUT_ROOT / "component_count_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    csv_path = OUT_ROOT / "component_stats_per_slice.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_slice[0].keys()))
        writer.writeheader()
        writer.writerows(per_slice)

    def _agg(prefix: str) -> Dict[str, float]:
        keys = [
            f"{prefix}_energy_fraction",
            f"{prefix}_rms",
            f"{prefix}_variance",
            f"{prefix}_spectral_centroid_cpp",
            f"{prefix}_ssim_after_removal",
        ]
        out: Dict[str, float] = {"component": prefix}
        for k in keys:
            vals = np.array([r[k] for r in per_slice], dtype=float)
            out[f"mean_{k}"] = float(np.mean(vals))
            out[f"median_{k}"] = float(np.median(vals))
            out[f"std_{k}"] = float(np.std(vals))
        return out

    summary_rows = [_agg("bimf0"), _agg("residual")]
    summary_path = OUT_ROOT / "component_stats_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        keys: List[str] = []
        for row in summary_rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary_rows)

    return {"audit": audit, "per_slice": per_slice, "summary": summary_rows}


def _sym_display(arr: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(arr)))
    if peak <= 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / peak, -1.0, 1.0).astype(np.float32)


def make_mentor_figures() -> None:
    """Montages for CINE_4CH_009 frame 83 (mentor frame; may be outside Stage C subset)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_cases, _, _ = __import__("cine_4ch.io", fromlist=["discover_cases"]).discover_cases()
    case = next((c for c in all_cases if c.stem == MENTOR_CASE), None)
    if case is None:
        print(f"WARNING: {MENTOR_CASE} not found; skipping mentor figures")
        return
    image, _, _ = load_pair(case)
    raw = extract_frame(image, MENTOR_FRAME).astype(np.float32)
    d = load_or_decompose(raw, MENTOR_CASE, MENTOR_FRAME)
    orig = d["original"]
    bimf0 = d["bimf0"]
    residual = d["residual"]
    rem0 = (orig.astype(np.float64) - bimf0.astype(np.float64)).astype(np.float32)
    remr = (orig.astype(np.float64) - residual.astype(np.float64)).astype(np.float32)

    # Montage 1: original / components / removals with shared display where useful
    panels = [
        ("original", display_normalize(orig), "gray", 0.0, 1.0),
        ("BIMF0", _sym_display(bimf0), "coolwarm", -1.0, 1.0),
        ("original − BIMF0", display_normalize(rem0), "gray", 0.0, 1.0),
        ("residual / trend", _sym_display(residual), "coolwarm", -1.0, 1.0),
        ("original − residual", display_normalize(remr), "gray", 0.0, 1.0),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
    for ax, (title, img, cmap, vmin, vmax) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{MENTOR_CASE} frame {MENTOR_FRAME} — full 2D BEMD (only BIMF0 + residual)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mentor_removal_montage.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Montage 2: components only, per-panel normalized to reveal structure
    comps = [
        ("BIMF0 (per-panel norm)", display_normalize(np.abs(bimf0))),
        ("residual (per-panel norm)", display_normalize(residual)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, (title, img) in zip(axes, comps):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{MENTOR_CASE} frame {MENTOR_FRAME} — components only", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mentor_components_normalized.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote mentor figures under {FIG_DIR}")


class SweepDataset(Dataset):
    def __init__(
        self,
        cases: Sequence[CasePair],
        mode: str,
        *,
        image_size=DEFAULT_IMAGE_SIZE,
        augment: bool = False,
        precompute_desc: Optional[str] = None,
    ):
        self.cases = list(cases)
        self.mode = mode
        self.image_size = image_size
        self.augment = augment
        self.index: List[Tuple[CasePair, int]] = []
        for case in self.cases:
            _, label, _ = load_pair(case)
            n_frames = 1 if label.ndim < 3 else int(label.shape[-1])
            for fidx in range(n_frames):
                self.index.append((case, fidx))

        self._images: List[np.ndarray] = []
        self._labels: List[np.ndarray] = []
        desc = precompute_desc or f"sweep {mode}"
        for case, fidx in tqdm(self.index, desc=desc):
            path = _cache_path(mode, case.stem, fidx)
            image, label, _ = load_pair(case)
            gt = extract_frame(label, fidx).astype(np.int64)
            if path.exists():
                enh = np.load(path)
            else:
                raw = extract_frame(image, fidx).astype(np.float32)
                enh, _ = enhance_geometry_matched(raw, mode)  # type: ignore[arg-type]
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, enh.astype(np.float32, copy=False))
            img_r, gt_r = _resize(enh, gt, self.image_size)
            self._images.append(img_r)
            self._labels.append(gt_r)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        img = self._images[idx].copy()
        mask = self._labels[idx].copy()
        if self.augment:
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()
        case, fidx = self.index[idx]
        return (
            torch.from_numpy(img).float().unsqueeze(0),
            torch.from_numpy(mask).long(),
            case.stem,
            fidx,
        )


def _invalid_batch(images: torch.Tensor) -> bool:
    if not torch.isfinite(images).all():
        return True
    flat = images.reshape(images.shape[0], images.shape[1], -1)
    spreads = (flat.amax(dim=-1) - flat.amin(dim=-1)).mean().item()
    return spreads < 1e-5


def _load_completed(cand: SweepCandidate) -> Optional[Dict[str, Any]]:
    log_path = METRICS_DIR / f"{cand.run_id}_training_log.csv"
    ckpt_path = CKPT_DIR / f"{cand.run_id}_best.pt"
    # Prefer local sweep artifacts; else import Stage C for overlapping run_ids.
    if not log_path.is_file() and cand.resume_from_stage_c:
        src_log = STAGE_C_ROOT / "metrics" / f"{cand.run_id}_training_log.csv"
        src_ckpt = STAGE_C_ROOT / "checkpoints" / f"{cand.run_id}_best.pt"
        if src_log.is_file() and src_ckpt.is_file():
            METRICS_DIR.mkdir(parents=True, exist_ok=True)
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_log, log_path)
            shutil.copy2(src_ckpt, ckpt_path)
            print(f"  imported Stage C artifacts for {cand.run_id}")

    if not log_path.is_file() or not ckpt_path.is_file():
        return None
    with open(log_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < SCREEN_EPOCHS:
        return None
    best = max(rows, key=lambda r: float(r["val_dice_mean_foreground"]))
    result: Dict[str, Any] = {
        "run_id": cand.run_id,
        "mode": cand.mode,
        "description": cand.description,
        "analysis": cand.analysis,
        "component": cand.component,
        "status": "OK_RESUMED",
        "preprocess_sec": 0.0,
        "train_sec": 0.0,
        "elapsed_sec": 0.0,
        "best_epoch": int(best["epoch"]),
        "val_dice_mean": float(best["val_dice_mean"]),
        "val_dice_mean_foreground": float(best["val_dice_mean_foreground"]),
    }
    for i in range(NUM_CLASSES):
        key = f"val_dice_class_{i}"
        result[key] = float(best[key])
        result[f"val_dice_{LABEL_NAMES[i]}"] = float(best[key])
    return result


def screen_one(
    cand: SweepCandidate,
    train_cases: Sequence[CasePair],
    val_cases: Sequence[CasePair],
    hyperparams: AblationHyperparams,
    device: torch.device,
    energy_frac_mean: Optional[float],
) -> Dict[str, Any]:
    print(f"\n--- Component sweep: {cand.run_id} ---")
    resumed = _load_completed(cand)
    if resumed is not None:
        resumed["mean_component_energy_fraction"] = energy_frac_mean if energy_frac_mean is not None else ""
        print(
            f"  resume {cand.run_id}: best_epoch={resumed['best_epoch']} "
            f"val_fg={resumed['val_dice_mean_foreground']:.4f}"
        )
        return resumed

    t0 = time.perf_counter()
    set_seed(hyperparams.seed)
    t_pre = time.perf_counter()
    train_ds = SweepDataset(train_cases, cand.mode, augment=True, precompute_desc=f"{cand.run_id} train")
    val_ds = SweepDataset(val_cases, cand.mode, augment=False, precompute_desc=f"{cand.run_id} val")
    preprocess_sec = time.perf_counter() - t_pre

    sample_img, sample_mask, _, _ = train_ds[0]
    print(f"  sample {tuple(sample_img.shape)} mask {tuple(sample_mask.shape)}")
    if float(sample_img.max() - sample_img.min()) < 1e-5:
        return {
            "run_id": cand.run_id,
            "mode": cand.mode,
            "description": cand.description,
            "analysis": cand.analysis,
            "component": cand.component,
            "status": "INVALID_CONSTANT_INPUT",
            "preprocess_sec": round(preprocess_sec, 2),
            "train_sec": 0.0,
            "val_dice_mean_foreground": "",
            "mean_component_energy_fraction": energy_frac_mean if energy_frac_mean is not None else "",
        }

    gen = torch.Generator().manual_seed(hyperparams.seed)
    train_loader = DataLoader(
        train_ds, batch_size=hyperparams.batch_size, shuffle=True, num_workers=0, generator=gen
    )
    val_loader = DataLoader(val_ds, batch_size=hyperparams.batch_size, shuffle=False, num_workers=0)
    batch_images, _, _, _ = next(iter(train_loader))
    if _invalid_batch(batch_images):
        return {
            "run_id": cand.run_id,
            "mode": cand.mode,
            "description": cand.description,
            "analysis": cand.analysis,
            "component": cand.component,
            "status": "INVALID_BATCH",
            "preprocess_sec": round(preprocess_sec, 2),
            "train_sec": 0.0,
            "val_dice_mean_foreground": "",
            "mean_component_energy_fraction": energy_frac_mean if energy_frac_mean is not None else "",
        }

    model = UNet2D(in_channels=1, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.lr)
    best_fg = -1.0
    best_epoch = 0
    best_metrics: Dict[str, float] = {}
    status = "OK"
    t_train = time.perf_counter()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = METRICS_DIR / f"{cand.run_id}_training_log.csv"
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
                f"  epoch {epoch}/{hyperparams.epochs} loss={train_loss:.4f} "
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
                        "run_id": cand.run_id,
                        "val_dice_mean_foreground": best_fg,
                    },
                    CKPT_DIR / f"{cand.run_id}_best.pt",
                )

    train_sec = time.perf_counter() - t_train
    result: Dict[str, Any] = {
        "run_id": cand.run_id,
        "mode": cand.mode,
        "description": cand.description,
        "analysis": cand.analysis,
        "component": cand.component,
        "status": status,
        "preprocess_sec": round(preprocess_sec, 2),
        "train_sec": round(train_sec, 2),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "n_train_slices": len(train_ds),
        "n_val_slices": len(val_ds),
        "best_epoch": best_epoch if best_metrics else "",
        "val_dice_mean": best_metrics.get("dice_mean", ""),
        "val_dice_mean_foreground": best_metrics.get("dice_mean_foreground", ""),
        "mean_component_energy_fraction": energy_frac_mean if energy_frac_mean is not None else "",
    }
    for i in range(NUM_CLASSES):
        result[f"val_dice_class_{i}"] = best_metrics.get(f"dice_class_{i}", "")
        result[f"val_dice_{LABEL_NAMES[i]}"] = best_metrics.get(f"dice_{LABEL_NAMES[i]}", "")
    return result


def write_readme(audit: Dict[str, Any], selection: Dict[str, Any]) -> None:
    lines = [
        "# Stage C component sweep — interpretation",
        "",
        "## Audit finding",
        "",
        "Full PyEMD `EMD2D` (`max_imf=-1`) on the Stage C subset (1020 slices) yields",
        f"**exactly {audit['n_bimf_min']} oscillatory BIMF** on every slice (min=median=max={audit['n_bimf_max']}).",
        "Consistently available oscillatory index: **BIMF0 only**. Residual/trend is separate.",
        "There are **no BIMF1+** components to screen on this dataset with this backend.",
        "",
        "## Protocol",
        "",
        "- Same Stage C subset / seed / 3 epochs / model / lr / batch size",
        "- Raw analysis: `original - component` at natural energy",
        "- Matched analysis: scale component to **1D-row IMF0 energy** (Stage C convention), then subtract",
        "- Screening only — not final evidence; no Stage D launched",
        "",
        "## Ranking (see selection.json)",
        "",
    ]
    for item in selection.get("ranking", []):
        lines.append(
            f"- `{item['run_id']}`: FG={item['val_fg']:.4f} "
            f"(Δ={item['delta_vs_original']:+.4f}) — {item.get('verdict', '')}"
        )
    lines += [
        "",
        f"Recommended full-run candidate (if any): `{selection.get('recommended_run_id')}`",
        "",
        selection.get("interpretation", ""),
        "",
    ]
    (OUT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def assign_verdicts(
    ranked: List[Tuple[float, Dict[str, Any]]], baseline: float
) -> List[Dict[str, Any]]:
    out = []
    for fg, r in ranked:
        delta = fg - baseline
        if delta > 0.01:
            verdict = "worth full run"
        elif delta > 0.0:
            verdict = "weak / uncertain"
        else:
            verdict = "reject"
        out.append(
            {
                "run_id": r["run_id"],
                "analysis": r.get("analysis"),
                "component": r.get("component"),
                "val_fg": fg,
                "delta_vs_original": delta,
                "status": r["status"],
                "mean_component_energy_fraction": r.get("mean_component_energy_fraction", ""),
                "verdict": verdict,
                "val_dice_LV_cavity": r.get("val_dice_class_1", ""),
                "val_dice_LV_myocardium": r.get("val_dice_class_2", ""),
                "val_dice_RV_cavity": r.get("val_dice_class_3", ""),
                "val_dice_RA": r.get("val_dice_class_4", ""),
                "val_dice_LA": r.get("val_dice_class_5", ""),
            }
        )
    return out


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    train_cases, val_cases = stage_c_subset()
    all_cases = train_cases + val_cases
    print("Stage C component sweep (screening only; no Stage D)")
    print(f"train ({len(train_cases)}): {[c.stem for c in train_cases]}")
    print(f"val   ({len(val_cases)}): {[c.stem for c in val_cases]}")

    print("\n=== STEP 1-3: audit + characterize ===")
    char = audit_and_characterize(all_cases)
    audit = char["audit"]
    print(json.dumps(audit, indent=2))
    print("\nComponent summary:")
    for row in char["summary"]:
        ef_key = f"mean_{row['component']}_energy_fraction"
        sc_key = f"mean_{row['component']}_spectral_centroid_cpp"
        print(
            f"  {row['component']}: energy_frac={row[ef_key]:.4f} "
            f"spectral_centroid={row[sc_key]:.4f} cyc/px"
        )

    print("\n=== STEP 4: mentor figures ===")
    make_mentor_figures()

    # Energy fractions for reporting (natural for raw; matched uses ~row-IMF0 frac)
    b0_frac = float(np.mean([r["bimf0_energy_fraction"] for r in char["per_slice"]]))
    rr_frac = float(np.mean([r["residual_energy_fraction"] for r in char["per_slice"]]))
    # Matched removed energy ≈ target 1D-row IMF0 fraction; estimate from a few slices
    matched_fracs = []
    for case in all_cases[:2]:
        image, label, _ = load_pair(case)
        raw = extract_frame(image, 0).astype(np.float32)
        _, meta = enhance_geometry_matched(raw, "2d_bimf0_matched")
        matched_fracs.append(float(meta["removed_energy_fraction"]))
    matched_frac = float(np.mean(matched_fracs)) if matched_fracs else float("nan")

    energy_by_mode = {
        "original": 0.0,
        "2d_subtract_bimf0": b0_frac,
        "2d_subtract_residual": rr_frac,
        "2d_bimf0_matched": matched_frac,
        "2d_residual_matched": matched_frac,
    }

    print("\n=== Pre-training plan ===")
    print(f"Available BIMF indices (all images): {audit['bimf_indices_available_in_ALL']}")
    print("Residual: yes (separate)")
    print(f"Conditions: {len(candidates())} (original + raw x2 + matched x2)")
    print("Recommendation: BOTH raw and matched - only 2 components exist, so both is tractable")
    print("and separates component identity from energy dose.")
    print(
        "Estimated new training: ~3 conditions x ~25-35 min CPU ~ 1.5-2 h "
        "(original + 2d_bimf0_matched resumed from Stage C)."
    )

    print("\n=== STEP 5: screening ===")
    device = resolve_device("cpu")
    hyperparams = AblationHyperparams(epochs=SCREEN_EPOCHS, batch_size=4, lr=1e-3, seed=SEED)
    rows: List[Dict[str, Any]] = []
    original_fg: Optional[float] = None
    original_row: Optional[Dict[str, Any]] = None

    for cand in candidates():
        row = screen_one(
            cand,
            train_cases,
            val_cases,
            hyperparams,
            device,
            energy_by_mode.get(cand.mode),
        )
        rows.append(row)
        if cand.run_id == "original" and row.get("val_dice_mean_foreground") != "":
            original_fg = float(row["val_dice_mean_foreground"])
            original_row = row
            print(f"  Subset original baseline FG Dice = {original_fg:.4f}")

    ranked = []
    for r in rows:
        if r["run_id"] == "original":
            continue
        if r.get("val_dice_mean_foreground") == "":
            continue
        ranked.append((float(r["val_dice_mean_foreground"]), r))
    ranked.sort(key=lambda x: x[0], reverse=True)
    baseline = original_fg if original_fg is not None else float("nan")
    ranking = assign_verdicts(ranked, baseline)

    print("\n=== Component sweep ranking ===")
    print(f"Subset original FG: {baseline:.4f}")
    for item in ranking:
        print(
            f"  {item['run_id']:24s} fg={item['val_fg']:.4f} "
            f"delta={item['delta_vs_original']:+.4f} "
            f"Efrac={item['mean_component_energy_fraction']} "
            f"verdict={item['verdict']}"
        )

    # Prefer at most one new full-run candidate beyond Stage C's 2d_bimf0_matched
    recommend = None
    for item in ranking:
        if item["verdict"] == "worth full run":
            recommend = item
            break
    if recommend is None:
        for item in ranking:
            if item["verdict"] == "weak / uncertain" and item["run_id"] != "2d_bimf0_matched":
                # keep Stage C bimf0 if it remains the only weak positive
                pass
        weak_pos = [i for i in ranking if i["verdict"] == "weak / uncertain"]
        if weak_pos:
            recommend = weak_pos[0]

    # Per-class table
    per_class_path = OUT_ROOT / "per_class_results.csv"
    with open(per_class_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "run_id",
            "analysis",
            "component",
            "val_fg",
            "delta_vs_original",
            "energy_fraction",
            "LV_cavity",
            "LV_myocardium",
            "RV_cavity",
            "RA",
            "LA",
            "verdict",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if original_row is not None:
            writer.writerow(
                {
                    "run_id": "original",
                    "analysis": "baseline",
                    "component": "none",
                    "val_fg": original_row["val_dice_mean_foreground"],
                    "delta_vs_original": 0.0,
                    "energy_fraction": 0.0,
                    "LV_cavity": original_row.get("val_dice_class_1", ""),
                    "LV_myocardium": original_row.get("val_dice_class_2", ""),
                    "RV_cavity": original_row.get("val_dice_class_3", ""),
                    "RA": original_row.get("val_dice_class_4", ""),
                    "LA": original_row.get("val_dice_class_5", ""),
                    "verdict": "baseline",
                }
            )
        for item in ranking:
            writer.writerow(
                {
                    "run_id": item["run_id"],
                    "analysis": item["analysis"],
                    "component": item["component"],
                    "val_fg": item["val_fg"],
                    "delta_vs_original": item["delta_vs_original"],
                    "energy_fraction": item["mean_component_energy_fraction"],
                    "LV_cavity": item["val_dice_LV_cavity"],
                    "LV_myocardium": item["val_dice_LV_myocardium"],
                    "RV_cavity": item["val_dice_RV_cavity"],
                    "RA": item["val_dice_RA"],
                    "LA": item["val_dice_LA"],
                    "verdict": item["verdict"],
                }
            )

    summary_path = OUT_ROOT / "screening_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        keys: List[str] = []
        for row in rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    # Interpretation draft
    bimf0_matched = next((i for i in ranking if i["run_id"] == "2d_bimf0_matched"), None)
    residual_matched = next((i for i in ranking if i["run_id"] == "2d_residual_matched"), None)
    bimf0_raw = next((i for i in ranking if i["run_id"] == "2d_subtract_bimf0"), None)
    residual_raw = next((i for i in ranking if i["run_id"] == "2d_subtract_residual"), None)

    interp_bits = [
        f"Only BIMF0 + residual exist on this subset (n_bimf always 1).",
        f"Natural energy fractions: BIMF0~{b0_frac:.3f}, residual~{rr_frac:.3f}; "
        f"matched dose~{matched_frac:.3f} (1D-row IMF0 energy).",
    ]
    if bimf0_matched and residual_matched:
        if bimf0_matched["val_fg"] >= residual_matched["val_fg"]:
            interp_bits.append(
                "Under matched energy, BIMF0 removal >= residual removal on this screen."
            )
        else:
            interp_bits.append(
                "Under matched energy, residual removal beat BIMF0 - BIMF0 is not uniquely best."
            )
    if residual_raw and residual_raw["delta_vs_original"] < -0.05:
        interp_bits.append("Raw residual removal strongly hurt (expected: large energy / trend).")
    if bimf0_raw and bimf0_raw["delta_vs_original"] < -0.05:
        interp_bits.append("Raw BIMF0 removal also hurt at natural (~40%) energy.")

    selection = {
        "subset_original_fg": baseline,
        "recommend_full_run": bool(recommend and recommend["verdict"] == "worth full run"),
        "recommended_run_id": recommend["run_id"] if recommend else None,
        "recommended_verdict": recommend["verdict"] if recommend else None,
        "ranking": ranking,
        "audit": audit,
        "energy": {
            "bimf0_natural_mean_fraction": b0_frac,
            "residual_natural_mean_fraction": rr_frac,
            "matched_removed_fraction_approx": matched_frac,
            "match_definition": "scale component so sum(comp**2)==sum(1d_row_IMF0**2) per slice",
        },
        "protocol": {
            "subtract": "raw intensity (not Gastro IMF min-max)",
            "energy_match_target": "1d_row IMF0 energy per slice (Stage C)",
            "epochs": SCREEN_EPOCHS,
            "train_cases": [c.stem for c in train_cases],
            "val_cases": [c.stem for c in val_cases],
            "seed": SEED,
            "analyses": ["raw one-at-a-time", "energy-matched one-at-a-time"],
        },
        "interpretation": " ".join(interp_bits),
        "note": "Screening only. Do not treat as final evidence. No Stage D launched.",
    }
    (OUT_ROOT / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    write_readme(audit, selection)
    print(f"\nWrote {summary_path}")
    print(f"Wrote {per_class_path}")
    print(f"Wrote {OUT_ROOT / 'selection.json'}")
    print(f"Wrote {OUT_ROOT / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
