#!/usr/bin/env python3
"""
Stage C — small geometry EMD screen (no full 15-epoch runs).

Compares energy-matched preprocessing conditions on a fixed small subset:
  - original
  - 1d_row_imf0_raw
  - 1d_col_imf0_matched  (energy matched to row IMF0)
  - 2d_bimf0_matched
  - fft_hp_matched

Uses raw subtract (not Gastro IMF min-max) so energy matching is meaningful.
Screening is for prioritization only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from src.preprocessing.emd_geometry import enhance_geometry_matched

OUT_ROOT = OUTPUTS_DIR / "emd_geometry_followup" / "stage_c"
CACHE_ROOT = OUTPUTS_DIR / "emd_geometry_cache" / "stage_c"
METRICS_DIR = OUT_ROOT / "metrics"
CKPT_DIR = OUT_ROOT / "checkpoints"

SEED = 42
MAX_TRAIN_CASES = 10
MAX_VAL_CASES = 3
SCREEN_EPOCHS = 3


@dataclass
class StageCCandidate:
    run_id: str
    mode: str
    description: str


def candidates() -> List[StageCCandidate]:
    return [
        StageCCandidate("original", "original", "Normalized original (no EMD)."),
        StageCCandidate(
            "1d_row_imf0_raw",
            "1d_row_imf0_raw",
            "1D row-major IMF0 raw subtract (energy reference).",
        ),
        StageCCandidate(
            "1d_col_imf0_matched",
            "1d_col_imf0_matched",
            "1D column-major IMF0 scaled to row IMF0 energy, raw subtract.",
        ),
        StageCCandidate(
            "2d_bimf0_matched",
            "2d_bimf0_matched",
            "PyEMD EMD2D BIMF0 scaled to row IMF0 energy, raw subtract.",
        ),
        StageCCandidate(
            "fft_hp_matched",
            "fft_hp_matched",
            "Fourier high-pass scaled to row IMF0 energy, raw subtract.",
        ),
    ]


def fixed_subset(cases: Sequence[CasePair], n: int, seed: int) -> List[CasePair]:
    if n >= len(cases):
        return list(cases)
    rng = np.random.default_rng(seed)
    idxs = sorted(rng.choice(len(cases), size=n, replace=False).tolist())
    return [cases[i] for i in idxs]


def _resize(image_2d: np.ndarray, label_2d: np.ndarray, size=DEFAULT_IMAGE_SIZE):
    label_t = torch.from_numpy(label_2d).float().unsqueeze(0).unsqueeze(0)
    label_r = F.interpolate(label_t, size=size, mode="nearest").squeeze().numpy().astype(np.int64)
    image_t = torch.from_numpy(image_2d).float().unsqueeze(0).unsqueeze(0)
    image_r = F.interpolate(image_t, size=size, mode="bilinear", align_corners=False)
    return image_r.squeeze().numpy().astype(np.float32), label_r


def _cache_key(mode: str) -> str:
    payload = {"stage": "C", "mode": mode, "protocol": "raw_energy_matched_v1"}
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return f"{mode}_{digest}"


def _cache_path(mode: str, case_stem: str, frame_idx: int) -> Path:
    return CACHE_ROOT / _cache_key(mode) / f"{case_stem}_f{frame_idx:03d}.npy"


class GeometryScreenDataset(Dataset):
    """Slice dataset with Stage-C geometry preprocess + disk cache."""

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
        desc = precompute_desc or f"geometry {mode}"
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
        image_t = torch.from_numpy(img).float().unsqueeze(0)
        mask_t = torch.from_numpy(mask).long()
        return image_t, mask_t, case.stem, fidx


def _invalid_batch(images: torch.Tensor) -> bool:
    if not torch.isfinite(images).all():
        return True
    flat = images.reshape(images.shape[0], images.shape[1], -1)
    spreads = (flat.amax(dim=-1) - flat.amin(dim=-1)).mean().item()
    return spreads < 1e-5


def _load_completed_screen(cand: StageCCandidate, epochs: int) -> Optional[Dict[str, Any]]:
    """Reuse a finished screen run (full epoch log + checkpoint) after restarts."""
    log_path = METRICS_DIR / f"{cand.run_id}_training_log.csv"
    ckpt_path = CKPT_DIR / f"{cand.run_id}_best.pt"
    if not log_path.is_file() or not ckpt_path.is_file():
        return None
    with open(log_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < epochs:
        return None
    best = max(rows, key=lambda r: float(r["val_dice_mean_foreground"]))
    result: Dict[str, Any] = {
        "run_id": cand.run_id,
        "mode": cand.mode,
        "description": cand.description,
        "status": "OK_RESUMED",
        "preprocess_sec": 0.0,
        "train_sec": 0.0,
        "elapsed_sec": 0.0,
        "n_train_slices": "",
        "n_val_slices": "",
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
    cand: StageCCandidate,
    train_cases: Sequence[CasePair],
    val_cases: Sequence[CasePair],
    hyperparams: AblationHyperparams,
    device: torch.device,
) -> Dict[str, Any]:
    print(f"\n--- Stage C screen: {cand.run_id} ---")
    resumed = _load_completed_screen(cand, hyperparams.epochs)
    if resumed is not None:
        print(
            f"  resume {cand.run_id}: best_epoch={resumed['best_epoch']} "
            f"val_fg={resumed['val_dice_mean_foreground']:.4f}"
        )
        return resumed

    t0 = time.perf_counter()
    set_seed(hyperparams.seed)

    t_pre = time.perf_counter()
    train_ds = GeometryScreenDataset(
        train_cases, cand.mode, augment=True, precompute_desc=f"{cand.run_id} train"
    )
    val_ds = GeometryScreenDataset(
        val_cases, cand.mode, augment=False, precompute_desc=f"{cand.run_id} val"
    )
    preprocess_sec = time.perf_counter() - t_pre

    sample_img, sample_mask, _, _ = train_ds[0]
    print(f"  sample {tuple(sample_img.shape)} mask {tuple(sample_mask.shape)}")
    if float(sample_img.max() - sample_img.min()) < 1e-5:
        return {
            "run_id": cand.run_id,
            "mode": cand.mode,
            "description": cand.description,
            "status": "INVALID_CONSTANT_INPUT",
            "preprocess_sec": round(preprocess_sec, 2),
            "train_sec": 0.0,
            "val_dice_mean_foreground": "",
        }

    gen = torch.Generator().manual_seed(hyperparams.seed)
    train_loader = DataLoader(
        train_ds, batch_size=hyperparams.batch_size, shuffle=True, num_workers=0, generator=gen
    )
    val_loader = DataLoader(
        val_ds, batch_size=hyperparams.batch_size, shuffle=False, num_workers=0
    )
    batch_images, _, _, _ = next(iter(train_loader))
    if _invalid_batch(batch_images):
        return {
            "run_id": cand.run_id,
            "mode": cand.mode,
            "description": cand.description,
            "status": "INVALID_BATCH",
            "preprocess_sec": round(preprocess_sec, 2),
            "train_sec": 0.0,
            "val_dice_mean_foreground": "",
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
            # Do not early-stop on final-baseline gap during short screens:
            # original itself is often ~0.55 FG at epoch 2 and only reaches
            # ~0.80 at epoch 3, so comparing mid-screen to final FG is invalid.

    train_sec = time.perf_counter() - t_train
    result: Dict[str, Any] = {
        "run_id": cand.run_id,
        "mode": cand.mode,
        "description": cand.description,
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
    return result


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = resolve_device("cpu")
    print(f"Device: {device}")
    print("Stage C protocol: raw energy-matched subtract; screen only; no full run.")

    train_all = load_split_cases(OUTPUTS_DIR / "splits_4ch.csv", "train")
    val_all = load_split_cases(OUTPUTS_DIR / "splits_4ch.csv", "val")
    train_cases = fixed_subset(train_all, MAX_TRAIN_CASES, SEED)
    val_cases = fixed_subset(val_all, MAX_VAL_CASES, SEED + 3)
    print(f"train ({len(train_cases)}): {[c.stem for c in train_cases]}")
    print(f"val   ({len(val_cases)}): {[c.stem for c in val_cases]}")

    hyperparams = AblationHyperparams(
        epochs=SCREEN_EPOCHS, batch_size=4, lr=1e-3, seed=SEED
    )
    rows: List[Dict[str, Any]] = []
    original_fg: Optional[float] = None

    for cand in candidates():
        row = screen_one(cand, train_cases, val_cases, hyperparams, device)
        rows.append(row)
        if cand.run_id == "original" and row.get("val_dice_mean_foreground") != "":
            original_fg = float(row["val_dice_mean_foreground"])
            print(f"  Subset original baseline FG Dice = {original_fg:.4f}")

    # Ranking
    ranked = []
    for r in rows:
        if r["run_id"] == "original":
            continue
        if r.get("val_dice_mean_foreground") == "":
            continue
        ranked.append((float(r["val_dice_mean_foreground"]), r))
    ranked.sort(key=lambda x: x[0], reverse=True)

    print("\n=== Stage C ranking (subset val FG Dice) ===")
    baseline = original_fg if original_fg is not None else float("nan")
    print(f"Subset original FG: {baseline:.4f}")
    for fg, r in ranked:
        print(f"  {r['run_id']:22s}  fg={fg:.4f}  delta={fg - baseline:+.4f}  status={r['status']}")

    recommend = None
    for fg, r in ranked:
        if fg > baseline:
            recommend = r
            break

    summary_path = OUT_ROOT / "screening_summary.csv"
    if rows:
        keys: List[str] = []
        for row in rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    selection = {
        "subset_original_fg": baseline,
        "recommend_full_run": recommend is not None,
        "recommended_run_id": recommend["run_id"] if recommend else None,
        "ranking": [
            {
                "run_id": r["run_id"],
                "val_fg": fg,
                "delta_vs_original": fg - baseline,
                "status": r["status"],
            }
            for fg, r in ranked
        ],
        "protocol": {
            "subtract": "raw intensity (not Gastro IMF min-max)",
            "energy_match_target": "1d_row IMF0 energy per slice",
            "epochs": SCREEN_EPOCHS,
            "train_cases": [c.stem for c in train_cases],
            "val_cases": [c.stem for c in val_cases],
            "seed": SEED,
        },
        "note": "Screening only. Do not treat as final evidence. No full run launched.",
    }
    (OUT_ROOT / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}")
    print(f"Wrote {OUT_ROOT / 'selection.json'}")
    if recommend:
        print(f"Worth considering for full run (after approval): {recommend['run_id']}")
    else:
        print("No matched-energy candidate beat subset original in this screen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
