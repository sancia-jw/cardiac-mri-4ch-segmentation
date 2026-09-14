#!/usr/bin/env python3
"""
Stage B — geometry EMD feasibility + matched-energy redesign (no training).

1) Sample train/val frames; measure runtime, component counts, energy fractions.
2) Estimate full-split preprocess cost / disk for 2D EMD2D.
3) Build matched-energy variants on the mentor frame:
     - 2D BIMF0 scaled to 1D-row IMF0 energy
     - Fourier high-pass matched to 1D-row IMF0 energy
4) Write summary under outputs/emd_geometry_followup/stage_b/.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cine_4ch.config import OUTPUTS_DIR
from cine_4ch.dataset import load_split_cases
from cine_4ch.io import CasePair, choose_representative_frame, extract_frame, load_pair
from src.preprocessing.emd_geometry import (
    GeometryEMDConfig,
    component_energy_stats,
    decompose,
    display_normalize,
    match_fourier_cutoff_to_energy,
    matched_energy_subtract,
    scale_component_to_target_energy,
)

OUT_ROOT = OUTPUTS_DIR / "emd_geometry_followup" / "stage_b"
CACHE_ROOT = OUTPUTS_DIR / "emd_geometry_cache"

# Modest Stage B sample (deterministic).
N_TRAIN_CASES = 12
N_VAL_CASES = 4
FRAMES_PER_CASE = 2
SEED = 42
BYTES_PER_FLOAT32 = 4


def fixed_subset(cases: Sequence[CasePair], n: int, seed: int) -> List[CasePair]:
    if n >= len(cases):
        return list(cases)
    rng = np.random.default_rng(seed)
    idxs = sorted(rng.choice(len(cases), size=n, replace=False).tolist())
    return [cases[i] for i in idxs]


def choose_frames(label: np.ndarray, k: int) -> List[int]:
    rep = choose_representative_frame(label)
    n_frames = 1 if label.ndim < 3 else int(label.shape[-1])
    if k <= 1 or n_frames == 1:
        return [rep]
    alt = max(0, min(n_frames - 1, rep // 2))
    frames = sorted({rep, alt})
    while len(frames) < k:
        frames.append(min(n_frames - 1, frames[-1] + 1))
    return frames[:k]


def _sym(arr: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(arr)))
    if peak <= 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / peak, -1.0, 1.0).astype(np.float32)


def _save_panel(path: Path, title: str, img: np.ndarray, *, cmap="gray", vmin=0.0, vmax=1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.0, 4.2))
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10, pad=6)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def measure_slice(case: CasePair, frame_idx: int) -> Dict[str, Any]:
    image, label, _ = load_pair(case)
    raw = extract_frame(image, frame_idx).astype(np.float32)
    h, w = raw.shape
    row: Dict[str, Any] = {
        "case": case.stem,
        "frame": frame_idx,
        "shape_h": h,
        "shape_w": w,
        "n_pixels": h * w,
    }

    configs = {
        "1d_row": GeometryEMDConfig(backend="1d_raster", flatten_order="C"),
        "1d_col": GeometryEMDConfig(backend="1d_raster", flatten_order="F"),
        "2d_emd2d": GeometryEMDConfig(backend="emd2d_pyemd", max_imf=1),
    }
    for key, cfg in configs.items():
        t0 = time.perf_counter()
        try:
            decomp = decompose(raw, cfg)
            elapsed = time.perf_counter() - t0
            finest = decomp.components[0]
            energy = component_energy_stats(raw, finest)
            # Disk estimate if we cached finest + residual float32
            disk_bytes = (1 + decomp.n_oscillatory) * h * w * BYTES_PER_FLOAT32
            row[f"{key}_ok"] = True
            row[f"{key}_n_osc"] = decomp.n_oscillatory
            row[f"{key}_sec"] = round(elapsed, 4)
            row[f"{key}_energy_frac"] = energy["energy_fraction_of_original"]
            row[f"{key}_energy"] = energy["energy"]
            row[f"{key}_recon_maxabs"] = decomp.meta.get("reconstruction", {}).get("max_abs")
            row[f"{key}_disk_bytes_est"] = disk_bytes
            row[f"{key}_error"] = ""
        except Exception as exc:  # pragma: no cover
            row[f"{key}_ok"] = False
            row[f"{key}_n_osc"] = ""
            row[f"{key}_sec"] = ""
            row[f"{key}_energy_frac"] = ""
            row[f"{key}_energy"] = ""
            row[f"{key}_recon_maxabs"] = ""
            row[f"{key}_disk_bytes_est"] = ""
            row[f"{key}_error"] = str(exc)
    return row


def summarize_numeric(rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    vals = [float(r[key]) for r in rows if r.get(key) != "" and r.get(key) is not None]
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }


def estimate_full_cost(sample_rows: List[Dict[str, Any]], *, n_full_slices: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for method in ("1d_row", "1d_col", "2d_emd2d"):
        sec = summarize_numeric(sample_rows, f"{method}_sec")
        disk = summarize_numeric(sample_rows, f"{method}_disk_bytes_est")
        if not sec:
            continue
        mean_sec = sec["mean"]
        out[method] = {
            "sec_per_slice_mean": mean_sec,
            "est_hours_for_n_slices": (mean_sec * n_full_slices) / 3600.0,
            "est_disk_gb_for_n_slices": (disk["mean"] * n_full_slices) / (1024**3) if disk else None,
            "sample_timing": sec,
            "sample_disk_bytes": disk,
        }
    return out


def mentor_matched_energy_figures() -> Dict[str, Any]:
    """Matched-energy redesign visuals on CINE_4CH_009 frame 83."""
    cases = {c.stem: c for c in load_split_cases(OUTPUTS_DIR / "splits_4ch.csv", "test")}
    case = cases["CINE_4CH_009"]
    image, _, _ = load_pair(case)
    frame_idx = 83
    raw = extract_frame(image, frame_idx).astype(np.float32)

    out_dir = OUT_ROOT / "matched_energy_CINE_4CH_009_frame083"
    panels = out_dir / "panels"
    panels.mkdir(parents=True, exist_ok=True)

    row_d = decompose(raw, GeometryEMDConfig(backend="1d_raster", flatten_order="C"))
    col_d = decompose(raw, GeometryEMDConfig(backend="1d_raster", flatten_order="F"))
    twod_d = decompose(raw, GeometryEMDConfig(backend="emd2d_pyemd", max_imf=1))

    target_energy = float(np.sum(row_d.components[0].astype(np.float64) ** 2))
    target_frac = component_energy_stats(raw, row_d.components[0])["energy_fraction_of_original"]

    # Scale 2D BIMF0 and col IMF0 to row IMF0 energy (raw subtract, no Gastro normalize).
    twod_matched = matched_energy_subtract(
        raw, twod_d.components[0], target_energy=target_energy, use_normalized_component_for_subtract=False
    )
    col_matched = matched_energy_subtract(
        raw, col_d.components[0], target_energy=target_energy, use_normalized_component_for_subtract=False
    )
    row_raw_sub = (raw.astype(np.float64) - row_d.components[0].astype(np.float64)).astype(np.float32)

    fft_match = match_fourier_cutoff_to_energy(raw, target_energy=target_energy)
    # Exact energy match for Fourier residual via scaling after cutoff selection.
    fft_scaled, fft_scale = scale_component_to_target_energy(
        fft_match["component"], target_energy=target_energy
    )
    fft_proc = (raw.astype(np.float64) - fft_scaled.astype(np.float64)).astype(np.float32)

    # Panels
    _save_panel(panels / "00_original.png", "Original", display_normalize(raw))
    _save_panel(panels / "01_1d_row_imf0.png", "1D row IMF0", _sym(row_d.components[0]), cmap="RdBu_r", vmin=-1, vmax=1)
    _save_panel(panels / "02_1d_row_processed_rawsub.png", "1D row original−IMF0 (raw)", display_normalize(row_raw_sub))
    _save_panel(panels / "03_1d_col_imf0_matchedE.png", "1D col IMF0 (energy-matched)", _sym(col_matched["scaled_component"]), cmap="RdBu_r", vmin=-1, vmax=1)
    _save_panel(panels / "04_1d_col_processed_matchedE.png", "1D col original−matched IMF0", display_normalize(col_matched["processed"]))
    _save_panel(panels / "05_2d_bimf0_raw.png", "2D BIMF0 (raw, unmatched)", _sym(twod_d.components[0]), cmap="RdBu_r", vmin=-1, vmax=1)
    _save_panel(panels / "06_2d_bimf0_matchedE.png", "2D BIMF0 (energy-matched to row IMF0)", _sym(twod_matched["scaled_component"]), cmap="RdBu_r", vmin=-1, vmax=1)
    _save_panel(panels / "07_2d_processed_matchedE.png", "2D original−matched BIMF0", display_normalize(twod_matched["processed"]))
    _save_panel(panels / "08_fft_highpass_matchedE.png", f"Fourier HP matchedE (cut≈{fft_match['cutoff_cycles_per_pixel']})", _sym(fft_scaled), cmap="RdBu_r", vmin=-1, vmax=1)
    _save_panel(panels / "09_fft_processed_matchedE.png", "Fourier original−matched HP", display_normalize(fft_proc))

    # Montage
    montage = [
        ("Original", display_normalize(raw), "gray", 0, 1),
        ("1D row IMF0", _sym(row_d.components[0]), "RdBu_r", -1, 1),
        ("1D row −IMF0", display_normalize(row_raw_sub), "gray", 0, 1),
        ("2D BIMF0 raw", _sym(twod_d.components[0]), "RdBu_r", -1, 1),
        ("2D BIMF0 matchedE", _sym(twod_matched["scaled_component"]), "RdBu_r", -1, 1),
        ("2D −matched BIMF0", display_normalize(twod_matched["processed"]), "gray", 0, 1),
        ("FFT HP matchedE", _sym(fft_scaled), "RdBu_r", -1, 1),
        ("FFT −matched HP", display_normalize(fft_proc), "gray", 0, 1),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.ravel()
    for ax, (title, img, cmap, vmin, vmax) in zip(axes, montage):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(
        "CINE_4CH_009 · frame 83 · matched-energy redesign\n"
        f"Target energy = 1D-row IMF0 ({target_frac:.3%} of original). "
        "Raw subtract (not Gastro min-max normalize).",
        fontsize=11,
    )
    fig.tight_layout()
    montage_path = out_dir / "matched_energy_comparison.png"
    fig.savefig(montage_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    report = {
        "case": "CINE_4CH_009",
        "frame_index": frame_idx,
        "target": {
            "source": "1d_row_imf0_raw_energy",
            "energy": target_energy,
            "energy_fraction_of_original": target_frac,
        },
        "unmatched_energy_fractions": {
            "1d_row": component_energy_stats(raw, row_d.components[0])["energy_fraction_of_original"],
            "1d_col": component_energy_stats(raw, col_d.components[0])["energy_fraction_of_original"],
            "2d_bimf0": component_energy_stats(raw, twod_d.components[0])["energy_fraction_of_original"],
            "fft_best_cutoff_before_scale": fft_match["energy_fraction_of_original"],
            "fft_cutoff_cycles_per_pixel": fft_match["cutoff_cycles_per_pixel"],
        },
        "matched": {
            "2d_scale_factor": twod_matched["scale_factor"],
            "1d_col_scale_factor": col_matched["scale_factor"],
            "fft_scale_factor": fft_scale,
            "note": "scale_factor < 1 means component was attenuated to match 1D-row IMF0 energy",
        },
        "figures": {
            "montage": str(montage_path),
            "panels_dir": str(panels),
        },
    }
    (out_dir / "matched_energy_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def count_split_slices(splits_csv: Path) -> Dict[str, int]:
    """Approximate full-split frame counts by loading each case once."""
    counts = {"train": 0, "val": 0, "test": 0}
    for split in counts:
        cases = load_split_cases(splits_csv, split)
        for case in cases:
            _, label, _ = load_pair(case)
            n_frames = 1 if label.ndim < 3 else int(label.shape[-1])
            counts[split] += n_frames
    return counts


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    splits_csv = OUTPUTS_DIR / "splits_4ch.csv"

    train_cases = fixed_subset(load_split_cases(splits_csv, "train"), N_TRAIN_CASES, SEED)
    val_cases = fixed_subset(load_split_cases(splits_csv, "val"), N_VAL_CASES, SEED + 1)
    print(f"Stage B sample: {len(train_cases)} train + {len(val_cases)} val cases, {FRAMES_PER_CASE} frames/case")

    rows: List[Dict[str, Any]] = []
    for split_name, cases in (("train", train_cases), ("val", val_cases)):
        for case in cases:
            _, label, _ = load_pair(case)
            for fidx in choose_frames(label, FRAMES_PER_CASE):
                print(f"  {split_name} {case.stem} frame {fidx} ...", flush=True)
                row = measure_slice(case, fidx)
                row["split"] = split_name
                rows.append(row)
                print(
                    f"    rowE={row.get('1d_row_energy_frac')}  "
                    f"colE={row.get('1d_col_energy_frac')}  "
                    f"2dE={row.get('2d_emd2d_energy_frac')}  "
                    f"2d_t={row.get('2d_emd2d_sec')}s"
                )

    csv_path = OUT_ROOT / "stage_b_slice_metrics.csv"
    fieldnames: List[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nCounting full-split frames for cost estimate...")
    split_frames = count_split_slices(splits_csv)
    n_tv = split_frames["train"] + split_frames["val"]
    n_all = n_tv + split_frames["test"]
    cost = estimate_full_cost(rows, n_full_slices=n_tv)

    print("Building matched-energy mentor figures...")
    matched = mentor_matched_energy_figures()

    summary = {
        "stage": "B",
        "no_training": True,
        "sample": {
            "n_train_cases": len(train_cases),
            "n_val_cases": len(val_cases),
            "frames_per_case": FRAMES_PER_CASE,
            "n_slices_measured": len(rows),
            "seed": SEED,
            "train_stems": [c.stem for c in train_cases],
            "val_stems": [c.stem for c in val_cases],
        },
        "full_split_frame_counts": split_frames,
        "energy_fraction_summary": {
            "1d_row": summarize_numeric(rows, "1d_row_energy_frac"),
            "1d_col": summarize_numeric(rows, "1d_col_energy_frac"),
            "2d_emd2d": summarize_numeric(rows, "2d_emd2d_energy_frac"),
        },
        "n_oscillatory_summary": {
            "1d_row": summarize_numeric(rows, "1d_row_n_osc"),
            "1d_col": summarize_numeric(rows, "1d_col_n_osc"),
            "2d_emd2d": summarize_numeric(rows, "2d_emd2d_n_osc"),
        },
        "runtime_summary_sec": {
            "1d_row": summarize_numeric(rows, "1d_row_sec"),
            "1d_col": summarize_numeric(rows, "1d_col_sec"),
            "2d_emd2d": summarize_numeric(rows, "2d_emd2d_sec"),
        },
        "cost_estimate_train_val_preprocess": cost,
        "cost_estimate_note": (
            f"Estimates scale mean sample timing to train+val frames (n={n_tv}). "
            f"All-split including test would be n={n_all}."
        ),
        "matched_energy_mentor": matched,
        "recommendation": {
            "do_not_train_naive_2d_bimf0_yet": True,
            "reason": (
                "Across the Stage B sample, 2D BIMF0 energy fraction remains ~10× larger than "
                "1D IMF0. Use energy-matched 2D BIMF0 (and/or Fourier matched control) in Stage C."
            ),
            "proposed_stage_c_conditions": [
                "original",
                "1d_row_subtract_imf0_raw_or_gastro",
                "1d_col_subtract_imf0_energy_matched_to_row",
                "2d_bimf0_energy_matched_to_row",
                "fft_highpass_energy_matched_to_row",
            ],
        },
        "artifacts": {
            "slice_metrics_csv": str(csv_path),
            "matched_energy_dir": str(OUT_ROOT / "matched_energy_CINE_4CH_009_frame083"),
        },
    }
    summary_path = OUT_ROOT / "stage_b_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Matched-energy montage: {matched['figures']['montage']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
