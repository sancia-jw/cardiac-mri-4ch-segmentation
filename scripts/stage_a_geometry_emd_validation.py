#!/usr/bin/env python3
"""
Stage A — geometry EMD validation (no training).

Compares on representative native-resolution 4CH frames:
  - original
  - 1D row-major IMF0 / original−IMF0
  - 1D column-major IMF0 / original−IMF0
  - PyEMD EMD2D BIMF0 / original−BIMF0
  - residuals

Writes figures + metrics under outputs/emd_geometry_followup/stage_a/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cine_4ch.config import OUTPUTS_DIR
from cine_4ch.dataset import load_split_cases
from cine_4ch.io import choose_representative_frame, extract_frame, load_pair
from src.preprocessing.emd_enhancement import safe_minmax_normalize
from src.preprocessing.emd_geometry import (
    GeometryEMDConfig,
    component_energy_stats,
    decompose,
    display_normalize,
    subtract_finest_component,
)

OUT_ROOT = OUTPUTS_DIR / "emd_geometry_followup" / "stage_a"
# Primary mentor frame + two additional test cases for robustness of Stage A.
SLICE_SPECS = [
    ("CINE_4CH_009", 83),
    ("CINE_4CH_014", None),  # None -> representative frame
    ("CINE_4CH_020", None),
]


def _save_panel(path: Path, title: str, img: np.ndarray, *, cmap: str = "gray", vmin=0.0, vmax=1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 4.4))
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11, pad=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _sym_display(arr: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(arr)))
    if peak <= 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / peak, -1.0, 1.0).astype(np.float32)


def analyze_slice(case_stem: str, frame_idx: int, image_2d: np.ndarray) -> Dict[str, Any]:
    case_dir = OUT_ROOT / f"{case_stem}_frame{frame_idx:03d}"
    panels_dir = case_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    original = image_2d.astype(np.float32)
    orig_disp = display_normalize(original)
    _save_panel(panels_dir / "00_original.png", "Original", orig_disp)

    methods = [
        (
            "1d_row",
            GeometryEMDConfig(backend="1d_raster", flatten_order="C", sift_thresh=1e-8),
            "1D row-major IMF 0",
            "1D row-major original − IMF 0",
            "1D row-major residual",
        ),
        (
            "1d_col",
            GeometryEMDConfig(backend="1d_raster", flatten_order="F", sift_thresh=1e-8),
            "1D column-major IMF 0",
            "1D column-major original − IMF 0",
            "1D column-major residual",
        ),
        (
            "2d_emd2d",
            GeometryEMDConfig(backend="emd2d_pyemd", max_imf=1),
            "2D EMD2D BIMF 0",
            "2D EMD2D original − BIMF 0",
            "2D EMD2D residual",
        ),
    ]

    method_rows: Dict[str, Any] = {}
    montage_panels: List[Tuple[str, np.ndarray, str, float, float]] = [
        ("Original", orig_disp, "gray", 0.0, 1.0)
    ]

    for key, cfg, fine_title, proc_title, resid_title in methods:
        t0 = time.perf_counter()
        processed, decomp, finest_for_sub = subtract_finest_component(
            original,
            cfg,
            use_normalized_component_for_subtract=True,  # Gastro training convention
        )
        wall = time.perf_counter() - t0

        finest_raw = decomp.components[0]
        energy = component_energy_stats(original, finest_raw)
        recon = decomp.meta.get("reconstruction", {})

        # Displays
        fine_sym = _sym_display(finest_raw)
        proc_disp = display_normalize(processed)
        resid_sym = _sym_display(decomp.residual)

        _save_panel(
            panels_dir / f"{key}_finest_component.png",
            fine_title,
            fine_sym,
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
        )
        _save_panel(panels_dir / f"{key}_processed.png", proc_title, proc_disp)
        _save_panel(
            panels_dir / f"{key}_residual.png",
            resid_title,
            resid_sym,
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
        )

        montage_panels.extend(
            [
                (fine_title, fine_sym, "RdBu_r", -1.0, 1.0),
                (proc_title, proc_disp, "gray", 0.0, 1.0),
            ]
        )

        method_rows[key] = {
            "backend": decomp.backend,
            "flatten_order": decomp.flatten_order,
            "n_oscillatory_components": decomp.n_oscillatory,
            "elapsed_sec_decompose_plus_subtract": round(wall, 4),
            "elapsed_sec_decompose_only": round(decomp.elapsed_sec, 4),
            "reconstruction": recon,
            "finest_component_energy": energy,
            "gastro_subtract_uses_normalized_component": True,
            "meta": {k: v for k, v in decomp.meta.items() if k != "reconstruction"},
        }

    # Comparison montage: Original | row IMF0 | row proc | col IMF0 | col proc | 2D BIMF0 | 2D proc
    fig, axes = plt.subplots(1, len(montage_panels), figsize=(3.1 * len(montage_panels), 3.6))
    if len(montage_panels) == 1:
        axes = [axes]
    for ax, (title, img, cmap, vmin, vmax) in zip(axes, montage_panels):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.suptitle(
        f"{case_stem} · frame {frame_idx} · Stage A geometry comparison\n"
        "Finest components: per-panel symmetric [-1,1]; processed: per-panel min–max [0,1]",
        fontsize=10,
        y=1.05,
    )
    fig.tight_layout()
    montage_path = case_dir / "geometry_comparison.png"
    fig.savefig(montage_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Residuals comparison (recompute raw decompositions for residual panels).
    resid_panels = []
    for key, cfg, _, _, resid_title in methods:
        decomp = decompose(
            original,
            GeometryEMDConfig(
                backend=cfg.backend,
                flatten_order=cfg.flatten_order,
                sift_thresh=cfg.sift_thresh,
                max_imf=cfg.max_imf,
                normalize_components=False,
            ),
        )
        resid_panels.append((resid_title, _sym_display(decomp.residual)))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, (title, img) in zip(axes, resid_panels):
        ax.imshow(img, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{case_stem} · frame {frame_idx} · residuals/trends", fontsize=10)
    fig.tight_layout()
    resid_path = case_dir / "residuals_comparison.png"
    fig.savefig(resid_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    report = {
        "case": case_stem,
        "frame_index": frame_idx,
        "image_shape_hw": list(original.shape),
        "intensity_range": [float(original.min()), float(original.max())],
        "methods": method_rows,
        "figures": {
            "montage": str(montage_path),
            "residuals": str(resid_path),
            "panels_dir": str(panels_dir),
        },
        "notes": [
            "EMD applied at native frame resolution (same as training preprocess before resize).",
            "Subtract uses Gastro convention: min-max normalize finest component, then original_raw - component.",
            "Energy stats use the raw (unnormalized) finest component.",
            "2D backend is PyEMD EMD2D (experimental; Apache-2.0 via EMD-signal).",
        ],
    }
    (case_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    test_cases = {c.stem: c for c in load_split_cases(OUTPUTS_DIR / "splits_4ch.csv", "test")}
    train_cases = {c.stem: c for c in load_split_cases(OUTPUTS_DIR / "splits_4ch.csv", "train")}
    all_cases = {**train_cases, **test_cases}

    summaries: List[Dict[str, Any]] = []
    for stem, frame in SLICE_SPECS:
        if stem not in all_cases:
            print(f"SKIP missing case {stem}")
            continue
        case = all_cases[stem]
        image, label, _ = load_pair(case)
        fidx = choose_representative_frame(label) if frame is None else int(frame)
        slice_2d = extract_frame(image, fidx).astype(np.float32)
        print(f"\n=== {stem} frame {fidx} shape={slice_2d.shape} ===")
        report = analyze_slice(stem, fidx, slice_2d)
        summaries.append(report)
        for key, row in report["methods"].items():
            e = row["finest_component_energy"]["energy_fraction_of_original"]
            r = row["reconstruction"]
            print(
                f"  {key:10s} n_osc={row['n_oscillatory_components']}  "
                f"t={row['elapsed_sec_decompose_only']:.3f}s  "
                f"energy_frac={e:.4f}  "
                f"recon_maxabs={r.get('max_abs', float('nan')):.3e}"
            )

    summary_path = OUT_ROOT / "stage_a_summary.json"
    summary = {
        "stage": "A",
        "n_slices": len(summaries),
        "slices": summaries,
        "library_choice": {
            "2d_backend": "PyEMD.EMD2d.EMD2D",
            "package": "EMD-signal",
            "license": "Apache-2.0",
            "caveat": "Author marks EMD2D experimental",
        },
        "no_training": True,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
