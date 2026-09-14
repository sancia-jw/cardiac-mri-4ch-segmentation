#!/usr/bin/env python3
"""
Visualize the full EMD decomposition for one representative 4CH MRI frame.

Uses existing project EMD code only (no training).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import emd

from cine_4ch.config import OUTPUTS_DIR
from cine_4ch.dataset import load_split_cases
from cine_4ch.io import extract_frame, load_pair, normalize_image_slice
from src.preprocessing.emd_enhancement import as_grayscale_slice, safe_minmax_normalize

OUT_DIR = OUTPUTS_DIR / "emd_ablation" / "mentor_figures"
CASE_STEM = "CINE_4CH_009"
FRAME_IDX = 83
SIFT_THRESH = 1e-8


def decompose_raw(image_2d: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Run ``emd.sift.sift`` and split oscillatory IMFs from the final residue.

    The ``emd`` library returns all columns from SIFT; the last column is the
    monotonic/low-frequency residue left after IMF extraction (not an oscillatory IMF).
    """
    slice_2d = as_grayscale_slice(image_2d)
    signal_1d = slice_2d.reshape(-1)
    all_imfs = emd.sift.sift(signal_1d, sift_thresh=SIFT_THRESH)
    n_cols = int(all_imfs.shape[1])
    if n_cols < 2:
        raise RuntimeError(f"Expected >=2 SIFT components, got {n_cols}")

    components = [
        all_imfs[:, i].reshape(slice_2d.shape).astype(np.float32) for i in range(n_cols)
    ]
    oscillatory = components[:-1]
    residue = components[-1]
    return slice_2d, oscillatory, residue


def scale_per_panel_symmetric(values: List[np.ndarray]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for v in values:
        peak = float(np.max(np.abs(v)))
        if peak <= 1e-12:
            out.append(np.zeros_like(v, dtype=np.float32))
        else:
            out.append(np.clip(v / peak, -1.0, 1.0).astype(np.float32))
    return out


def save_decomposition_figure(
    panels: List[Tuple[str, np.ndarray]],
    *,
    case_stem: str,
    frame_idx: int,
    scaling_note: str,
    cmap: str,
    vmin,
    vmax,
    path: Path,
) -> None:
    n = len(panels)
    ncols = min(n, 6)
    nrows = math.ceil(n / ncols)
    fig_w = min(ncols * 2.8, 18)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, nrows * 2.9))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, (title, img) in zip(axes_flat, panels):
        if cmap == "gray":
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10, pad=6)
        ax.axis("off")

    for ax in axes_flat[len(panels) :]:
        ax.axis("off")

    n_imf = sum(1 for t, _ in panels if t.startswith("IMF "))
    fig.suptitle(
        f"{case_stem}  ·  frame {frame_idx}  ·  {n_imf} oscillatory IMF(s) + residual/trend\n"
        f"Scaling: {scaling_note}",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_labeled_panels(
    panels: List[Tuple[str, np.ndarray]],
    *,
    out_dir: Path,
    cmap: str,
    vmin,
    vmax,
    filename_prefix: str = "",
) -> List[str]:
    """Save one labeled PNG per panel."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    slug_map = {
        "Original": "00_original",
        "Residual/Trend": "09_residual_trend",
    }

    for title, img in panels:
        if title in slug_map:
            slug = slug_map[title]
        elif title.startswith("IMF "):
            idx = int(title.split()[1])
            slug = f"{idx + 1:02d}_imf_{idx}"
        else:
            slug = title.lower().replace(" ", "_").replace("/", "_")

        fig, ax = plt.subplots(figsize=(4, 4.3))
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12, pad=8)
        ax.axis("off")
        fig.tight_layout()

        fname = f"{filename_prefix}{slug}.png" if filename_prefix else f"{slug}.png"
        path = out_dir / fname
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(str(path))

    return saved


def save_labeled_single(
    title: str,
    img: np.ndarray,
    path: Path,
    *,
    cmap: str = "gray",
    vmin=None,
    vmax=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4.3))
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12, pad=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_reconstruction_figures(
    original: np.ndarray,
    oscillatory: List[np.ndarray],
    residue: np.ndarray,
    *,
    case_stem: str,
    frame_idx: int,
    panels_dir: Path,
) -> dict:
    """Sum all SIFT components and compare to the original slice."""
    reconstructed = np.sum(
        np.stack(oscillatory + [residue], axis=0), axis=0, dtype=np.float64
    ).astype(np.float32)
    difference = (original.astype(np.float64) - reconstructed.astype(np.float64)).astype(np.float32)

    orig_disp = normalize_image_slice(original)
    vmin = float(orig_disp.min())
    vmax = float(orig_disp.max())
    recon_disp = safe_minmax_normalize(reconstructed, clip=True)

    max_abs = float(np.max(np.abs(difference)))
    mae = float(np.mean(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference**2)))

    # Only amplify the difference panel if error is visually meaningful.
    if max_abs <= 1e-6:
        diff_disp = np.zeros_like(difference, dtype=np.float32)
        diff_title = f"Difference (Original − Sum)\nmax |Δ|={max_abs:.2e} (≈0)"
        diff_cmap = "gray"
        diff_vmin, diff_vmax = 0.0, 1.0
    else:
        diff_disp = np.clip(difference / max_abs, -1.0, 1.0).astype(np.float32)
        diff_title = f"Difference (Original − Sum)\nmax |Δ|={max_abs:.4g}"
        diff_cmap = "RdBu_r"
        diff_vmin, diff_vmax = -1.0, 1.0

    save_labeled_single(
        "Sum of all IMFs + Residual",
        recon_disp,
        panels_dir / "10_sum_all_imfs_and_residual.png",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )
    save_labeled_single(
        diff_title,
        diff_disp,
        panels_dir / "11_difference_original_minus_sum.png",
        cmap=diff_cmap,
        vmin=diff_vmin,
        vmax=diff_vmax,
    )

    combined_path = OUT_DIR / "emd_reconstruction_comparison.png"
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5))
    panels = [
        ("Original", orig_disp, "gray", vmin, vmax),
        ("Sum of all IMFs + Residual", recon_disp, "gray", vmin, vmax),
        (diff_title.replace("\n", " — "), diff_disp, diff_cmap, diff_vmin, diff_vmax),
    ]
    for ax, (title, img, cmap, lo, hi) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=10, pad=6)
        ax.axis("off")
    fig.suptitle(
        f"{case_stem}  ·  frame {frame_idx}\n"
        f"Reconstruction error: max |Δ|={max_abs:.4g}, MAE={mae:.4g}, RMSE={rmse:.4g}",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(combined_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "reconstructed_equals_original_within_float": bool(np.allclose(original, reconstructed, rtol=1e-5, atol=1e-3)),
        "max_abs_difference": max_abs,
        "mae": mae,
        "rmse": rmse,
        "combined_figure": str(combined_path),
        "panel_sum": str(panels_dir / "10_sum_all_imfs_and_residual.png"),
        "panel_difference": str(panels_dir / "11_difference_original_minus_sum.png"),
        "display_note": (
            "Original and sum use the same grayscale range (original min–max). "
            "Difference is original minus sum, scaled by max |difference| per panel."
        ),
    }


def main() -> int:
    cases = {c.stem: c for c in load_split_cases(OUTPUTS_DIR / "splits_4ch.csv", "test")}
    case = cases[CASE_STEM]
    image, _, _ = load_pair(case)
    raw = extract_frame(image, FRAME_IDX).astype(np.float32)

    slice_2d, oscillatory, residue = decompose_raw(raw)
    n_imf = len(oscillatory)
    has_separate_residual = True

    labels = ["Original"] + [f"IMF {i}" for i in range(n_imf)] + ["Residual/Trend"]
    raw_panels = [slice_2d] + oscillatory + [residue]

    # Version 1: per-panel min-max -> [0, 1] (grayscale, structure visible).
    disp_minmax = [normalize_image_slice(slice_2d)] + [
        safe_minmax_normalize(v, clip=True) for v in raw_panels[1:]
    ]
    panels_minmax = list(zip(labels, disp_minmax))
    path_minmax = OUT_DIR / "emd_decomposition.png"
    save_decomposition_figure(
        panels_minmax,
        case_stem=CASE_STEM,
        frame_idx=FRAME_IDX,
        scaling_note="per-panel min–max to [0, 1] (Original normalized; each IMF/residual scaled independently)",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        path=path_minmax,
    )

    # Version 2: per-panel symmetric around zero (diverging colormap).
    disp_sym = scale_per_panel_symmetric(raw_panels)
    panels_sym = list(zip(labels, disp_sym))
    path_sym = OUT_DIR / "emd_decomposition_symmetric.png"
    save_decomposition_figure(
        panels_sym,
        case_stem=CASE_STEM,
        frame_idx=FRAME_IDX,
        scaling_note="per-panel symmetric: each panel divided by its max |value| (range shown as [-1, 1])",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        path=path_sym,
    )

    panels_dir = OUT_DIR / "emd_decomposition_panels"
    panels_sym_dir = OUT_DIR / "emd_decomposition_panels_symmetric"
    saved_panels = save_labeled_panels(
        panels_minmax,
        out_dir=panels_dir,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    saved_panels_sym = save_labeled_panels(
        panels_sym,
        out_dir=panels_sym_dir,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
    )

    recon_meta = save_reconstruction_figures(
        slice_2d,
        oscillatory,
        residue,
        case_stem=CASE_STEM,
        frame_idx=FRAME_IDX,
        panels_dir=panels_dir,
    )

    meta = {
        "case": CASE_STEM,
        "frame_index": FRAME_IDX,
        "split": "test",
        "n_sift_columns": n_imf + 1,
        "n_oscillatory_imfs": n_imf,
        "has_separate_residual": has_separate_residual,
        "residual_index_in_sift_output": n_imf,
        "index_guide": {
            "IMF 0": "highest-frequency oscillatory component",
            f"IMF {n_imf - 1}": "lowest-frequency oscillatory component",
            "Residual/Trend": f"SIFT column {n_imf} (monotonic remainder; project code also refers to this as IMF[-1])",
        },
        "negative_index_mapping": {
            "IMF[-1]": "Residual/Trend (SIFT column 8; included in project imf list as last index)",
            "IMF[-2]": f"IMF {n_imf - 1}",
            "IMF[-3]": f"IMF {n_imf - 2}",
        },
        "figures": {
            "per_panel_minmax": str(path_minmax),
            "per_panel_symmetric": str(path_sym),
            "panel_folder_grayscale": str(panels_dir),
            "panel_folder_symmetric": str(panels_sym_dir),
        },
        "panel_files": {
            "grayscale": saved_panels,
            "symmetric": saved_panels_sym,
        },
        "reconstruction": recon_meta,
        "scaling": {
            "emd_decomposition.png": "per-panel min–max to [0, 1]",
            "emd_decomposition_symmetric.png": "per-panel max-abs normalization to [-1, 1]",
        },
        "sift_thresh": SIFT_THRESH,
        "note": "Decomposition via emd.sift.sift on flattened slice; no training.",
    }
    meta_path = OUT_DIR / "emd_decomposition_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Case/frame: {CASE_STEM} frame {FRAME_IDX}")
    print(f"Oscillatory IMFs: {n_imf}")
    print(f"Separate residual/trend: {has_separate_residual}")
    print(f"Wrote: {path_minmax}")
    print(f"Wrote: {path_sym}")
    print(f"Wrote panel folder: {panels_dir} ({len(saved_panels)} images)")
    print(f"Wrote panel folder: {panels_sym_dir} ({len(saved_panels_sym)} images)")
    print(f"Wrote reconstruction: {recon_meta['combined_figure']}")
    print(f"Wrote: {recon_meta['panel_sum']}")
    print(f"Wrote: {recon_meta['panel_difference']}")
    print(f"Reconstruction max |diff|={recon_meta['max_abs_difference']:.6g}")
    print(f"Wrote: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
