#!/usr/bin/env python3
"""
Sanity-check figures from cached bemd_default_square_pad decompositions.

Figure 1: Original | BIMF0..3 | Residual  (BIMF panels: symmetric viz around 0)
Figure 2: Original | subtract BIMF0..3 | subtract BIMF0+1  (preprocessing scaling)

Visualization scaling is for display only — training inputs are unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cine_4ch.bemd_cache import BEMD_CACHE_ROOT, is_valid_cache_entry, load_decomposition
from cine_4ch.bemd_dataset import default_bemd_ablation_specs
from cine_4ch.config import OUTPUTS_DIR
from cine_4ch.dataset import load_split_cases
from cine_4ch.io import choose_representative_frame, discover_cases, load_pair
from src.preprocessing.bemd_square_pad import enhance_from_bemd_decomp
from src.preprocessing.emd_enhancement import safe_minmax_normalize

FIG_ROOT = OUTPUTS_DIR / "bemd_ablation" / "figures"


def _sym_imshow(ax, arr: np.ndarray, title: str) -> None:
    a = arr.astype(np.float64)
    lim = float(np.percentile(np.abs(a), 99.5)) or 1.0
    ax.imshow(a, cmap="coolwarm", vmin=-lim, vmax=lim)
    ax.set_title(title)
    ax.axis("off")


def _gray01(ax, arr: np.ndarray, title: str) -> None:
    a = safe_minmax_normalize(arr, clip=True)
    ax.imshow(a, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")


def make_figures_for_frame(
    case_stem: str,
    frame_idx: int,
    out_dir: Path,
    cache_root: Path,
) -> None:
    decomp = load_decomposition(case_stem, frame_idx, cache_root=cache_root)
    n = decomp.n_bimf
    # Figure 1 — components
    ncols = 2 + min(n, 4)  # original + up to 4 BIMFs + residual
    fig1, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.4))
    _gray01(axes[0], decomp.original, "Original\n(display min-max)")
    for i in range(min(n, 4)):
        _sym_imshow(
            axes[1 + i],
            decomp.bimfs[i],
            f"BIMF {i}\n(symmetric viz; not train scale)",
        )
    _sym_imshow(axes[-1], decomp.residual, "Residual\n(symmetric viz)")
    fig1.suptitle(
        f"{case_stem} frame {frame_idx:03d} — BEMD components "
        f"(n_bimf={n}; BIMF0 typically finest / highest spatial frequency)"
    )
    fig1.tight_layout()
    out1 = out_dir / f"{case_stem}_frame{frame_idx:03d}_components.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # Figure 2 — subtract conditions (actual preprocessing path)
    specs = default_bemd_ablation_specs()
    fig2, axes2 = plt.subplots(1, len(specs), figsize=(3.0 * len(specs), 3.4))
    for ax, spec in zip(axes2, specs):
        img = enhance_from_bemd_decomp(
            decomp,
            mode=spec.mode,
            bimf_indices=spec.bimf_indices,
            normalize_bimfs=spec.normalize_bimfs,
            clip_output=spec.clip_output,
        )
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(spec.run_id.replace("subtract_", "−"), fontsize=9)
        ax.axis("off")
    fig2.suptitle(
        f"{case_stem} frame {frame_idx:03d} — Gastro-style subtract inputs "
        "(per-BIMF min-max then finalize; same as training)"
    )
    fig2.tight_layout()
    out2 = out_dir / f"{case_stem}_frame{frame_idx:03d}_subtract_conditions.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote {out1}")
    print(f"Wrote {out2}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BEMD ablation sanity figures.")
    p.add_argument("--cache-root", type=Path, default=BEMD_CACHE_ROOT)
    p.add_argument("--output-dir", type=Path, default=FIG_ROOT)
    p.add_argument("--splits-csv", type=Path, default=OUTPUTS_DIR / "splits_4ch.csv")
    p.add_argument("--num-cases", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--case-stem", type=str, default=None)
    p.add_argument("--frame-idx", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.case_stem is not None:
        fidx = 0 if args.frame_idx is None else args.frame_idx
        if not is_valid_cache_entry(args.case_stem, fidx, cache_root=args.cache_root):
            print(f"No valid cache for {args.case_stem} frame {fidx}", file=sys.stderr)
            return 1
        make_figures_for_frame(args.case_stem, fidx, args.output_dir, args.cache_root)
        return 0

    if args.splits_csv.exists():
        cases = load_split_cases(args.splits_csv, "val")
    else:
        cases, _, _ = discover_cases()

    rng = np.random.default_rng(args.seed)
    picks = list(cases)
    rng.shuffle(picks)
    made = 0
    for case in picks:
        _, label, _ = load_pair(case)
        fidx = choose_representative_frame(label) if args.frame_idx is None else args.frame_idx
        if not is_valid_cache_entry(case.stem, fidx, cache_root=args.cache_root, require_n_bimf=4):
            # try frame 0
            fidx = 0
            if not is_valid_cache_entry(case.stem, fidx, cache_root=args.cache_root, require_n_bimf=4):
                continue
        make_figures_for_frame(case.stem, fidx, args.output_dir, args.cache_root)
        made += 1
        if made >= args.num_cases:
            break
    if made == 0:
        print("No valid cached frames found; run preprocess_bemd.py first.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
