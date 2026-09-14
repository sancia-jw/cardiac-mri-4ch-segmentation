#!/usr/bin/env python3
"""
Visualize EMD enhancement on cardiac MRI slices with segmentation context.

Loads one CINE 4CH case, selects frames with visible anatomy, and saves
comparison figures for multiple IMF settings to outputs/emd_visualization/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cine_4ch.config import DATA_ROOT, LABEL_COLORS, LABEL_NAMES, OUTPUTS_DIR
from cine_4ch.io import discover_cases, extract_frame, load_pair
from cine_4ch.viz import label_to_rgb, overlay_label
from src.preprocessing.emd_enhancement import (
    compute_imfs,
    reconstruct_from_imfs,
    safe_minmax_normalize,
    subtract_imfs_from_image,
)

# IMF index sets to compare (boundaries vs noise).
IMF_SETTINGS: List[Tuple[str, List[int]]] = [
    ("imf_0", [0]),
    ("imf_1", [1]),
    ("imf_0_1", [0, 1]),
    ("imf_1_2", [1, 2]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize EMD enhancement on CINE 4CH cardiac MRI slices."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "emd_visualization",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Case stem or ID (e.g. 001 or CINE_4CH_001). Default: first paired case.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=3,
        help="Number of representative temporal frames to visualize per IMF setting.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_case(cases, case_arg: str | None):
    if case_arg is None:
        return cases[0]
    matches = [c for c in cases if c.stem == case_arg or c.case_id == case_arg or c.stem.endswith(case_arg)]
    if not matches:
        raise FileNotFoundError(f"No case matching '{case_arg}'.")
    return sorted(matches, key=lambda c: c.stem)[0]


def pick_representative_frames(label: np.ndarray, num_frames: int, seed: int) -> List[int]:
    """
    Choose temporal frames with substantial foreground anatomy.

    Picks the frame with the most labels, then spreads additional picks across
    other high-foreground frames so systole/diastole differences may appear.
    """
    if label.ndim < 3:
        return [0]

    n_frames = label.shape[-1]
    fg_per_frame = (label > 0).reshape(-1, n_frames).sum(axis=0)
    ranked = np.argsort(fg_per_frame)[::-1]

    if fg_per_frame[ranked[0]] == 0:
        mid = n_frames // 2
        return [mid]

    num_frames = min(num_frames, n_frames)
    chosen = [int(ranked[0])]

    if num_frames > 1:
        # Candidate pool: top half of frames by foreground count.
        pool = [int(i) for i in ranked if fg_per_frame[i] > 0]
        pool = pool[: max(len(pool) // 2, 1)]

        rng = np.random.default_rng(seed)
        while len(chosen) < num_frames and len(pool) > len(chosen):
            candidate = int(rng.choice(pool))
            if candidate not in chosen:
                chosen.append(candidate)

        # Evenly spaced fallback if pool is small.
        if len(chosen) < num_frames:
            step = max(1, n_frames // num_frames)
            for f in range(0, n_frames, step):
                if f not in chosen:
                    chosen.append(f)
                if len(chosen) >= num_frames:
                    break

    return sorted(chosen[:num_frames])


def _format_indices(indices: Sequence[int]) -> str:
    return "-".join(str(i) for i in indices)


def save_comparison_figure(
    original_disp: np.ndarray,
    imf_only_disp: np.ndarray,
    subtract_disp: np.ndarray,
    label_slice: np.ndarray,
    stem: str,
    frame_idx: int,
    imf_label: str,
    imf_indices: Sequence[int],
    save_path: Path,
) -> None:
    """Save comparison figure: original, IMF only, subtract, mask, and overlay."""
    mask_rgb = label_to_rgb(label_slice.astype(np.int64))
    overlay = overlay_label(original_disp, label_slice.astype(np.int64))

    unique_labels = sorted(int(v) for v in np.unique(label_slice))
    patches = [
        mpatches.Patch(
            color=tuple(c / 255.0 for c in LABEL_COLORS[lid]),
            label=f"{lid}: {LABEL_NAMES[lid]}",
        )
        for lid in unique_labels
        if lid in LABEL_NAMES
    ]

    idx_str = _format_indices(imf_indices)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"{stem} — frame {frame_idx} — IMF [{idx_str}] ({imf_label})",
        fontsize=13,
    )

    axes[0, 0].imshow(original_disp, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("1. Original MRI")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(imf_only_disp, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title(f"2. IMF only {list(imf_indices)}")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(subtract_disp, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title(f"3. Original − IMF {list(imf_indices)}")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(mask_rgb)
    axes[1, 0].set_title("4a. Segmentation mask")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("4b. Original + mask overlay")
    axes[1, 1].axis("off")

    diff = np.abs(subtract_disp - original_disp)
    axes[1, 2].imshow(diff, cmap="magma", vmin=0, vmax=max(float(diff.max()), 1e-6))
    axes[1, 2].set_title("|subtract − original| (change map)")
    axes[1, 2].axis("off")

    if patches:
        fig.legend(handles=patches, loc="lower center", ncol=min(len(patches), 4), frameon=False)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cases, _, _ = discover_cases(args.data_root)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    case = resolve_case(cases, args.case)
    image, label, _ = load_pair(case)
    frame_indices = pick_representative_frames(label, args.num_frames, args.seed)

    print(f"Case: {case.stem}")
    print(f"Volume shape: image {image.shape}, label {label.shape}")
    print(f"Frames: {frame_indices}")
    print(f"Output: {args.output_dir}")

    case_dir = args.output_dir / case.stem
    case_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in frame_indices:
        image_slice = extract_frame(image, frame_idx)
        label_slice = extract_frame(label, frame_idx).astype(np.int64)

        # Display-normalized original (masks unchanged).
        original_disp = safe_minmax_normalize(image_slice, clip=True)

        # Compute IMFs once per frame; reuse across IMF settings.
        imfs = compute_imfs(image_slice, normalize_imfs=True)

        for imf_label, imf_indices in IMF_SETTINGS:
            imf_only = reconstruct_from_imfs(
                image_slice,
                imf_indices,
                imfs=imfs,
                normalize_imfs=True,
            )
            subtracted = subtract_imfs_from_image(
                image_slice,
                imf_indices,
                imfs=imfs,
                normalize_imfs=True,
            )

            imf_only_disp = safe_minmax_normalize(imf_only, clip=True)
            subtract_disp = safe_minmax_normalize(subtracted, clip=True)

            out_name = f"{imf_label}_frame{frame_idx:03d}.png"
            save_path = case_dir / out_name
            save_comparison_figure(
                original_disp,
                imf_only_disp,
                subtract_disp,
                label_slice,
                case.stem,
                frame_idx,
                imf_label,
                imf_indices,
                save_path,
            )
            print(f"  saved {save_path.relative_to(args.output_dir)}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
