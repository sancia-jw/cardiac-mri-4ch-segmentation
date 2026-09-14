#!/usr/bin/env python3
"""Explore a sample from CMR-MULTI CINE_MULTI/4CH_TR."""

from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np

NII_EXTENSIONS = (".nii.gz", ".nii")
VIEW = "4CH_TR"

LABEL_NAMES: Dict[int, str] = {
    0: "Background (no structure)",
    1: "Left Ventricle Cavity (blood pool)",
    2: "Left Ventricle Myocardium",
    3: "Right Ventricle Cavity",
    4: "Right Atrium",
    5: "Left Atrium",
}

LABEL_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 102, 255),
    4: (255, 215, 0),
    5: (255, 0, 255),
}

LFS_HINT = (
    "No paired NIfTI files found. If you cloned CMR-MULTI from HuggingFace, "
    "request access and run from the CMR-MULTI directory:\n"
    "  git lfs install\n"
    "  git lfs pull"
)


def _strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def _collect_nii_paths(directory: Path) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for extension in NII_EXTENSIONS:
        for path in sorted(directory.glob(f"*{extension}")):
            paths[_strip_nii_suffix(path.name)] = path
    return paths


def find_sample(data_root: Path, case: str | None = None) -> Tuple[str, Path, Path]:
    image_dir = data_root / VIEW / "image"
    anno_dir = data_root / VIEW / "anno"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}\n{LFS_HINT}")
    if not anno_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {anno_dir}\n{LFS_HINT}")

    image_map = _collect_nii_paths(image_dir)
    anno_map = _collect_nii_paths(anno_dir)
    shared = sorted(set(image_map) & set(anno_map))

    if not shared:
        raise FileNotFoundError(
            f"No paired NIfTI files in {data_root / VIEW}.\n{LFS_HINT}"
        )

    if case is not None:
        matches = [name for name in shared if name.endswith(case) or name == case]
        if not matches:
            raise FileNotFoundError(
                f"No paired sample matching case '{case}'. "
                f"Available stems (first 10): {shared[:10]}"
            )
        stem = sorted(matches)[0]
    else:
        stem = shared[0]

    return stem, image_map[stem], anno_map[stem]


def load_volume(path: Path) -> Tuple[np.ndarray, Tuple[float, ...]]:
    nii = nib.load(str(path))
    data = np.asarray(nii.get_fdata(dtype=np.float32))
    # Squeeze singleton dimensions (e.g. 4D with T=1).
    data = np.squeeze(data)
    spacing = tuple(float(v) for v in nii.header.get_zooms()[: data.ndim])
    return data, spacing


def choose_frame_index(label: np.ndarray, frame: int | None = None) -> int:
    if frame is not None:
        if frame < 0 or frame >= label.shape[-1]:
            raise ValueError(
                f"Frame index {frame} out of range for axis size {label.shape[-1]}"
            )
        return frame

    if label.ndim < 3:
        return 0

    foreground_per_frame = (label > 0).reshape(-1, label.shape[-1]).sum(axis=0)
    if foreground_per_frame.max() > 0:
        return int(np.argmax(foreground_per_frame))
    return label.shape[-1] // 2


def extract_frame(volume: np.ndarray, frame_idx: int) -> np.ndarray:
    if volume.ndim == 2:
        return volume
    if volume.ndim == 3:
        return volume[..., frame_idx]
    raise ValueError(f"Expected 2D or 3D volume after squeeze, got shape {volume.shape}")


def normalize_to_float01(slice_2d: np.ndarray) -> np.ndarray:
    vmin = float(slice_2d.min())
    vmax = float(slice_2d.max())
    if vmax <= vmin:
        return np.zeros_like(slice_2d, dtype=np.float32)
    return ((slice_2d - vmin) / (vmax - vmin)).astype(np.float32)


def label_to_rgb(mask_2d: np.ndarray) -> np.ndarray:
    h, w = mask_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in LABEL_COLORS.items():
        rgb[mask_2d == class_id] = color
    return rgb


def overlay_label(base_gray: np.ndarray, label_2d: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = np.clip(base_gray, 0.0, 1.0)
    base_rgb = np.stack([base, base, base], axis=-1)
    label_rgb = label_to_rgb(label_2d).astype(np.float32) / 255.0
    fg_mask = (label_2d > 0)[..., None].astype(np.float32)
    out = base_rgb * (1.0 - fg_mask * alpha) + label_rgb * (fg_mask * alpha)
    return np.clip(out, 0.0, 1.0)


def print_metadata(
    stem: str,
    image_path: Path,
    anno_path: Path,
    image: np.ndarray,
    label: np.ndarray,
    image_spacing: Tuple[float, ...],
    label_spacing: Tuple[float, ...],
) -> List[int]:
    unique_labels = sorted(int(v) for v in np.unique(label))

    print(f"Sample: {stem}")
    print(f"Image path: {image_path}")
    print(f"Anno path:  {anno_path}")
    print(f"Image shape: {image.shape}")
    print(f"Label shape: {label.shape}")
    print(f"Image spacing (mm): {image_spacing}")
    print(f"Label spacing (mm): {label_spacing}")
    print(f"Unique labels: {unique_labels}")
    print()

    print("4CH_TR CINE label definitions:")
    print(f"{'ID':<4} {'Structure':<42} {'In sample':<10} {'Voxels':>12}")
    print("-" * 72)
    for label_id in sorted(LABEL_NAMES):
        present = label_id in unique_labels
        count = int((label == label_id).sum()) if present else 0
        marker = "yes" if present else "no"
        print(f"{label_id:<4} {LABEL_NAMES[label_id]:<42} {marker:<10} {count:>12,}")
    print()

    unknown = [v for v in unique_labels if v not in LABEL_NAMES]
    if unknown:
        print(f"Warning: unexpected label IDs not in 4CH_TR schema: {unknown}")
        print()

    return unique_labels


def visualize(
    image_slice: np.ndarray,
    label_slice: np.ndarray,
    stem: str,
    frame_idx: int,
    unique_labels: List[int],
    save_path: Path | None,
) -> None:
    base_gray = normalize_to_float01(image_slice)
    mask_rgb = label_to_rgb(label_slice)
    overlay = overlay_label(base_gray, label_slice)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"CMR-MULTI 4CH_TR — {stem} (frame {frame_idx})")

    axes[0].imshow(base_gray, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("CINE image")
    axes[0].axis("off")

    axes[1].imshow(mask_rgb)
    axes[1].set_title("Segmentation mask")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    patches = [
        mpatches.Patch(
            color=tuple(c / 255.0 for c in LABEL_COLORS[label_id]),
            label=f"{label_id}: {LABEL_NAMES[label_id]}",
        )
        for label_id in unique_labels
        if label_id in LABEL_NAMES
    ]
    if patches:
        fig.legend(handles=patches, loc="lower center", ncol=min(len(patches), 3), frameon=False)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore a CMR-MULTI CINE 4CH_TR sample (image + annotation)."
    )
    default_root = Path(__file__).resolve().parent / "CMR-MULTI" / "CINE_MULTI"
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_root,
        help=f"Path to CINE_MULTI directory (default: {default_root})",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Case stem to load (e.g. 001). Default: first paired sample.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Temporal frame index along the last axis. Default: frame with most labels.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the figure (e.g. output_4ch_001.png).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        stem, image_path, anno_path = find_sample(args.data_root, args.case)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    image, image_spacing = load_volume(image_path)
    label, label_spacing = load_volume(anno_path)

    if image.shape != label.shape:
        print(
            f"Warning: image shape {image.shape} != label shape {label.shape}",
            file=sys.stderr,
        )

    unique_labels = print_metadata(
        stem, image_path, anno_path, image, label, image_spacing, label_spacing
    )

    frame_idx = choose_frame_index(label, args.frame)
    image_slice = extract_frame(image, frame_idx)
    label_slice = extract_frame(label, frame_idx).astype(np.int64)

    print(f"Displaying frame {frame_idx} of {label.shape[-1] if label.ndim >= 3 else 1}")
    visualize(image_slice, label_slice, stem, frame_idx, unique_labels, args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
