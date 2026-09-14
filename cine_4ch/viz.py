"""Visualization helpers for 4CH_TR."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from cine_4ch.config import LABEL_COLORS, LABEL_NAMES
from cine_4ch.io import normalize_image_slice


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


def save_case_figure(
    image_slice: np.ndarray,
    label_slice: np.ndarray,
    stem: str,
    frame_idx: int,
    save_path: Path,
    unique_labels: Optional[List[int]] = None,
) -> None:
    """Save raw image, mask, and overlay side by side."""
    base_gray = normalize_image_slice(image_slice)
    mask_rgb = label_to_rgb(label_slice.astype(np.int64))
    overlay = overlay_label(base_gray, label_slice.astype(np.int64))

    if unique_labels is None:
        unique_labels = sorted(int(v) for v in np.unique(label_slice))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"4CH_TR — {stem} (frame {frame_idx})")

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

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
