#!/usr/bin/env python3
"""
Build mentor-facing EMD ablation figures from existing checkpoints only.

No training. Uses outputs/emd_ablation/*/checkpoints/best.pt and the fixed split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cine_4ch.ablation import default_ablation_runs
from cine_4ch.config import DEFAULT_IMAGE_SIZE, NUM_CLASSES, OUTPUTS_DIR
from cine_4ch.dataset import load_split_cases
from cine_4ch.io import choose_representative_frame, extract_frame, load_pair
from cine_4ch.model import UNet2D
from cine_4ch.viz import overlay_label
from src.preprocessing.emd_enhancement import (
    EMEnhancementConfig,
    enhance_mri_slice,
    safe_minmax_normalize,
)

ABLATION_ROOT = OUTPUTS_DIR / "emd_ablation"
OUT_DIR = ABLATION_ROOT / "mentor_figures"
SPLITS_CSV = OUTPUTS_DIR / "splits_4ch.csv"

# Prefer the same cases already used in qualitative overlays (seed=42 sample).
CANDIDATE_STEMS = [
    "CINE_4CH_009",
    "CINE_4CH_013",
    "CINE_4CH_014",
    "CINE_4CH_015",
    "CINE_4CH_020",
    "CINE_4CH_023",
    "CINE_4CH_054",
    "CINE_4CH_065",
    "CINE_4CH_066",
    "CINE_4CH_067",
]

PREPROC_PANELS = [
    ("Original MRI", EMEnhancementConfig(mode="original", imf_indices=[])),
    ("Subtract IMF 0", EMEnhancementConfig(mode="subtract", imf_indices=[0])),
    ("Subtract IMF 2", EMEnhancementConfig(mode="subtract", imf_indices=[2])),
    (
        "Subtract IMFs [-2, -3]",
        EMEnhancementConfig(mode="subtract", imf_indices=[-2, -3]),
    ),
]


def resize_hw(image_2d: np.ndarray, label_2d: np.ndarray, size=DEFAULT_IMAGE_SIZE):
    label_t = torch.from_numpy(label_2d).float().unsqueeze(0).unsqueeze(0)
    label_r = F.interpolate(label_t, size=size, mode="nearest").squeeze().numpy().astype(np.int64)
    image_t = torch.from_numpy(image_2d).float().unsqueeze(0).unsqueeze(0)
    image_r = F.interpolate(image_t, size=size, mode="bilinear", align_corners=False)
    return image_r.squeeze().numpy().astype(np.float32), label_r


def score_case(case) -> tuple[int, int, str]:
    image, label, _ = load_pair(case)
    fidx = choose_representative_frame(label)
    gt = extract_frame(label, fidx).astype(np.int64)
    n_classes = int(len(np.unique(gt[gt > 0])))
    fg = int((gt > 0).sum())
    return n_classes, fg, case.stem


def pick_case(test_cases):
    by_stem = {c.stem: c for c in test_cases}
    candidates = [by_stem[s] for s in CANDIDATE_STEMS if s in by_stem]
    if not candidates:
        candidates = list(test_cases)
    ranked = sorted((score_case(c) for c in candidates), reverse=True)
    best_stem = ranked[0][2]
    return by_stem[best_stem]


def enhance_display(raw_2d: np.ndarray, cfg: EMEnhancementConfig) -> np.ndarray:
    out = enhance_mri_slice(raw_2d, cfg)
    if out.ndim == 3:
        out = out[..., 0]
    return safe_minmax_normalize(out, clip=True)


def load_model(run_id: str, device: torch.device) -> tuple[UNet2D, dict]:
    ckpt_path = ABLATION_ROOT / run_id / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Look up in_channels from catalog; fall back to 1.
    specs = {s.run_id: s for s in default_ablation_runs()}
    in_ch = 1
    if run_id in specs:
        from src.preprocessing.emd_enhancement import output_channels

        in_ch = output_channels(specs[run_id].emd_config)
    model = UNet2D(in_channels=in_ch, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, {
        "path": str(ckpt_path),
        "epoch": checkpoint.get("epoch"),
        "val_dice_mean_foreground": checkpoint.get("val_dice_mean_foreground"),
    }


@torch.no_grad()
def predict(model: UNet2D, image_2d: np.ndarray, cfg: EMEnhancementConfig, device: torch.device) -> np.ndarray:
    enhanced = enhance_mri_slice(image_2d, cfg)
    if enhanced.ndim == 2:
        enhanced = enhanced[..., None]
    # Match training: resize after enhance.
    h, w, c = enhanced.shape
    img_t = torch.from_numpy(enhanced).float().permute(2, 0, 1).unsqueeze(0)
    img_t = F.interpolate(img_t, size=DEFAULT_IMAGE_SIZE, mode="bilinear", align_corners=False)
    logits = model(img_t.to(device))
    return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)


def save_preproc_figure(panels: list[tuple[str, np.ndarray]], case_stem: str, frame_idx: int, path: Path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"{case_stem}  ·  frame {frame_idx}", fontsize=12, y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_pred_figure(
    base_gray: np.ndarray,
    gt: np.ndarray,
    pred_original: np.ndarray,
    pred_imf0: np.ndarray,
    case_stem: str,
    frame_idx: int,
    path: Path,
):
    panels = [
        ("Ground truth", overlay_label(base_gray, gt)),
        ("Original model", overlay_label(base_gray, pred_original)),
        ("Subtract IMF 0 model", overlay_label(base_gray, pred_imf0)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"{case_stem}  ·  frame {frame_idx}", fontsize=12, y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    device = torch.device("cpu")
    test_cases = load_split_cases(SPLITS_CSV, "test")
    case = pick_case(test_cases)
    image, label, _ = load_pair(case)
    frame_idx = choose_representative_frame(label)
    raw = extract_frame(image, frame_idx).astype(np.float32)
    gt_full = extract_frame(label, frame_idx).astype(np.int64)

    # Preprocessing panels (same raw frame, per-panel min-max to [0,1]).
    preproc_panels = [(title, enhance_display(raw, cfg)) for title, cfg in PREPROC_PANELS]
    preproc_path = OUT_DIR / f"{case.stem}_frame{frame_idx:03d}_preprocessing_comparison.png"
    save_preproc_figure(preproc_panels, case.stem, frame_idx, preproc_path)

    # Prediction figure: overlay on original-normalized image for fair visual comparison.
    original_disp = enhance_display(raw, EMEnhancementConfig(mode="original", imf_indices=[]))
    _, gt_resized = resize_hw(original_disp, gt_full)

    model_orig, meta_orig = load_model("original", device)
    model_imf0, meta_imf0 = load_model("subtract_imf_0", device)

    pred_orig = predict(
        model_orig,
        raw,
        EMEnhancementConfig(mode="original", imf_indices=[]),
        device,
    )
    pred_imf0 = predict(
        model_imf0,
        raw,
        EMEnhancementConfig(mode="subtract", imf_indices=[0]),
        device,
    )

    # Overlay base at model resolution.
    base_resized, _ = resize_hw(original_disp, gt_full)
    pred_path = OUT_DIR / f"{case.stem}_frame{frame_idx:03d}_prediction_comparison.png"
    save_pred_figure(base_resized, gt_resized, pred_orig, pred_imf0, case.stem, frame_idx, pred_path)

    meta = {
        "case": case.stem,
        "frame_index": int(frame_idx),
        "split": "test",
        "image_size": list(DEFAULT_IMAGE_SIZE),
        "figures": {
            "preprocessing_comparison": str(preproc_path),
            "prediction_comparison": str(pred_path),
        },
        "checkpoints_used": {
            "original": meta_orig,
            "subtract_imf_0": meta_imf0,
        },
        "note": "Inference only from existing best.pt checkpoints; no training.",
    }
    meta_path = OUT_DIR / "figure_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Selected:", case.stem, "frame", frame_idx)
    print("Wrote:", preproc_path)
    print("Wrote:", pred_path)
    print("Wrote:", meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
