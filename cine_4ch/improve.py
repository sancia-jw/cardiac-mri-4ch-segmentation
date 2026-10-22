"""Training-recipe experiments for 4CH segmentation (no decomposition).

Everything runs from raw NIfTI volumes held in memory. Batches are sampled and
augmented on the GPU, so a run is limited by the model, not the data loader.

Each 4CH volume stacks several slice positions, each a full cine loop
(e.g. 75 frames = 3 slices x 25 phases). ``slice_blocks`` finds those loops so
temporal context (neighbouring phases of the same slice) never crosses into
another slice.

Two preprocessing presets:
- ``legacy``: per-frame min-max, bilinear resize to 160x160 (the earlier
  baseline in ``cine_4ch.ablation``).
- ``canvas``: per-volume percentile clip + z-score at native pixel size,
  zero-padded to a canvas divisible by 32.

All models are evaluated the same way: logits are mapped back to the native
grid, and Dice is pooled over all frames of a case, then averaged over cases.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from cine_4ch.config import LABEL_NAMES, NUM_CLASSES
from cine_4ch.io import CasePair, load_pair
from cine_4ch.metrics import combined_loss
from cine_4ch.model import UNet2D

LEGACY_SIZE = (160, 160)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def slice_blocks(image: np.ndarray, expected: Sequence[int] = (3, 2, 4, 1)) -> List[Tuple[int, int]]:
    """Split the frame axis of an (H, W, T) cine volume into per-slice loops.

    A new slice position shows up as a jump in mean absolute frame difference
    several times larger than the phase-to-phase change. Falls back to equal
    blocks when the jumps don't give equal-length loops.
    """
    t = image.shape[-1]
    if t < 4:
        return [(0, t)]
    frames = image.reshape(-1, t).astype(np.float64)
    diff = np.abs(np.diff(frames, axis=1)).mean(axis=0) / (frames.mean() + 1e-8)
    cuts = [i + 1 for i in np.flatnonzero(diff > 4.0 * np.median(diff))]
    bounds = [0, *cuts, t]
    blocks = list(zip(bounds[:-1], bounds[1:]))
    lengths = {e - s for s, e in blocks}
    if len(blocks) > 1 and len(lengths) == 1 and min(lengths) >= 8:
        return blocks
    for n in expected:
        if t % n == 0 and t // n >= 8:
            step = t // n
            return [(i * step, (i + 1) * step) for i in range(n)]
    return [(0, t)]


def context_offsets(context: int) -> List[int]:
    """Phase offsets fed as channels, centre frame in the middle (context is odd)."""
    if context < 1 or context % 2 == 0:
        raise ValueError(f"context must be a positive odd number, got {context}")
    half = context // 2
    return list(range(-half, half + 1))


def neighbour_index(blocks: Sequence[Tuple[int, int]], offsets: Sequence[int]) -> np.ndarray:
    """(T, K) frame indices of each frame's phase neighbours, wrapping within its cine loop."""
    t = blocks[-1][1]
    out = np.zeros((t, len(offsets)), dtype=np.int64)
    for start, end in blocks:
        n = end - start
        for f in range(start, end):
            out[f] = [start + (f - start + o) % n for o in offsets]
    return out


def normalize_volume(image: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    """Clip a whole case to its [low, high] percentiles, then z-score it."""
    lo, hi = np.percentile(image, [low, high])
    clipped = np.clip(image, lo, hi)
    std = clipped.std()
    return ((clipped - clipped.mean()) / (std if std > 0 else 1.0)).astype(np.float32)


def minmax_frames(image: np.ndarray) -> np.ndarray:
    """Per-frame min-max to [0, 1] (the legacy normalisation)."""
    lo = image.min(axis=(0, 1), keepdims=True)
    hi = image.max(axis=(0, 1), keepdims=True)
    return ((image - lo) / np.maximum(hi - lo, 1e-8)).astype(np.float32)


@dataclass
class CaseVolume:
    stem: str
    image: np.ndarray  # (T, H, W) preprocessed, model grid
    label: np.ndarray  # (T, H, W) uint8, model grid (training target)
    native_label: np.ndarray  # (T, h, w) uint8, original grid (evaluation)
    neighbours: np.ndarray  # (T, K)
    native_hw: Tuple[int, int]
    offset: Tuple[int, int]  # top-left of the (scaled) image inside the canvas
    scaled_hw: Optional[Tuple[int, int]] = None  # image size on the model grid when upsampled
    triplets: Optional[np.ndarray] = None  # (T, 3) phases t-1, t, t+1 within each cine loop


def canvas_shape(cases_hw: Sequence[Tuple[int, int]], multiple: int = 32) -> Tuple[int, int]:
    h = max(hw[0] for hw in cases_hw)
    w = max(hw[1] for hw in cases_hw)
    return (int(math.ceil(h / multiple) * multiple), int(math.ceil(w / multiple) * multiple))


def _pad_to(arr: np.ndarray, shape: Tuple[int, int], value: float) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Centre-pad (T, h, w) to (T, *shape)."""
    t, h, w = arr.shape
    top, left = (shape[0] - h) // 2, (shape[1] - w) // 2
    out = np.full((t, *shape), value, dtype=arr.dtype)
    out[:, top:top + h, left:left + w] = arr
    return out, (top, left)


def _resize_stack(arr: np.ndarray, size: Tuple[int, int], mode: str) -> np.ndarray:
    tensor = torch.from_numpy(arr.astype(np.float32)).unsqueeze(1)
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    return F.interpolate(tensor, size=size, mode=mode, **kwargs).squeeze(1).numpy()


def load_cases(
    cases: Sequence[CasePair],
    preset: str,
    context: int,
    canvas: Optional[Tuple[int, int]] = None,
    upsample: float = 1.0,
) -> List[CaseVolume]:
    """Load and preprocess cases. ``canvas`` is required for the canvas preset.

    ``upsample`` > 1 (canvas preset) enlarges image (bilinear) and training label
    (nearest) before padding; evaluation still uses the native label.
    """
    offsets = context_offsets(context)
    out: List[CaseVolume] = []
    for case in cases:
        image, label, _ = load_pair(case)
        if image.ndim == 2:
            image, label = image[..., None], label[..., None]
        blocks = slice_blocks(image)
        image_t = np.moveaxis(image, -1, 0)
        label_t = np.moveaxis(label, -1, 0).astype(np.uint8)
        native_hw = image_t.shape[1:]
        if preset == "legacy":
            img = _resize_stack(minmax_frames(image).transpose(2, 0, 1), LEGACY_SIZE, "bilinear")
            lab = _resize_stack(label_t, LEGACY_SIZE, "nearest").astype(np.uint8)
            offset = (0, 0)
        elif preset == "canvas":
            if canvas is None:
                raise ValueError("canvas preset needs a canvas shape")
            norm = np.moveaxis(normalize_volume(image), -1, 0)
            lab = label_t
            if upsample != 1.0:
                size = scaled_size(native_hw, upsample)
                norm = _resize_stack(norm, size, "bilinear")
                lab = _resize_stack(label_t, size, "nearest").astype(np.uint8)
            img, offset = _pad_to(norm, canvas, float(norm.min()))
            lab, _ = _pad_to(lab, canvas, 0)
        else:
            raise ValueError(f"Unknown preset {preset}")
        out.append(CaseVolume(case.stem, img.astype(np.float32), lab, label_t, neighbour_index(blocks, offsets),
                              tuple(native_hw), offset,
                              lab_shape(native_hw, upsample) if preset == "canvas" else None,
                              neighbour_index(blocks, [-1, 0, 1])))
    return out


def scaled_size(hw: Sequence[int], upsample: float) -> Tuple[int, int]:
    return (int(round(hw[0] * upsample)), int(round(hw[1] * upsample)))


def lab_shape(hw: Sequence[int], upsample: float) -> Tuple[int, int]:
    return scaled_size(hw, upsample) if upsample != 1.0 else (int(hw[0]), int(hw[1]))


class GPUFrames:
    """All training frames on one device; batches are gathered by index."""

    def __init__(self, volumes: Sequence[CaseVolume], device: torch.device) -> None:
        images, labels, neighbours = [], [], []
        base = 0
        for vol in volumes:
            images.append(vol.image)
            labels.append(vol.label)
            neighbours.append(vol.neighbours + base)
            base += vol.image.shape[0]
        self.images = torch.from_numpy(np.concatenate(images)).to(device=device, dtype=torch.float16)
        self.labels = torch.from_numpy(np.concatenate(labels)).to(device=device, dtype=torch.uint8)
        self.neighbours = torch.from_numpy(np.concatenate(neighbours)).to(device)
        self.device = device

    def __len__(self) -> int:
        return self.images.shape[0]

    def batch(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[self.neighbours[idx]].float(), self.labels[idx].long()


# ---------------------------------------------------------------------------
# Augmentation (batched, on device)
# ---------------------------------------------------------------------------

@dataclass
class AugmentConfig:
    hflip: float = 0.0  # legacy only; mirrors the anatomy
    affine: float = 0.0
    rotate_deg: float = 25.0
    scale: Tuple[float, float] = (0.8, 1.25)
    shift: float = 0.08
    elastic: float = 0.0
    elastic_alpha: float = 0.03
    brightness: float = 0.0
    contrast: float = 0.0
    gamma: float = 0.0
    noise: float = 0.0
    blur: float = 0.0
    lowres: float = 0.0


LEGACY_AUG = AugmentConfig(hflip=0.5)
STRONG_AUG = AugmentConfig(affine=0.7, elastic=0.25, brightness=0.3, contrast=0.3, gamma=0.3,
                           noise=0.15, blur=0.2, lowres=0.2)


def _coin(p: float, n: int, device) -> torch.Tensor:
    return torch.rand(n, device=device) < p


def _gaussian_blur(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Per-sample separable Gaussian blur; x (B, C, H, W), sigma (B,)."""
    radius = 3
    grid = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(grid[None] ** 2) / (2 * sigma[:, None] ** 2))
    kernel = kernel / kernel.sum(1, keepdim=True)  # (B, k)
    b, c, h, w = x.shape
    y = x.reshape(1, b * c, h, w)
    k = kernel.repeat_interleave(c, 0)
    y = F.conv2d(F.pad(y, (radius, radius, 0, 0), mode="replicate"), k[:, None, None, :], groups=b * c)
    y = F.conv2d(F.pad(y, (0, 0, radius, radius), mode="replicate"), k[:, None, :, None], groups=b * c)
    return y.reshape(b, c, h, w)


def augment(images: torch.Tensor, labels: torch.Tensor, cfg: AugmentConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same spatial transform for every context channel of a sample."""
    b, _, h, w = images.shape
    dev = images.device
    if cfg.hflip:
        flip = _coin(cfg.hflip, b, dev)
        images = torch.where(flip[:, None, None, None], images.flip(-1), images)
        labels = torch.where(flip[:, None, None], labels.flip(-1), labels)

    if cfg.affine or cfg.elastic:
        theta = torch.eye(2, 3, device=dev).repeat(b, 1, 1)
        do = _coin(cfg.affine, b, dev)
        ang = (torch.rand(b, device=dev) * 2 - 1) * math.radians(cfg.rotate_deg) * do
        lo, hi = cfg.scale
        zoom = torch.exp(torch.empty(b, device=dev).uniform_(math.log(lo), math.log(hi)))
        zoom = torch.where(do, zoom, torch.ones_like(zoom))
        # zoom > 1 enlarges the anatomy, so sample a smaller region
        cos, sin = torch.cos(ang) / zoom, torch.sin(ang) / zoom
        theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = cos, -sin, sin, cos
        theta[:, :, 2] = (torch.rand(b, 2, device=dev) * 2 - 1) * cfg.shift * 2 * do[:, None]
        grid = F.affine_grid(theta, (b, 1, h, w), align_corners=False)
        if cfg.elastic:
            field_ = torch.randn(b, 2, 5, 5, device=dev) * cfg.elastic_alpha
            field_ = field_ * _coin(cfg.elastic, b, dev)[:, None, None, None]
            grid = grid + F.interpolate(field_, size=(h, w), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
        floor = images.amin(dim=(1, 2, 3), keepdim=True)
        images = F.grid_sample(images - floor, grid, mode="bilinear", padding_mode="zeros",
                               align_corners=False) + floor
        labels = F.grid_sample(labels[:, None].float(), grid, mode="nearest", padding_mode="zeros",
                               align_corners=False)[:, 0].long()

    if cfg.brightness:
        m = torch.empty(b, device=dev).uniform_(0.75, 1.25)
        images = torch.where(_coin(cfg.brightness, b, dev)[:, None, None, None], images * m[:, None, None, None], images)
    if cfg.contrast:
        m = torch.empty(b, device=dev).uniform_(0.75, 1.25)[:, None, None, None]
        mean = images.mean(dim=(1, 2, 3), keepdim=True)
        images = torch.where(_coin(cfg.contrast, b, dev)[:, None, None, None], (images - mean) * m + mean, images)
    if cfg.gamma:
        g = torch.exp(torch.empty(b, device=dev).uniform_(math.log(0.7), math.log(1.5)))[:, None, None, None]
        lo = images.amin(dim=(1, 2, 3), keepdim=True)
        hi = images.amax(dim=(1, 2, 3), keepdim=True)
        unit = ((images - lo) / (hi - lo + 1e-6)).clamp(0, 1)
        images = torch.where(_coin(cfg.gamma, b, dev)[:, None, None, None], unit ** g * (hi - lo) + lo, images)
    if cfg.blur:
        sig = torch.empty(b, device=dev).uniform_(0.5, 1.2)
        images = torch.where(_coin(cfg.blur, b, dev)[:, None, None, None], _gaussian_blur(images, sig), images)
    if cfg.lowres:
        f = float(np.random.uniform(0.5, 0.85))
        small = F.interpolate(images, scale_factor=f, mode="nearest")
        coarse = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
        images = torch.where(_coin(cfg.lowres, b, dev)[:, None, None, None], coarse, images)
    if cfg.noise:
        sd = torch.empty(b, device=dev).uniform_(0.0, 0.1)[:, None, None, None]
        images = torch.where(_coin(cfg.noise, b, dev)[:, None, None, None],
                             images + torch.randn_like(images) * sd, images)
    return images, labels


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = None
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(y + (x if self.skip is None else self.skip(x)))


class ResUNet(nn.Module):
    """Residual U-Net, strided-conv downsampling (nnU-Net-like). Input divisible by 2**(len(widths)-1)."""

    def __init__(self, in_channels: int, num_classes: int = NUM_CLASSES,
                 widths: Sequence[int] = (32, 64, 128, 256, 320, 320)) -> None:
        super().__init__()
        self.stem = ResBlock(in_channels, widths[0])
        self.down = nn.ModuleList()
        for i in range(1, len(widths)):
            self.down.append(nn.Sequential(ResBlock(widths[i - 1], widths[i], stride=2), ResBlock(widths[i], widths[i])))
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(widths) - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(widths[i], widths[i - 1], 2, stride=2))
            self.dec.append(ResBlock(widths[i - 1] * 2, widths[i - 1]))
        self.head = nn.Conv2d(widths[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.stem(x)]
        for block in self.down:
            skips.append(block(skips[-1]))
        y = skips.pop()
        for up, dec in zip(self.up, self.dec):
            y = dec(torch.cat([up(y), skips.pop()], dim=1))
        return self.head(y)


def build_model(name: str, in_channels: int) -> nn.Module:
    if name == "unet_legacy":
        return UNet2D(in_channels=in_channels, num_classes=NUM_CLASSES)
    if name == "resunet":
        return ResUNet(in_channels)
    if name.startswith("smp_"):
        import segmentation_models_pytorch as smp

        return smp.Unet(name[4:], encoder_weights="imagenet", in_channels=in_channels, classes=NUM_CLASSES)
    raise ValueError(f"Unknown model {name}")


# ---------------------------------------------------------------------------
# Config, training, evaluation
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    name: str
    preset: str = "canvas"
    model: str = "resunet"
    context: int = 1
    aug: str = "strong"
    epochs: int = 80
    steps_per_epoch: int = 250
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    schedule: str = "cosine"
    warmup_epochs: int = 3
    amp: bool = True
    checkpoint: str = "final"  # "final" or "best_val"
    extra_train: List[str] = field(default_factory=list)  # e.g. ["4CH_VAL"]
    train_on_val: bool = False
    val_every: int = 5
    upsample: float = 1.0  # canvas preset: train and predict on an enlarged grid
    # cine_4ch.improve_semi: temporal consistency and partial-label external data
    temporal_weight: float = 0.0
    smooth_weight: float = 0.0
    smooth_tolerance: float = 0.04
    unlabelled: List[str] = field(default_factory=list)  # image-only views, e.g. ["4CH_TST"]
    unlabelled_triplets: int = 4
    external: Optional[str] = None  # folder under --extra-root, e.g. "4CH_CMRX"
    external_frac: float = 0.375
    external_merge: List[int] = field(default_factory=lambda: [0, 2])  # myocardium not annotated

    @property
    def augment_config(self) -> AugmentConfig:
        return {"legacy": LEGACY_AUG, "strong": STRONG_AUG, "none": AugmentConfig()}[self.aug]


def run_config_from_dict(name: str, raw: Dict) -> RunConfig:
    known = set(RunConfig.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{name}: unknown keys {sorted(unknown)}")
    return RunConfig(name=name, **{k: v for k, v in raw.items() if k != "name"})


def lr_factor(cfg: RunConfig, step: int, total: int) -> float:
    if cfg.schedule == "constant":
        return 1.0
    warm = cfg.warmup_epochs * cfg.steps_per_epoch
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


@torch.no_grad()
def predict_case(models: Sequence[nn.Module], vol: CaseVolume, preset: str, device: torch.device,
                 batch: int = 32) -> np.ndarray:
    """Mean softmax over ``models`` on the native grid: (T, C, h, w) float32."""
    t = vol.image.shape[0]
    h, w = vol.native_hw
    top, left = vol.offset
    image = torch.from_numpy(vol.image).to(device)
    nbr = torch.from_numpy(vol.neighbours).to(device)
    out = np.zeros((t, NUM_CLASSES, h, w), dtype=np.float32)
    for start in range(0, t, batch):
        idx = torch.arange(start, min(start + batch, t), device=device)
        x = image[nbr[idx]]
        probs = 0
        for model in models:
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(x).float()
            if preset == "legacy":
                logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            else:
                sh, sw = vol.scaled_hw or (h, w)
                logits = logits[:, :, top:top + sh, left:left + sw]
                if (sh, sw) != (h, w):
                    logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            probs = probs + logits.softmax(1)
        out[start:start + len(idx)] = (probs / len(models)).cpu().numpy()
    return out


def largest_component(pred: np.ndarray) -> np.ndarray:
    """Keep the largest 2D connected component of each foreground class in each frame."""
    out = pred.copy()
    for f in range(pred.shape[0]):
        for c in range(1, NUM_CLASSES):
            mask = pred[f] == c
            lab, n = ndimage.label(mask)
            if n > 1:
                sizes = ndimage.sum(mask, lab, range(1, n + 1))
                out[f][mask & (lab != 1 + int(np.argmax(sizes)))] = 0
    return out


def dice_per_class(pred: np.ndarray, gt: np.ndarray) -> List[float]:
    """Dice over every voxel of pred/gt for classes 1..C-1; NaN when a class is absent from both."""
    scores = []
    for c in range(1, NUM_CLASSES):
        p, g = pred == c, gt == c
        denom = p.sum() + g.sum()
        scores.append(float(2 * (p & g).sum() / denom) if denom else float("nan"))
    return scores


def legacy_batch_dice(pred: np.ndarray, gt: np.ndarray, batch: int = 4) -> List[float]:
    """Per-batch Dice (4 consecutive frames, eps-smoothed as cine_4ch.metrics) for each batch: fg mean."""
    out = []
    for s in range(0, pred.shape[0], batch):
        p, g = pred[s:s + batch], gt[s:s + batch]
        fg = []
        for c in range(1, NUM_CLASSES):
            pc, gc = p == c, g == c
            fg.append((2 * (pc & gc).sum() + 1e-6) / (pc.sum() + gc.sum() + 1e-6))
        out.append(float(np.mean(fg)))
    return out


def evaluate(models: Sequence[nn.Module], volumes: Sequence[CaseVolume], preset: str,
             device: torch.device) -> Tuple[Dict[str, float], List[Dict]]:
    """Case-pooled Dice (raw and largest-component), averaged over cases."""
    for m in models:
        m.eval()
    rows = []
    legacy = []
    for vol in volumes:
        probs = predict_case(models, vol, preset, device)
        pred = probs.argmax(1).astype(np.uint8)
        gt = vol.native_label
        raw = dice_per_class(pred, gt)
        lcc = dice_per_class(largest_component(pred), gt)
        legacy.extend(legacy_batch_dice(pred, gt))
        row = {"case": vol.stem}
        for c, (r, l) in enumerate(zip(raw, lcc), start=1):
            row[f"dice_{c}"] = r
            row[f"dice_lcc_{c}"] = l
        row["fg"] = float(np.nanmean(raw))
        row["fg_lcc"] = float(np.nanmean(lcc))
        rows.append(row)
    summary = {"fg": float(np.mean([r["fg"] for r in rows])),
               "fg_lcc": float(np.mean([r["fg_lcc"] for r in rows])),
               "legacy_fg": float(np.mean(legacy))}
    for c in range(1, NUM_CLASSES):
        summary[f"dice_{c}"] = float(np.nanmean([r[f"dice_{c}"] for r in rows]))
        summary[f"dice_lcc_{c}"] = float(np.nanmean([r[f"dice_lcc_{c}"] for r in rows]))
    return summary, rows


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(cfg: RunConfig, train_vols: Sequence[CaseVolume], val_vols: Sequence[CaseVolume],
          device: torch.device, seed: int, run_dir: Path, log=print) -> nn.Module:
    """Train one model; writes train_log.csv and final.pt (and best_val.pt) to run_dir."""
    set_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    frames = GPUFrames(train_vols, device)
    model = build_model(cfg.model, cfg.context).to(device)
    if cfg.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    aug = cfg.augment_config
    steps = cfg.steps_per_epoch if cfg.steps_per_epoch > 0 else math.ceil(len(frames) / cfg.batch_size)
    cfg_steps = RunConfig(**{**asdict(cfg), "steps_per_epoch": steps})
    total = cfg.epochs * steps
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best, step = -1.0, 0
    fields = ["epoch", "loss", "lr", "val_fg", "seconds"]
    t0 = time.time()
    with open(run_dir / "train_log.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            order = torch.randperm(len(frames), generator=gen)
            losses = []
            for i in range(steps):
                sel = order[(i * cfg.batch_size) % len(frames):][:cfg.batch_size]
                if len(sel) < cfg.batch_size:
                    sel = torch.randint(len(frames), (cfg.batch_size,), generator=gen)
                x, y = frames.batch(sel.to(device))
                x, y = augment(x, y, aug)
                for g in opt.param_groups:
                    g["lr"] = cfg.lr * lr_factor(cfg_steps, step, total)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=use_amp):
                    logits = model(x)
                loss = combined_loss(logits.float(), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                losses.append(loss.item())
                step += 1
            row = {"epoch": epoch, "loss": f"{np.mean(losses):.5f}", "lr": f"{opt.param_groups[0]['lr']:.2e}",
                   "val_fg": "", "seconds": f"{time.time() - t0:.0f}"}
            if val_vols and (epoch % cfg.val_every == 0 or epoch == cfg.epochs or cfg.checkpoint == "best_val"):
                val, _ = evaluate([model], val_vols, cfg.preset, device)
                row["val_fg"] = f"{val['fg']:.5f}"
                if val["fg"] > best:
                    best = val["fg"]
                    torch.save(model.state_dict(), run_dir / "best_val.pt")
            writer.writerow(row)
            fh.flush()
            log(f"  [{cfg.name} s{seed}] ep {epoch:3d}/{cfg.epochs} loss {row['loss']} val_fg {row['val_fg'] or '-'} "
                f"{row['seconds']}s")
    torch.save(model.state_dict(), run_dir / "final.pt")
    if cfg.checkpoint == "best_val" and (run_dir / "best_val.pt").exists():
        model.load_state_dict(torch.load(run_dir / "best_val.pt", map_location=device))
    del frames
    return model


def write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def class_names() -> Dict[int, str]:
    return {c: LABEL_NAMES[c] for c in range(1, NUM_CLASSES)}
