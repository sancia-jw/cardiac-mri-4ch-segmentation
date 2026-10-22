"""Temporal-consistency and partial-label training on top of ``cine_4ch.improve``.

Two additions to the improve recipe, both switched on from RunConfig:

- Temporal consistency across the cine loop. Training draws triplets of
  consecutive phases (t-1, t, t+1) of one slice, transformed identically.
  ``temporal_weight``: on annotated triplets, the predicted change in each
  structure's (soft) area between phases must match the annotated change.
  ``smooth_weight``: on unannotated cine loops (e.g. 4CH_TST), the area of a
  structure at phase t should lie near the mean of its neighbours; only
  curvature above ``smooth_tolerance`` (relative to the area) is penalised,
  since annotated loops show ~2-4 % (mean) phase-to-phase curvature.

- Partial labels from another dataset (``external``), e.g. CMRxRecon 4CH,
  which annotates the four chambers but not the myocardium. Its frames are
  trained with a marginal loss: background and myocardium probabilities are
  summed into one class, so the model is not taught that myocardium is
  background. See scripts/convert_cmrxrecon.py.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cine_4ch import improve
from cine_4ch.config import NUM_CLASSES
from cine_4ch.io import load_volume
from cine_4ch.metrics import combined_loss

CODE_BASE = NUM_CLASSES  # labels of a triplet packed as y0 + 6*y1 + 36*y2 so one nearest-sample moves all three


def triplet_index(blocks: Sequence[Tuple[int, int]]) -> np.ndarray:
    """(T, 3) indices of phases t-1, t, t+1, wrapping within each cine loop."""
    return improve.neighbour_index(blocks, [-1, 0, 1])


def load_unlabelled(image_paths: Sequence[Path], context: int, canvas: Tuple[int, int]) -> List[improve.CaseVolume]:
    """Cine volumes without annotation (labels all 0, never used as targets)."""
    offsets = improve.context_offsets(context)
    out = []
    for path in image_paths:
        image = load_volume(path)[0]
        if image.ndim == 2:
            image = image[..., None]
        blocks = improve.slice_blocks(image)
        norm = np.moveaxis(improve.normalize_volume(image), -1, 0)
        img, offset = improve._pad_to(norm, canvas, float(norm.min()))
        empty = np.zeros(img.shape, dtype=np.uint8)
        out.append(improve.CaseVolume(path.name.split(".")[0], img, empty, empty[:, :1, :1],
                                      improve.neighbour_index(blocks, offsets), tuple(norm.shape[1:]), offset,
                                      triplets=triplet_index(blocks)))
    return out


def static_context(vols: Sequence[improve.CaseVolume], context: int) -> None:
    """External frames are not consecutive phases: every context channel shows the frame itself."""
    for vol in vols:
        t = vol.image.shape[0]
        vol.neighbours = np.repeat(np.arange(t)[:, None], context, axis=1)


class TripletFrames(improve.GPUFrames):
    """GPUFrames plus each frame's (t-1, t, t+1) indices."""

    def __init__(self, volumes: Sequence[improve.CaseVolume], device: torch.device) -> None:
        super().__init__(volumes, device)
        trips, base = [], 0
        for vol in volumes:
            trips.append(vol.triplets + base)
            base += vol.image.shape[0]
        self.triplets = torch.from_numpy(np.concatenate(trips)).to(device)

    def triplet_batch(self, centres: torch.Tensor, aug: improve.AugmentConfig) -> Tuple[torch.Tensor, torch.Tensor]:
        """(3n, K, H, W) images and (3n, H, W) labels, one random transform per triplet."""
        idx = self.triplets[centres]  # (n, 3)
        n = idx.shape[0]
        x, y = self.batch(idx.reshape(-1))
        k, h, w = x.shape[1:]
        x = x.reshape(n, 3 * k, h, w)
        y = y.reshape(n, 3, h, w)
        code = y[:, 0] + CODE_BASE * y[:, 1] + CODE_BASE ** 2 * y[:, 2]
        x, code = improve.augment(x, code, aug)
        y = torch.stack([code % CODE_BASE, (code // CODE_BASE) % CODE_BASE, code // CODE_BASE ** 2], dim=1)
        return x.reshape(3 * n, k, h, w), y.reshape(3 * n, h, w)


def soft_areas(logits: torch.Tensor) -> torch.Tensor:
    """(3n, C, H, W) logits -> (n, 3, C) fraction of the image per class."""
    p = logits.float().softmax(1).mean(dim=(2, 3))
    return p.reshape(-1, 3, p.shape[1])


def area_dynamics_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mismatch between predicted and annotated area change t-1->t and t->t+1, relative to the area."""
    pred = soft_areas(logits)
    gt = F.one_hot(labels, NUM_CLASSES).float().mean(dim=(1, 2)).reshape(pred.shape)
    d_pred, d_gt = pred[:, 1:] - pred[:, :-1], gt[:, 1:] - gt[:, :-1]
    scale = gt[:, 1:2, 1:] + 1e-3
    return ((d_pred[..., 1:] - d_gt[..., 1:]).abs() / scale).mean()


def smoothness_loss(logits: torch.Tensor, tolerance: float) -> torch.Tensor:
    """Penalise area curvature |A_t - (A_{t-1}+A_{t+1})/2| / A_t above ``tolerance`` (foreground classes)."""
    a = soft_areas(logits)[..., 1:]
    curv = (a[:, 1] - 0.5 * (a[:, 0] + a[:, 2])).abs() / (a[:, 1].detach() + 1e-3)
    return F.relu(curv - tolerance).mean()


def marginal_loss(logits: torch.Tensor, labels: torch.Tensor, merge: Sequence[int]) -> torch.Tensor:
    """CE + soft Dice after summing the probabilities of the ``merge`` classes (first one is the target id)."""
    keep = [c for c in range(logits.shape[1]) if c not in merge]
    p = logits.float().softmax(1)
    merged = torch.cat([p[:, list(merge)].sum(1, keepdim=True), p[:, keep]], dim=1)
    remap = torch.zeros(NUM_CLASSES, dtype=torch.long, device=labels.device)
    for i, c in enumerate(keep, start=1):
        remap[c] = i
    target = remap[labels]
    ce = F.nll_loss(torch.log(merged.clamp_min(1e-7)), target)
    onehot = F.one_hot(target, merged.shape[1]).permute(0, 3, 1, 2).float()
    inter = (merged * onehot).sum((0, 2, 3))
    union = merged.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    dice = (2 * inter + 1e-6) / (union + 1e-6)
    return 0.5 * ce + 0.5 * (1 - dice.mean())


def uses_semi(cfg: improve.RunConfig) -> bool:
    return bool(cfg.temporal_weight or cfg.smooth_weight or cfg.external)


def train_semi(cfg: improve.RunConfig, train_vols, val_vols, device: torch.device, seed: int, run_dir: Path,
               unlabelled_vols=(), external_vols=(), log=print) -> nn.Module:
    """improve.train with temporal triplets, unannotated loops and/or partial-label external frames."""
    improve.set_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    frames = TripletFrames(train_vols, device)
    unl = TripletFrames(unlabelled_vols, device) if cfg.smooth_weight and unlabelled_vols else None
    ext = improve.GPUFrames(external_vols, device) if cfg.external and external_vols else None
    n_ext = int(round(cfg.batch_size * cfg.external_frac)) if ext else 0
    n_lab = cfg.batch_size - n_ext
    n_trip = n_lab // 3 if cfg.temporal_weight else 0
    model = improve.build_model(cfg.model, cfg.context).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    aug = cfg.augment_config
    steps = cfg.steps_per_epoch
    total = cfg.epochs * steps
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best, step = -1.0, 0
    fields = ["epoch", "loss", "sup", "temporal", "smooth", "external", "lr", "val_fg", "seconds"]
    t0 = time.time()
    with open(run_dir / "train_log.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            sums = {k: 0.0 for k in ("loss", "sup", "temporal", "smooth", "external")}
            for _ in range(steps):
                xs, parts = [], []
                if n_trip:
                    xl, yl = frames.triplet_batch(torch.randint(len(frames), (n_trip,), generator=gen).to(device), aug)
                else:
                    xl, yl = frames.batch(torch.randint(len(frames), (n_lab,), generator=gen).to(device))
                    xl, yl = improve.augment(xl, yl, aug)
                xs.append(xl)
                parts.append(len(xl))
                if ext is not None:
                    xe, ye = ext.batch(torch.randint(len(ext), (n_ext,), generator=gen).to(device))
                    xe, ye = improve.augment(xe, ye, aug)
                    xs.append(xe)
                    parts.append(len(xe))
                if unl is not None:
                    xu, _ = unl.triplet_batch(torch.randint(len(unl), (cfg.unlabelled_triplets,),
                                                            generator=gen).to(device), aug)
                    xs.append(xu)
                    parts.append(len(xu))
                for g in opt.param_groups:
                    g["lr"] = cfg.lr * improve.lr_factor(cfg, step, total)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=use_amp):
                    logits = model(torch.cat(xs)).float()
                chunks = list(torch.split(logits, parts))
                lab_logits = chunks.pop(0)
                sup = combined_loss(lab_logits, yl)
                loss = sup
                terms = {"sup": sup.item(), "temporal": 0.0, "smooth": 0.0, "external": 0.0}
                if n_trip:
                    tl = area_dynamics_loss(lab_logits, yl)
                    loss = loss + cfg.temporal_weight * tl
                    terms["temporal"] = tl.item()
                if ext is not None:
                    el = marginal_loss(chunks.pop(0), ye, cfg.external_merge)
                    loss = loss + el * n_ext / n_lab
                    terms["external"] = el.item()
                if unl is not None:
                    sl = smoothness_loss(chunks.pop(0), cfg.smooth_tolerance)
                    loss = loss + cfg.smooth_weight * sl
                    terms["smooth"] = sl.item()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                terms["loss"] = loss.item()
                for k, v in terms.items():
                    sums[k] += v
                step += 1
            row = {k: f"{v / steps:.5f}" for k, v in sums.items()}
            row.update({"epoch": epoch, "lr": f"{opt.param_groups[0]['lr']:.2e}", "val_fg": "",
                        "seconds": f"{time.time() - t0:.0f}"})
            if val_vols and (epoch % cfg.val_every == 0 or epoch == cfg.epochs):
                val, _ = improve.evaluate([model], val_vols, cfg.preset, device)
                row["val_fg"] = f"{val['fg']:.5f}"
                if val["fg"] > best:
                    best = val["fg"]
                    torch.save(model.state_dict(), run_dir / "best_val.pt")
            writer.writerow(row)
            fh.flush()
            log(f"  [{cfg.name} s{seed}] ep {epoch:3d}/{cfg.epochs} loss {row['loss']} (sup {row['sup']} "
                f"tc {row['temporal']} sm {row['smooth']} ext {row['external']}) val_fg {row['val_fg'] or '-'} "
                f"{row['seconds']}s")
    torch.save(model.state_dict(), run_dir / "final.pt")
    return model
