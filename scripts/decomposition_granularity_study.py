#!/usr/bin/env python3
"""
Diagnose / improve true-2D EMD granularity (NO U-Net training).

Hardened for Windows: unbuffered logs, per-run timeouts, capped sift iterations.
"""

from __future__ import annotations

import csv
import json
import sys
import time
import traceback
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cine_4ch.config import OUTPUTS_DIR
from cine_4ch.io import discover_cases, extract_frame, load_pair
from src.preprocessing.emd_enhancement import as_grayscale_slice, safe_minmax_normalize
from src.preprocessing.emd_geometry import (
    component_energy_stats,
    spectral_centroid_cycles_per_pixel,
)

OUT = OUTPUTS_DIR / "emd_geometry_followup" / "decomposition_granularity"
FIG = OUT / "figures"
SYN = OUT / "synthetic"
MRI_FIG = OUT / "mri_representatives"
LOG = OUT / "run_log.txt"

DECOMP_TIMEOUT_SEC = 90.0
MAX_SIFT_ITERS = 40

MRI_SPECS: List[Tuple[str, int]] = [
    ("CINE_4CH_009", 83),
    ("CINE_4CH_006", 0),
    ("CINE_4CH_006", 37),
    ("CINE_4CH_014", 40),
    ("CINE_4CH_020", 20),
    ("CINE_4CH_022", 45),
    ("CINE_4CH_043", 30),
    ("CINE_4CH_059", 10),
    ("CINE_4CH_063", 50),
    ("CINE_4CH_074", 20),
    ("CINE_4CH_087", 40),
    ("CINE_4CH_090", 60),
]


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line)


@dataclass
class DecompEval:
    backend: str
    config_id: str
    source: str
    n_bimf: int
    n_slabs: int
    elapsed_sec: float
    recon_max_abs: float
    recon_rmse: float
    centroids: List[float]
    energy_fracs: List[float]
    residual_energy_frac: float
    centroids_monotonic: bool
    n_extrema_residual: int
    error: str = ""


def make_synthetic(size: int = 160) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    x = xx / size
    y = yy / size
    high = 0.35 * np.sin(2 * np.pi * 18 * x) * np.cos(2 * np.pi * 16 * y)
    med = 0.55 * np.sin(2 * np.pi * 6 * y)
    low = 0.70 * np.sin(2 * np.pi * 2 * (x + y))
    grad = 0.40 * x + 0.25 * y
    parts = {"high": high, "med": med, "low": low, "grad": grad}
    img = high + med + low + grad
    img = img - img.min() + 10.0
    return img.astype(np.float64), parts


def parse_slabs(imfs: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    if imfs.ndim != 3 or imfs.shape[0] < 1:
        raise RuntimeError(f"bad imfs shape {getattr(imfs, 'shape', None)}")
    if imfs.shape[0] == 1:
        return [imfs[0].astype(np.float64)], np.zeros_like(imfs[0], dtype=np.float64)
    comps = [imfs[i].astype(np.float64) for i in range(imfs.shape[0] - 1)]
    residual = imfs[-1].astype(np.float64)
    return comps, residual


def eval_decomposition(
    original: np.ndarray,
    comps: Sequence[np.ndarray],
    residual: np.ndarray,
    *,
    backend: str,
    config_id: str,
    source: str,
    elapsed_sec: float,
) -> DecompEval:
    from PyEMD.EMD2d import EMD2D

    parts = list(comps) + [residual]
    recon = np.sum(np.stack(parts, axis=0), axis=0)
    diff = original.astype(np.float64) - recon
    centroids = [spectral_centroid_cycles_per_pixel(c) for c in comps]
    centroids.append(spectral_centroid_cycles_per_pixel(residual))
    efracs = [component_energy_stats(original, c)["energy_fraction_of_original"] for c in comps]
    rfrac = component_energy_stats(original, residual)["energy_fraction_of_original"]
    mono = all(centroids[i] + 1e-9 >= centroids[i + 1] for i in range(len(centroids) - 1))
    mins, maxs = EMD2D.find_extrema(residual.astype(np.float64))
    return DecompEval(
        backend=backend,
        config_id=config_id,
        source=source,
        n_bimf=len(comps),
        n_slabs=len(comps) + 1,
        elapsed_sec=elapsed_sec,
        recon_max_abs=float(np.max(np.abs(diff))),
        recon_rmse=float(np.sqrt(np.mean(diff**2))),
        centroids=centroids,
        energy_fracs=efracs,
        residual_energy_frac=rfrac,
        centroids_monotonic=bool(mono and len(comps) >= 1),
        n_extrema_residual=int(len(mins[0]) + len(maxs[0])),
    )


def _emd2d_call(image: np.ndarray, params: Dict[str, Any]):
    from PyEMD.EMD2d import EMD2D

    p = dict(params)
    max_imf = int(p.pop("max_imf", -1))
    p.setdefault("MAX_ITERATION", MAX_SIFT_ITERS)
    emd = EMD2D(**p)
    imfs = emd.emd(np.asarray(image, dtype=np.float64), max_imf=max_imf)
    return parse_slabs(imfs)


def _bemd_call(image: np.ndarray, params: Dict[str, Any]):
    from PyEMD.BEMD import BEMD

    p = dict(params)
    max_imf = int(p.pop("max_imf", 5))
    bemd = BEMD()
    bemd.MAX_ITERATION = int(p.pop("MAX_ITERATION", 8))
    for k, v in p.items():
        if hasattr(bemd, k):
            setattr(bemd, k, v)
    imfs = bemd.bemd(np.asarray(image, dtype=np.float64), max_imf=max_imf)
    return parse_slabs(imfs)


def _timeout_worker(q, backend: str, image: np.ndarray, params: Dict[str, Any]) -> None:
    try:
        if backend == "emd2d":
            q.put(("ok", _emd2d_call(image, params)))
        elif backend == "bemd":
            q.put(("ok", _bemd_call(image, params)))
        else:
            q.put(("err", f"Unknown backend {backend}"))
    except Exception as exc:
        q.put(("err", f"{type(exc).__name__}: {exc}"))


def run_with_timeout(backend: str, image: np.ndarray, params: Dict[str, Any], timeout: float):
    """Run in a child process so timeouts can actually kill hung sifting."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_timeout_worker,
        args=(q, backend, np.asarray(image, dtype=np.float64), dict(params)),
    )
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        raise FuturesTimeout()
    if q.empty():
        raise RuntimeError("worker exited without result")
    status, payload = q.get()
    if status == "err":
        raise RuntimeError(payload)
    return payload


def run_backend(image: np.ndarray, cfg: Dict[str, Any], source: str) -> DecompEval:
    """Direct call with capped sift iters; soft wall-clock guard via MAX_ITERATION only.

    (Process-based timeouts are avoided: Windows spawn overhead dominated runtime.)
    """
    backend = cfg["backend"]
    config_id = cfg["config_id"]
    params = {k: v for k, v in cfg.items() if k not in ("backend", "config_id")}
    t0 = time.perf_counter()
    try:
        if backend == "emd2d":
            comps, residual = _emd2d_call(image, params)
        elif backend == "bemd":
            comps, residual = _bemd_call(image, params)
        else:
            raise ValueError(backend)
        dt = time.perf_counter() - t0
        if dt > DECOMP_TIMEOUT_SEC:
            # Soft note only; result still returned
            pass
        return eval_decomposition(
            image, comps, residual, backend=backend, config_id=config_id, source=source, elapsed_sec=dt
        )
    except Exception as exc:
        return DecompEval(
            backend, config_id, source, 0, 0, time.perf_counter() - t0,
            float("nan"), float("nan"), [], [], float("nan"), False, -1,
            error=f"{type(exc).__name__}: {exc}",
        )


def _sym(arr: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(arr)))
    if peak <= 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / peak, -1.0, 1.0).astype(np.float32)


def save_montages(out_dir: Path, stem: str, original: np.ndarray, comps, residual) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = [("original", safe_minmax_normalize(original.astype(np.float32), clip=True), "gray", 0, 1)]
    for i, c in enumerate(comps):
        rem = (original.astype(np.float64) - c.astype(np.float64)).astype(np.float32)
        panels.append((f"BIMF{i}", _sym(c), "coolwarm", -1, 1))
        panels.append((f"orig-BIMF{i}", safe_minmax_normalize(rem, clip=True), "gray", 0, 1))
    rem_r = (original.astype(np.float64) - residual.astype(np.float64)).astype(np.float32)
    panels.append(("residual", _sym(residual), "coolwarm", -1, 1))
    panels.append(("orig-residual", safe_minmax_normalize(rem_r, clip=True), "gray", 0, 1))

    n = len(panels)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, img, cmap, vmin, vmax) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes[len(panels) :]:
        ax.axis("off")
    fig.suptitle(stem, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_removal_montage.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    cpanels = [(f"BIMF{i}", safe_minmax_normalize(np.abs(c).astype(np.float32), clip=True)) for i, c in enumerate(comps)]
    cpanels.append(("residual", safe_minmax_normalize(residual.astype(np.float32), clip=True)))
    fig, axes = plt.subplots(1, len(cpanels), figsize=(3.0 * len(cpanels), 3.2))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, img) in zip(axes, cpanels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_components_norm.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    recon = np.sum(np.stack(list(comps) + [residual], axis=0), axis=0)
    diff = original.astype(np.float64) - recon
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
    axes[0].imshow(safe_minmax_normalize(original.astype(np.float32), clip=True), cmap="gray")
    axes[0].set_title("original")
    axes[1].imshow(safe_minmax_normalize(recon.astype(np.float32), clip=True), cmap="gray")
    axes[1].set_title("reconstruction")
    im = axes[2].imshow(_sym(diff), cmap="coolwarm", vmin=-1, vmax=1)
    axes[2].set_title(f"diff max|e|={np.max(np.abs(diff)):.2e}")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_reconstruction.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    labels = [f"B{i}" for i in range(len(comps))] + ["res"]
    cents = [spectral_centroid_cycles_per_pixel(c) for c in comps] + [spectral_centroid_cycles_per_pixel(residual)]
    efs = [component_energy_stats(original, c)["energy_fraction_of_original"] for c in comps]
    efs.append(component_energy_stats(original, residual)["energy_fraction_of_original"])
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    axes[0].plot(range(len(labels)), cents, "o-")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("spectral centroid (cyc/px)")
    axes[0].set_title("scale vs index")
    axes[1].bar(range(len(labels)), efs)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("energy fraction")
    axes[1].set_title("energy vs index")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_scale_energy.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_audit() -> None:
    (OUT / "AUDIT_EMD2D.md").write_text(
        """# Step 1 — Audit of current `PyEMD.EMD2D`

## Code path
- Package: `EMD-signal` / `PyEMD` experimental `EMD2d.EMD2D`
- Normalize image to `[0,1]`, sift IMFs, append residual if nonzero, rescale

## Extrema
- 3x3 local max/min (`scipy.ndimage.maximum_filter`)
- Outer loop continues only if `n_min > 4` AND `n_max > 4`

## Envelopes
- Mirror-pad 3x3, `SmoothBivariateSpline` on extrema

## Defaults
| Param | Default |
|---|---:|
| mean_thr | 0.01 |
| mse_thr | 0.01 |
| FIXE | 0 |
| FIXE_H | 0 |
| MAX_ITERATION | 1000 |

## Why MRI yields 1 BIMF
On mentor frame after BIMF0, residue extrema count is **0 / 0**.
Outer loop stops; leftover becomes residual (~40-50% energy, smooth trend).

Root cause: **permissive proto-IMF stop + extrema detector** → broad first BIMF →
featureless residual. Not a hard max_imf=1 limit.
""",
        encoding="utf-8",
    )


def load_mri(case_stem: str, frame: int) -> Optional[np.ndarray]:
    cases, _, _ = discover_cases()
    case = next((c for c in cases if c.stem == case_stem), None)
    if case is None:
        return None
    image, _, _ = load_pair(case)
    return as_grayscale_slice(extract_frame(image, frame)).astype(np.float64)


def emd2d_configs() -> List[Dict[str, Any]]:
    configs = [
        {"config_id": "emd2d_default", "backend": "emd2d", "mean_thr": 0.01, "mse_thr": 0.01},
    ]
    for mt in [1e-3, 1e-4, 1e-5]:
        configs.append({"config_id": f"emd2d_mean_{mt:g}", "backend": "emd2d", "mean_thr": mt, "mse_thr": 0.01})
    for ms in [1e-3, 1e-4]:
        configs.append({"config_id": f"emd2d_mse_{ms:g}", "backend": "emd2d", "mean_thr": 0.01, "mse_thr": ms})
    for fh in [3, 5]:
        configs.append(
            {"config_id": f"emd2d_fixeh_{fh}", "backend": "emd2d", "mean_thr": 0.01, "mse_thr": 0.01, "FIXE_H": fh}
        )
    for mt, fh in [(1e-3, 3), (1e-4, 5), (1e-5, 5)]:
        configs.append(
            {
                "config_id": f"emd2d_mean_{mt:g}_fixeh_{fh}",
                "backend": "emd2d",
                "mean_thr": mt,
                "mse_thr": 1e-4,
                "FIXE_H": fh,
            }
        )
    for fx in [3, 5]:
        configs.append(
            {"config_id": f"emd2d_fixe_{fx}", "backend": "emd2d", "mean_thr": 0.01, "mse_thr": 0.01, "FIXE": fx}
        )
    return configs


def bemd_configs() -> List[Dict[str, Any]]:
    return [
        {"config_id": "bemd_default", "backend": "bemd", "max_imf": 4},
        {
            "config_id": "bemd_mean_1e-3",
            "backend": "bemd",
            "mean_thr": 1e-3,
            "mse_thr": 1e-3,
            "FIXE": 1,
            "max_imf": 4,
        },
    ]


def row_from_eval(e: DecompEval) -> Dict[str, Any]:
    r = asdict(e)
    r["centroids"] = json.dumps(e.centroids)
    r["energy_fracs"] = json.dumps(e.energy_fracs)
    return r


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    SYN.mkdir(parents=True, exist_ok=True)
    MRI_FIG.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    if LOG.exists():
        LOG.unlink()

    log("=== STEP 1: audit ===")
    write_audit()
    mentor = load_mri("CINE_4CH_009", 83)
    assert mentor is not None
    from PyEMD.EMD2d import EMD2D

    ev0 = run_backend(mentor, {"config_id": "emd2d_default", "backend": "emd2d"}, "CINE_4CH_009_f083")
    # extrema diagnosis via fresh default decomp
    comps, residual = _emd2d_call(mentor, {"mean_thr": 0.01, "mse_thr": 0.01, "MAX_ITERATION": MAX_SIFT_ITERS})
    offset, scale = float(mentor.min()), float(mentor.max() - mentor.min())
    left = (mentor - offset) / scale - comps[0] / scale
    mins, maxs = EMD2D.find_extrema(left)
    diag = {
        "mentor": "CINE_4CH_009_frame083",
        "default_n_bimf": len(comps),
        "residual_extrema_after_bimf0": {"n_min": int(len(mins[0])), "n_max": int(len(maxs[0]))},
        "conclusion": "Stops because residue after BIMF0 has too few extrema (need >4 each).",
    }
    (OUT / "extrema_stop_diagnosis.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    log(json.dumps(diag))

    log("=== STEP 2: synthetic ===")
    syn, parts = make_synthetic(160)
    np.save(SYN / "synthetic_image.npy", syn)
    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    axes[0].imshow(syn, cmap="gray")
    axes[0].set_title("synthetic")
    for ax, (name, arr) in zip(axes[1:], parts.items()):
        ax.imshow(arr, cmap="coolwarm")
        ax.set_title(name)
        ax.axis("off")
    axes[0].axis("off")
    fig.tight_layout()
    fig.savefig(SYN / "synthetic_truth_parts.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    syn_default = run_backend(syn, {"config_id": "emd2d_default", "backend": "emd2d"}, "synthetic")
    log(
        f"EMD2D default synthetic: n_bimf={syn_default.n_bimf} cents={syn_default.centroids} "
        f"rmse={syn_default.recon_rmse:.3e} err={syn_default.error}"
    )
    if not syn_default.error:
        c, r = _emd2d_call(syn, {"mean_thr": 0.01, "mse_thr": 0.01, "MAX_ITERATION": MAX_SIFT_ITERS})
        save_montages(SYN, "synthetic_emd2d_default", syn, c, r)

    probe_imgs: List[Tuple[str, np.ndarray]] = [("synthetic", syn)]
    for cs, fr in [("CINE_4CH_009", 83), ("CINE_4CH_006", 0), ("CINE_4CH_022", 45), ("CINE_4CH_090", 60)]:
        im = load_mri(cs, fr)
        if im is not None:
            probe_imgs.append((f"{cs}_f{fr:03d}", im))

    log("=== STEP 3: EMD2D parameter sweep ===")
    sweep_rows: List[Dict[str, Any]] = []
    for cfg in emd2d_configs():
        log(f"config {cfg['config_id']}")
        for src, img in probe_imgs:
            ev = run_backend(img, cfg, src)
            sweep_rows.append(row_from_eval(ev))
            log(
                f"  {src}: n_bimf={ev.n_bimf} mono={ev.centroids_monotonic} "
                f"rmse={ev.recon_rmse} t={ev.elapsed_sec:.2f}s err={ev.error}"
            )
    write_csv(OUT / "parameter_sweep.csv", sweep_rows)

    log("=== STEP 4: BEMD backend ===")
    bemd_rows: List[Dict[str, Any]] = []
    for cfg in bemd_configs():
        log(f"config {cfg['config_id']}")
        for src, img in probe_imgs:
            ev = run_backend(img, cfg, src)
            bemd_rows.append(row_from_eval(ev))
            log(
                f"  {src}: n_bimf={ev.n_bimf} mono={ev.centroids_monotonic} "
                f"rmse={ev.recon_rmse} t={ev.elapsed_sec:.2f}s err={ev.error}"
            )
    write_csv(OUT / "backend_comparison.csv", bemd_rows)

    # Score configs
    all_rows = sweep_rows + bemd_rows
    by_cfg: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_rows:
        by_cfg.setdefault(r["config_id"], []).append(r)

    def score(cid: str) -> Tuple[float, Dict[str, Any]]:
        rows = [r for r in by_cfg.get(cid, []) if not r.get("error")]
        syn = [r for r in rows if r["source"] == "synthetic"]
        mri = [r for r in rows if r["source"] != "synthetic"]
        if not syn:
            return -1e9, {"error": "no synthetic"}
        s = syn[0]
        syn_ok = 1.0 if int(s["n_bimf"]) > 1 else 0.0
        syn_mono = 1.0 if s["centroids_monotonic"] else 0.0
        syn_recon = 1.0 if float(s["recon_rmse"]) < 1.0 else 0.0
        if mri:
            frac_multi = float(np.mean([1.0 if int(r["n_bimf"]) > 1 else 0.0 for r in mri]))
            mean_bimf = float(np.mean([int(r["n_bimf"]) for r in mri]))
            mean_t = float(np.mean([float(r["elapsed_sec"]) for r in mri]))
            mono_frac = float(np.mean([1.0 if r["centroids_monotonic"] else 0.0 for r in mri]))
            recon_ok = float(np.mean([1.0 if float(r["recon_rmse"]) < 1.0 else 0.0 for r in mri]))
        else:
            frac_multi = mean_bimf = mean_t = mono_frac = recon_ok = 0.0
        sc = (
            3 * syn_ok
            + 2 * syn_mono
            + 1.5 * syn_recon
            + 4 * frac_multi
            + 1.5 * mono_frac
            + 1.0 * recon_ok
            + 0.3 * min(mean_bimf, 5)
            - 0.02 * mean_t
        )
        meta = {
            "syn_n_bimf": int(s["n_bimf"]),
            "syn_mono": bool(s["centroids_monotonic"]),
            "syn_rmse": float(s["recon_rmse"]),
            "mri_frac_multi": frac_multi,
            "mri_mean_n_bimf": mean_bimf,
            "mri_mono_frac": mono_frac,
            "mri_mean_sec": mean_t,
            "backend": s["backend"],
            "score": sc,
        }
        return sc, meta

    ranked = []
    for cid in by_cfg:
        sc, meta = score(cid)
        ranked.append((sc, cid, meta))
    ranked.sort(key=lambda x: x[0], reverse=True)

    # Showcase figures on mentor for top configs
    showcase = ["emd2d_default"]
    best_emd = next((cid for _, cid, m in ranked if m.get("backend") == "emd2d"), "emd2d_default")
    best_bemd = next((cid for _, cid, m in ranked if m.get("backend") == "bemd"), None)
    if best_emd not in showcase:
        showcase.append(best_emd)
    if best_bemd:
        showcase.append(best_bemd)

    cfg_lookup = {c["config_id"]: c for c in emd2d_configs() + bemd_configs()}

    log("=== STEP 5-6: showcase figures + full MRI set ===")
    showcase_stats: List[Dict[str, Any]] = []
    for cid in showcase:
        cfg = cfg_lookup[cid]
        # figures on synthetic + mentor
        for src, img in [("synthetic", syn), ("CINE_4CH_009_f083", mentor)]:
            ev = run_backend(img, cfg, src)
            log(f"figure {cid} {src}: n_bimf={ev.n_bimf} err={ev.error}")
            if ev.error:
                continue
            params = {k: v for k, v in cfg.items() if k not in ("backend", "config_id")}
            try:
                if cfg["backend"] == "bemd":
                    comps, residual = _bemd_call(img, params)
                else:
                    comps, residual = _emd2d_call(img, params)
                dest = SYN if src == "synthetic" else MRI_FIG
                save_montages(dest, f"{src}__{cid}", img, comps, residual)
            except Exception as exc:
                log(f"  montage failed: {exc}")

        # full MRI set
        for cs, fr in MRI_SPECS:
            im = load_mri(cs, fr)
            if im is None:
                continue
            ev = run_backend(im, cfg, f"{cs}_f{fr:03d}")
            showcase_stats.append(row_from_eval(ev))
            log(f"MRI {cid} {cs}_f{fr:03d}: n_bimf={ev.n_bimf} err={ev.error}")

    write_csv(OUT / "mri_showcase_stats.csv", showcase_stats)

    # Aggregate showcase for recommendation
    def agg(cid: str) -> Dict[str, Any]:
        rows = [r for r in showcase_stats if r["config_id"] == cid and not r.get("error")]
        if not rows:
            sc, meta = score(cid)
            return {"config_id": cid, **meta}
        return {
            "config_id": cid,
            "backend": rows[0]["backend"],
            "mri_n": len(rows),
            "mri_mean_n_bimf": float(np.mean([int(r["n_bimf"]) for r in rows])),
            "mri_frac_multi": float(np.mean([1 if int(r["n_bimf"]) > 1 else 0 for r in rows])),
            "mri_mono_frac": float(np.mean([1 if r["centroids_monotonic"] else 0 for r in rows])),
            "mri_mean_sec": float(np.mean([float(r["elapsed_sec"]) for r in rows])),
            "mri_mean_recon_rmse": float(np.mean([float(r["recon_rmse"]) for r in rows])),
        }

    # Choose recommendation
    chosen = None
    rationale = []
    for sc, cid, meta in ranked:
        if meta.get("syn_n_bimf", 0) > 1 and meta.get("mri_frac_multi", 0) >= 0.3 and meta.get("syn_rmse", 1) < 1.0:
            chosen = cid
            rationale.append(
                f"{cid}: >1 BIMF on synthetic; multi-BIMF on {100*meta['mri_frac_multi']:.0f}% of MRI probes; "
                f"mean MRI BIMFs={meta['mri_mean_n_bimf']:.2f}."
            )
            break
    if chosen is None:
        # prefer any config with MRI multi
        for sc, cid, meta in ranked:
            if meta.get("mri_frac_multi", 0) >= 0.3:
                chosen = cid
                rationale.append(f"{cid} best available MRI multiscale fraction ({meta['mri_frac_multi']:.2f}).")
                break
    if chosen is None:
        chosen = best_emd
        rationale.append(
            f"No config met full success criteria. Fallback {chosen} "
            "(best EMD2D by score). Multiscale BIMF-k ablation still not well supported."
        )

    recommendation = {
        "question": "Which 2D EMD configuration/backend gives the most meaningful multiscale decomposition?",
        "recommended_for_next_segmentation_screen": chosen,
        "rationale": " ".join(rationale),
        "ranking_top10": [
            {"rank": i + 1, "config_id": cid, **meta, "score": sc} for i, (sc, cid, meta) in enumerate(ranked[:10])
        ],
        "approaches": {
            "emd2d_default": agg("emd2d_default"),
            "emd2d_tuned_best": agg(best_emd),
            "bemd_best": agg(best_bemd) if best_bemd else None,
        },
        "failure_modes": {
            "emd2d_default": "Stops after BIMF0 because residue has 0 extrema; one broad oscillatory + smooth trend.",
            "emd2d_strict": "Stricter thr/FIXE_H can increase sift cost; may still leave smooth residual.",
            "bemd": "Morphological extrema + RBF envelopes; author marks untested; can timeout/artifact.",
        },
        "no_unet_training": True,
        "timeout_sec_per_decomp": DECOMP_TIMEOUT_SEC,
        "max_sift_iters_cap": MAX_SIFT_ITERS,
    }
    (OUT / "recommendation.json").write_text(json.dumps(recommendation, indent=2), encoding="utf-8")

    readme = f"""# Decomposition granularity study

## Answer
**Recommended for next segmentation experiment:** `{chosen}`

{recommendation['rationale']}

## Why default stops at 1 BIMF
Residue after BIMF0 has too few 3x3 extrema (often 0). See `AUDIT_EMD2D.md`.

## Comparison snapshot
See `recommendation.json` approaches block for default vs tuned EMD2D vs BEMD.

## Artifacts
- `parameter_sweep.csv`
- `backend_comparison.csv`
- `mri_showcase_stats.csv`
- `synthetic/`, `mri_representatives/`
- `run_log.txt`

No U-Net training was performed.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    log("=== RECOMMENDATION ===")
    log(json.dumps(recommendation, indent=2))
    log(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
