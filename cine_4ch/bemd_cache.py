"""
Persistent on-disk cache for ``bemd_default_square_pad`` decompositions.

Layout
------
outputs/bemd_cache/bemd_default_square_pad/<case_stem>/frame_XXX/
    bimf_0.npy ... bimf_K.npy
    residual.npy
    original.npy
    metadata.json

Valid entries are never deleted automatically. Incomplete/corrupt entries are
detected and recomputed when preprocess runs (unless they validate).
"""

from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from cine_4ch.config import OUTPUTS_DIR
from cine_4ch.io import CasePair, extract_frame, load_pair
from src.preprocessing.bemd_square_pad import (
    METHOD_ID,
    BEMDConfig,
    BEMDDecomposition,
    component_characterization,
    pyemd_version,
)

BEMD_CACHE_ROOT = OUTPUTS_DIR / "bemd_cache" / METHOD_ID

# Soft reconstruction gate (native intensity units).
# With float64 components, padded-domain recon is ~1e-12; cropped sum matches original.
RECON_RMSE_WARN = 1e-6
RECON_RMSE_FAIL = 1e-3


def frame_cache_dir(case_stem: str, frame_idx: int, cache_root: Path = BEMD_CACHE_ROOT) -> Path:
    return cache_root / case_stem / f"frame_{frame_idx:03d}"


def _meta_path(d: Path) -> Path:
    return d / "metadata.json"


def is_valid_cache_entry(
    case_stem: str,
    frame_idx: int,
    *,
    cache_root: Path = BEMD_CACHE_ROOT,
    require_n_bimf: Optional[int] = None,
) -> bool:
    d = frame_cache_dir(case_stem, frame_idx, cache_root)
    meta_p = _meta_path(d)
    if not meta_p.is_file():
        return False
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    except Exception:
        return False
    if meta.get("method_id") != METHOD_ID:
        return False
    if meta.get("status") not in ("ok", "ok_warn_recon"):
        return False
    n = int(meta.get("n_bimf", -1))
    if n < 1:
        return False
    if require_n_bimf is not None and n < require_n_bimf:
        return False
    needed = [d / f"bimf_{i}.npy" for i in range(n)] + [d / "residual.npy", d / "original.npy"]
    if not all(p.is_file() for p in needed):
        return False
    recon = meta.get("reconstruction") or {}
    rmse = recon.get("rmse")
    if rmse is None or float(rmse) > RECON_RMSE_FAIL:
        return False
    return True


def load_decomposition(
    case_stem: str,
    frame_idx: int,
    *,
    cache_root: Path = BEMD_CACHE_ROOT,
) -> BEMDDecomposition:
    d = frame_cache_dir(case_stem, frame_idx, cache_root)
    meta = json.loads(_meta_path(d).read_text(encoding="utf-8"))
    n = int(meta["n_bimf"])
    # Prefer float64 for reconstruction fidelity; enhance path casts as needed.
    bimfs = [np.load(d / f"bimf_{i}.npy") for i in range(n)]
    residual = np.load(d / "residual.npy")
    original = np.load(d / "original.npy")
    return BEMDDecomposition(
        original=original,
        bimfs=bimfs,
        residual=residual,
        padded_shape=tuple(meta["padded_shape"]),
        pad_hw=tuple(meta["pad_hw"]),
        elapsed_sec=float(meta.get("elapsed_sec", 0.0)),
        meta=meta,
    )


def save_decomposition(
    case_stem: str,
    frame_idx: int,
    decomp: BEMDDecomposition,
    *,
    cache_root: Path = BEMD_CACHE_ROOT,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    d = frame_cache_dir(case_stem, frame_idx, cache_root)
    d.mkdir(parents=True, exist_ok=True)
    for i, bimf in enumerate(decomp.bimfs):
        np.save(d / f"bimf_{i}.npy", np.asarray(bimf, dtype=np.float64))
    np.save(d / "residual.npy", np.asarray(decomp.residual, dtype=np.float64))
    np.save(d / "original.npy", np.asarray(decomp.original, dtype=np.float64))

    char = component_characterization(decomp)
    recon = decomp.reconstruction_error()
    status = "ok"
    if recon["rmse"] > RECON_RMSE_FAIL:
        status = "bad_recon"
    elif recon["rmse"] > RECON_RMSE_WARN:
        status = "ok_warn_recon"

    meta: Dict[str, Any] = {
        "status": status,
        "method_id": METHOD_ID,
        "case_stem": case_stem,
        "frame_idx": int(frame_idx),
        "original_shape": list(decomp.original.shape),
        "padded_shape": list(decomp.padded_shape),
        "pad_hw": list(decomp.pad_hw),
        "padding": "zero_bottom_right",
        "n_bimf": decomp.n_bimf,
        "elapsed_sec": decomp.elapsed_sec,
        "reconstruction": recon,
        "component_stats": char["components"],
        "bemd_settings": {
            k: decomp.meta.get(k)
            for k in ("max_imf", "mean_thr", "mse_thr", "FIXE", "FIXE_H", "MAX_ITERATION")
        },
        "library": decomp.meta.get("library"),
        "library_version": decomp.meta.get("library_version", pyemd_version()),
        "component_dtype": "float64",
        "component_dtype_note": (
            "Components stored as float64 so sum(BIMFs)+residual reconstructs "
            "original after crop; float32-per-slab breaks large cancelling amplitudes."
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra_meta:
        meta.update(extra_meta)
    # Atomic-ish write: write temp then replace
    tmp = d / "metadata.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(_meta_path(d))
    return d


def _worker_one(args: tuple) -> Dict[str, Any]:
    """Process-pool worker for one (case_stem, image_path, frame_idx)."""
    case_stem, image_path, frame_idx, cache_root_s, force, cfg_dict = args
    from cine_4ch.io import extract_frame, load_volume
    from src.preprocessing.bemd_square_pad import BEMDConfig, decompose_bemd_square_pad

    cache_root = Path(cache_root_s)
    if (not force) and is_valid_cache_entry(case_stem, frame_idx, cache_root=cache_root):
        meta = json.loads(_meta_path(frame_cache_dir(case_stem, frame_idx, cache_root)).read_text(encoding="utf-8"))
        comps = meta.get("component_stats") or []
        bimf_comps = [c for c in comps if str(c.get("name", "")).startswith("bimf_")]
        return {
            "case_stem": case_stem,
            "frame_idx": frame_idx,
            "status": "skipped_valid",
            "n_bimf": meta.get("n_bimf"),
            "recon_rmse": (meta.get("reconstruction") or {}).get("rmse"),
            "elapsed_sec": 0.0,
            "centroids": [c.get("spectral_centroid_cpp") for c in bimf_comps],
            "energy_fracs": [c.get("energy_fraction_of_original") for c in bimf_comps],
        }

    try:
        from src.preprocessing.bemd_square_pad import component_characterization as _char

        image, _ = load_volume(Path(image_path))
        raw = extract_frame(image, frame_idx)
        # Only pass BEMD solver fields into decompose config.
        decomp_keys = ("max_imf", "mean_thr", "mse_thr", "FIXE", "FIXE_H", "MAX_ITERATION", "method_id")
        decomp = decompose_bemd_square_pad(
            raw,
            BEMDConfig(**{k: cfg_dict[k] for k in decomp_keys if k in cfg_dict}),
        )
        save_decomposition(case_stem, frame_idx, decomp, cache_root=cache_root)
        recon = decomp.reconstruction_error()
        char = _char(decomp)
        bimf_comps = [c for c in char["components"] if str(c["name"]).startswith("bimf_")]
        return {
            "case_stem": case_stem,
            "frame_idx": frame_idx,
            "status": "ok" if recon["rmse"] <= RECON_RMSE_FAIL else "bad_recon",
            "n_bimf": decomp.n_bimf,
            "recon_rmse": recon["rmse"],
            "elapsed_sec": decomp.elapsed_sec,
            "centroids": [c["spectral_centroid_cpp"] for c in bimf_comps],
            "energy_fracs": [c["energy_fraction_of_original"] for c in bimf_comps],
        }
    except Exception as exc:
        # Record failure sidecar without wiping siblings
        d = frame_cache_dir(case_stem, frame_idx, cache_root)
        d.mkdir(parents=True, exist_ok=True)
        fail = {
            "status": "error",
            "method_id": METHOD_ID,
            "case_stem": case_stem,
            "frame_idx": frame_idx,
            "error": f"{type(exc).__name__}: {exc}",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        (d / "metadata.json").write_text(json.dumps(fail, indent=2), encoding="utf-8")
        return {
            "case_stem": case_stem,
            "frame_idx": frame_idx,
            "status": "error",
            "error": fail["error"],
            "n_bimf": None,
            "recon_rmse": None,
            "elapsed_sec": None,
        }


def iter_case_frames(cases: Sequence[CasePair]) -> List[Tuple[str, str, int]]:
    """Return list of (case_stem, image_path, frame_idx)."""
    jobs = []
    for case in cases:
        _, label, _ = load_pair(case)
        n_frames = 1 if label.ndim < 3 else int(label.shape[-1])
        for fidx in range(n_frames):
            jobs.append((case.stem, str(case.image_path), fidx))
    return jobs


def preprocess_cases(
    cases: Sequence[CasePair],
    *,
    cache_root: Path = BEMD_CACHE_ROOT,
    config: Optional[BEMDConfig] = None,
    force: bool = False,
    workers: int = 1,
    max_frames: Optional[int] = None,
    start_index: int = 0,
    end_index: Optional[int] = None,
    audit_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Decompose and cache BEMD for all frames in ``cases``.

    Saves incrementally per frame. Safe to resume (skips valid cache).
    """
    cfg = config or BEMDConfig()
    cache_root.mkdir(parents=True, exist_ok=True)
    jobs = iter_case_frames(cases)
    if start_index or end_index is not None:
        jobs = jobs[start_index:end_index]
    if max_frames is not None:
        jobs = jobs[: max(0, int(max_frames))]

    work = [
        (stem, ipath, fidx, str(cache_root), bool(force), asdict(cfg))
        for stem, ipath, fidx in jobs
    ]

    rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    if workers <= 1:
        for args in tqdm(work, desc="bemd preprocess"):
            rows.append(_worker_one(args))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_worker_one, args) for args in work]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"bemd preprocess x{workers}"):
                rows.append(fut.result())

    # Stable sort for audit
    rows.sort(key=lambda r: (r.get("case_stem") or "", int(r.get("frame_idx") or -1)))

    n_bimf_vals = [int(r["n_bimf"]) for r in rows if r.get("n_bimf") is not None]
    short = [r for r in rows if r.get("n_bimf") is not None and int(r["n_bimf"]) < 4]
    errors = [r for r in rows if r.get("status") == "error"]
    bad = [r for r in rows if r.get("status") == "bad_recon"]

    summary = {
        "method_id": METHOD_ID,
        "cache_root": str(cache_root),
        "n_jobs": len(work),
        "n_ok": sum(1 for r in rows if r.get("status") in ("ok", "skipped_valid", "ok_warn_recon")),
        "n_skipped_valid": sum(1 for r in rows if r.get("status") == "skipped_valid"),
        "n_errors": len(errors),
        "n_bad_recon": len(bad),
        "n_bimf_lt_4": len(short),
        "n_bimf_min": int(min(n_bimf_vals)) if n_bimf_vals else None,
        "n_bimf_median": float(np.median(n_bimf_vals)) if n_bimf_vals else None,
        "n_bimf_max": int(max(n_bimf_vals)) if n_bimf_vals else None,
        "n_bimf_histogram": {
            str(k): int(v)
            for k, v in sorted(
                {n: n_bimf_vals.count(n) for n in set(n_bimf_vals)}.items()
            )
        },
        "wall_sec": time.perf_counter() - t0,
        "workers": workers,
        "force": force,
        "frames_with_lt_4_bimfs": [
            {"case_stem": r["case_stem"], "frame_idx": r["frame_idx"], "n_bimf": r["n_bimf"]}
            for r in short
        ],
        "errors": [
            {"case_stem": r["case_stem"], "frame_idx": r["frame_idx"], "error": r.get("error")}
            for r in errors
        ],
    }

    if audit_csv is not None:
        audit_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "case_stem",
            "frame_idx",
            "status",
            "n_bimf",
            "recon_rmse",
            "elapsed_sec",
            "centroids",
            "energy_fracs",
            "error",
        ]
        with open(audit_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                out = dict(r)
                if isinstance(out.get("centroids"), list):
                    out["centroids"] = json.dumps(out["centroids"])
                if isinstance(out.get("energy_fracs"), list):
                    out["energy_fracs"] = json.dumps(out["energy_fracs"])
                w.writerow(out)

    summary_path = cache_root / "preprocess_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def default_workers() -> int:
    cpu = os.cpu_count() or 2
    # BEMD is heavy; stay conservative (Kaggle-friendly).
    return max(1, min(4, cpu - 1))
