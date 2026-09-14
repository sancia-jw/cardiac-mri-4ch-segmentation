"""
True bidimensional EMD (PyEMD BEMD) with square zero-pad / crop-back.

Protocol ``bemd_default_square_pad`` (from decomposition_granularity recommendation):
  1. zero-pad native (H, W) to square max(H, W)
  2. run PyEMD.BEMD with default-ish params (FIXE=1, mean_thr=0.01, mse_thr=0.01)
  3. crop each BIMF + residual back to native FOV

Does **not** use flattened 1D EMD or EMD2D.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.preprocessing.emd_enhancement import (
    as_grayscale_slice,
    safe_minmax_normalize,
)
from src.preprocessing.emd_geometry import (
    component_energy_stats,
    spectral_centroid_cycles_per_pixel,
)

METHOD_ID = "bemd_default_square_pad"


@dataclass
class BEMDConfig:
    """Settings for the recommended square-pad BEMD backend."""

    method_id: str = METHOD_ID
    max_imf: int = 4
    mean_thr: float = 0.01
    mse_thr: float = 0.01
    FIXE: int = 1
    FIXE_H: int = 0
    MAX_ITERATION: int = 8
    # Gastro-compatible subtract: min-max each BIMF before subtracting from raw.
    normalize_bimfs: bool = True
    clip_output: bool = True


@dataclass
class BEMDDecomposition:
    original: np.ndarray  # native FOV float32
    bimfs: List[np.ndarray]  # oscillatory, highest-frequency-first, native FOV
    residual: np.ndarray
    padded_shape: Tuple[int, int]
    pad_hw: Tuple[int, int]  # (pad_h, pad_w) added on bottom/right
    elapsed_sec: float
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_bimf(self) -> int:
        return len(self.bimfs)

    def reconstruction_error(self) -> Dict[str, float]:
        parts = list(self.bimfs) + [self.residual]
        recon = np.sum(np.stack([p.astype(np.float64) for p in parts], axis=0), axis=0)
        diff = self.original.astype(np.float64) - recon
        return {
            "max_abs": float(np.max(np.abs(diff))),
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff**2))),
        }


def square_zero_pad(image_2d: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Zero-pad (H, W) to square max(H, W) on bottom/right.

    Returns (padded, original_hw, pad_hw).
    """
    img = as_grayscale_slice(image_2d)
    h, w = int(img.shape[0]), int(img.shape[1])
    side = max(h, w)
    out = np.zeros((side, side), dtype=np.float64)
    out[:h, :w] = img.astype(np.float64)
    return out, (h, w), (side - h, side - w)


def crop_to_hw(arr: np.ndarray, hw: Tuple[int, int], *, dtype=np.float64) -> np.ndarray:
    """
    Crop padded component back to native FOV.

    Keep float64 by default: BIMF/residual slabs can be huge with near-cancelling
    amplitudes; casting each slab to float32 *before* summing breaks reconstruction.
    """
    h, w = hw
    return np.asarray(arr[:h, :w], dtype=dtype)


def pyemd_version() -> str:
    try:
        import PyEMD

        return str(getattr(PyEMD, "__version__", "unknown"))
    except Exception:
        return "unknown"


def decompose_bemd_square_pad(
    image_2d: np.ndarray,
    config: Optional[BEMDConfig] = None,
) -> BEMDDecomposition:
    """
    Run ``bemd_default_square_pad`` and return native-FOV components.

    Raises if BEMD fails (e.g. without pad on non-square — we always pad).
    """
    import time

    from PyEMD.BEMD import BEMD

    cfg = config or BEMDConfig()
    orig = as_grayscale_slice(image_2d)
    padded, hw, pad_hw = square_zero_pad(orig)

    bemd = BEMD()
    bemd.mean_thr = float(cfg.mean_thr)
    bemd.mse_thr = float(cfg.mse_thr)
    bemd.FIXE = int(cfg.FIXE)
    bemd.FIXE_H = int(cfg.FIXE_H)
    bemd.MAX_ITERATION = int(cfg.MAX_ITERATION)

    t0 = time.perf_counter()
    imfs = bemd.bemd(padded, max_imf=int(cfg.max_imf))
    elapsed = time.perf_counter() - t0

    if imfs.ndim != 3 or imfs.shape[0] < 1:
        raise RuntimeError(f"BEMD returned unexpected shape {getattr(imfs, 'shape', None)}")

    n = int(imfs.shape[0])
    if n == 1:
        bimfs = [crop_to_hw(imfs[0], hw, dtype=np.float64)]
        residual = np.zeros(hw, dtype=np.float64)
    else:
        # Last slab is residual/trend; earlier slabs are oscillatory BIMFs.
        bimfs = [crop_to_hw(imfs[i], hw, dtype=np.float64) for i in range(n - 1)]
        residual = crop_to_hw(imfs[-1], hw, dtype=np.float64)

    decomp = BEMDDecomposition(
        # Keep native intensity as float64 for exact recon checks; enhance casts later.
        original=orig.astype(np.float64, copy=False),
        bimfs=bimfs,
        residual=residual,
        padded_shape=(int(padded.shape[0]), int(padded.shape[1])),
        pad_hw=pad_hw,
        elapsed_sec=float(elapsed),
        meta={
            "method_id": cfg.method_id,
            "max_imf": cfg.max_imf,
            "mean_thr": cfg.mean_thr,
            "mse_thr": cfg.mse_thr,
            "FIXE": cfg.FIXE,
            "FIXE_H": cfg.FIXE_H,
            "MAX_ITERATION": cfg.MAX_ITERATION,
            "n_returned_slabs": n,
            "library": "EMD-signal / PyEMD.BEMD",
            "library_version": pyemd_version(),
            "author_caveat": "BEMD marked experimental / lightly tested by PyEMD authors",
        },
    )
    decomp.meta["reconstruction"] = decomp.reconstruction_error()
    return decomp


def component_characterization(decomp: BEMDDecomposition) -> Dict[str, Any]:
    """Spectral / energy stats for audit (does not affect preprocessing)."""
    rows = []
    for i, bimf in enumerate(decomp.bimfs):
        stats = component_energy_stats(decomp.original, bimf)
        rows.append(
            {
                "name": f"bimf_{i}",
                "spectral_centroid_cpp": spectral_centroid_cycles_per_pixel(bimf),
                **stats,
            }
        )
    rstats = component_energy_stats(decomp.original, decomp.residual)
    rows.append(
        {
            "name": "residual",
            "spectral_centroid_cpp": spectral_centroid_cycles_per_pixel(decomp.residual),
            **rstats,
        }
    )
    return {"n_bimf": decomp.n_bimf, "components": rows, "reconstruction": decomp.reconstruction_error()}


def enhance_from_bemd_decomp(
    decomp: BEMDDecomposition,
    *,
    mode: str,
    bimf_indices: Sequence[int],
    normalize_bimfs: bool = True,
    clip_output: bool = True,
) -> np.ndarray:
    """
    Build model input from a cached / live BEMD decomposition.

    Matches Gastro-style subtract used in ``emd_enhancement.enhance_mri_slice``:
      processed = original - sum(selected BIMFs [optionally min-max each])
      then per-slice min-max finalize to [0, 1].

    Modes: ``original`` | ``subtract``.
    """
    if mode == "original":
        return safe_minmax_normalize(decomp.original, clip=clip_output)

    if mode != "subtract":
        raise ValueError(f"Unsupported BEMD enhance mode: {mode}")

    if not bimf_indices:
        raise ValueError("subtract mode requires bimf_indices")

    selected: List[np.ndarray] = []
    for idx in bimf_indices:
        if idx >= decomp.n_bimf or idx < -decomp.n_bimf:
            raise IndexError(
                f"BIMF index {idx} out of range for n_bimf={decomp.n_bimf} "
                f"(need indices covering requested ablation conditions)."
            )
        comp = decomp.bimfs[idx].astype(np.float32, copy=False)
        if normalize_bimfs:
            comp = safe_minmax_normalize(comp, clip=False)
        selected.append(comp)

    removed = np.zeros_like(decomp.original, dtype=np.float32)
    for c in selected:
        removed = removed + c
    processed = (decomp.original.astype(np.float32) - removed).astype(np.float32)
    return safe_minmax_normalize(processed, clip=clip_output)


def config_dict(cfg: BEMDConfig) -> Dict[str, Any]:
    return asdict(cfg)
