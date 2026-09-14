"""
Geometry-aware EMD helpers for the 4CH follow-up experiment.

Backends
--------
- ``1d_raster`` : existing Holman ``emd.sift.sift`` on a flattened image
  (``flatten_order='C'`` = row-major, ``'F'`` = column-major).
- ``emd2d_pyemd`` : PyEMD ``EMD2D`` (true 2D extrema + bivariate envelopes).
  Package: ``EMD-signal`` (Apache-2.0). Author marks the module experimental.

This module does **not** claim that 1D raster EMD is bidimensional EMD.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from src.preprocessing.emd_enhancement import (
    as_grayscale_slice,
    safe_minmax_normalize,
)

FlattenOrder = Literal["C", "F"]
DecompBackend = Literal["1d_raster", "emd2d_pyemd"]


@dataclass
class GeometryEMDConfig:
    """Decomposition settings for geometry follow-up (separate from Gastro training config)."""

    backend: DecompBackend = "1d_raster"
    flatten_order: FlattenOrder = "C"
    sift_thresh: float = 1e-8
    # PyEMD EMD2D: negative = all; 1 = stop after first oscillatory BIMF (+ residual appended).
    max_imf: int = 1
    normalize_components: bool = False  # raw components for science metrics / fair subtract


@dataclass
class DecompositionResult:
    original: np.ndarray
    components: List[np.ndarray]  # oscillatory IMFs / BIMFs, highest-frequency first
    residual: np.ndarray
    backend: str
    flatten_order: Optional[str]
    elapsed_sec: float
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_oscillatory(self) -> int:
        return len(self.components)

    def all_including_residual(self) -> List[np.ndarray]:
        return list(self.components) + [self.residual]


def _reconstruction_errors(original: np.ndarray, parts: Sequence[np.ndarray]) -> Dict[str, float]:
    recon = np.sum(np.stack([np.asarray(p, dtype=np.float64) for p in parts], axis=0), axis=0)
    diff = original.astype(np.float64) - recon
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
    }


def component_energy_stats(original: np.ndarray, component: np.ndarray) -> Dict[str, float]:
    """Quantify how much signal a component carries relative to the original."""
    orig = original.astype(np.float64)
    comp = component.astype(np.float64)
    orig_energy = float(np.sum(orig**2))
    comp_energy = float(np.sum(comp**2))
    return {
        "rms": float(np.sqrt(np.mean(comp**2))),
        "variance": float(np.var(comp)),
        "energy": comp_energy,
        "energy_fraction_of_original": comp_energy / max(orig_energy, 1e-12),
        "mean_abs": float(np.mean(np.abs(comp))),
        "max_abs": float(np.max(np.abs(comp))),
    }


def decompose_1d_raster(
    image_2d: np.ndarray,
    *,
    flatten_order: FlattenOrder = "C",
    sift_thresh: float = 1e-8,
    normalize_components: bool = False,
) -> DecompositionResult:
    """1D EMD on row-major (C) or column-major (F) flattened image."""
    import emd

    slice_2d = as_grayscale_slice(image_2d)
    t0 = time.perf_counter()
    signal_1d = slice_2d.reshape(-1, order=flatten_order)
    all_imfs = emd.sift.sift(signal_1d, sift_thresh=sift_thresh)
    n_cols = int(all_imfs.shape[1])
    if n_cols < 2:
        raise RuntimeError(f"1D sift returned <2 components ({n_cols})")

    comps: List[np.ndarray] = []
    for i in range(n_cols - 1):
        imf = all_imfs[:, i].reshape(slice_2d.shape, order=flatten_order).astype(np.float32)
        if normalize_components:
            imf = safe_minmax_normalize(imf, clip=False)
        comps.append(imf)
    residual = all_imfs[:, -1].reshape(slice_2d.shape, order=flatten_order).astype(np.float32)
    if normalize_components:
        residual = safe_minmax_normalize(residual, clip=False)

    elapsed = time.perf_counter() - t0
    parts = comps + [residual]
    return DecompositionResult(
        original=slice_2d,
        components=comps,
        residual=residual,
        backend="1d_raster",
        flatten_order=flatten_order,
        elapsed_sec=elapsed,
        meta={
            "sift_thresh": sift_thresh,
            "n_sift_columns": n_cols,
            "reconstruction": _reconstruction_errors(slice_2d, parts),
            "emd_package": "emd",
        },
    )


def decompose_emd2d_pyemd(
    image_2d: np.ndarray,
    *,
    max_imf: int = 1,
    normalize_components: bool = False,
) -> DecompositionResult:
    """
    True 2D EMD via PyEMD ``EMD2D``.

    Returns oscillatory BIMFs (highest-frequency first) and a residual/trend.
    With ``max_imf=1``, typically yields BIMF0 + residual.
    """
    try:
        from PyEMD.EMD2d import EMD2D
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyEMD EMD2D requires: pip install EMD-signal"
        ) from exc

    slice_2d = as_grayscale_slice(image_2d)
    # EMD2D internally rescales; pass float64 for numerical stability.
    img = slice_2d.astype(np.float64, copy=False)

    t0 = time.perf_counter()
    imfs = EMD2D()(img, max_imf=max_imf)
    elapsed = time.perf_counter() - t0

    if imfs.ndim != 3 or imfs.shape[0] < 1:
        raise RuntimeError(f"EMD2D returned unexpected shape {getattr(imfs, 'shape', None)}")

    # Convention: last slab is residual when reconstruction includes it.
    # EMD2D appends residual if nonzero remainder remains.
    n = int(imfs.shape[0])
    if n == 1:
        comps = [imfs[0].astype(np.float32)]
        residual = np.zeros_like(slice_2d, dtype=np.float32)
    else:
        comps = [imfs[i].astype(np.float32) for i in range(n - 1)]
        residual = imfs[-1].astype(np.float32)

    if normalize_components:
        comps = [safe_minmax_normalize(c, clip=False) for c in comps]
        residual = safe_minmax_normalize(residual, clip=False)

    parts = comps + [residual]
    try:
        import PyEMD

        pyemd_ver = getattr(PyEMD, "__version__", "unknown")
    except Exception:
        pyemd_ver = "unknown"

    return DecompositionResult(
        original=slice_2d,
        components=comps,
        residual=residual,
        backend="emd2d_pyemd",
        flatten_order=None,
        elapsed_sec=elapsed,
        meta={
            "max_imf": max_imf,
            "n_returned_slabs": n,
            "reconstruction": _reconstruction_errors(slice_2d, parts),
            "library": "EMD-signal / PyEMD.EMD2d.EMD2D",
            "library_version": pyemd_ver,
            "license": "Apache-2.0",
            "author_caveat": "EMD2D marked experimental by PyEMD authors",
        },
    )


def decompose(image_2d: np.ndarray, config: GeometryEMDConfig) -> DecompositionResult:
    if config.backend == "1d_raster":
        return decompose_1d_raster(
            image_2d,
            flatten_order=config.flatten_order,
            sift_thresh=config.sift_thresh,
            normalize_components=config.normalize_components,
        )
    if config.backend == "emd2d_pyemd":
        return decompose_emd2d_pyemd(
            image_2d,
            max_imf=config.max_imf,
            normalize_components=config.normalize_components,
        )
    raise ValueError(f"Unknown backend: {config.backend}")


def subtract_finest_component(
    image_2d: np.ndarray,
    config: GeometryEMDConfig,
    *,
    use_normalized_component_for_subtract: bool = True,
) -> Tuple[np.ndarray, DecompositionResult, np.ndarray]:
    """
    Return (processed_raw_space, decomp, finest_component_used_in_subtract).

    If ``use_normalized_component_for_subtract`` is True (Gastro training convention),
    the finest oscillatory component is min-max normalized before subtraction from the
    raw image. Metrics should still use the raw component from ``decomp``.
    """
    # Always decompose in raw space for interpretable metrics.
    cfg = GeometryEMDConfig(**{**asdict(config), "normalize_components": False})
    decomp = decompose(image_2d, cfg)
    if not decomp.components:
        raise RuntimeError("No oscillatory component available to subtract.")
    finest_raw = decomp.components[0]
    if use_normalized_component_for_subtract:
        finest_sub = safe_minmax_normalize(finest_raw, clip=False)
    else:
        finest_sub = finest_raw
    processed = (decomp.original - finest_sub).astype(np.float32)
    return processed, decomp, finest_sub


def display_normalize(image_2d: np.ndarray) -> np.ndarray:
    return safe_minmax_normalize(as_grayscale_slice(image_2d), clip=True)


def scale_component_to_target_energy(
    component: np.ndarray,
    *,
    target_energy: float,
) -> Tuple[np.ndarray, float]:
    """
    Scale a component so sum(component**2) == target_energy.

    Returns (scaled_component, scale_factor).
    """
    comp = np.asarray(component, dtype=np.float64)
    energy = float(np.sum(comp**2))
    if energy <= 1e-12 or target_energy <= 0:
        return np.zeros_like(comp, dtype=np.float32), 0.0
    scale = float(np.sqrt(target_energy / energy))
    return (comp * scale).astype(np.float32), scale


def matched_energy_subtract(
    original: np.ndarray,
    component: np.ndarray,
    *,
    target_energy: float,
    use_normalized_component_for_subtract: bool = False,
) -> Dict[str, Any]:
    """
    Subtract a component after scaling it to ``target_energy`` (raw intensity space).

    For fair geometry comparisons, prefer ``use_normalized_component_for_subtract=False``
    so energy matching is meaningful. Gastro-normalized subtract is reported separately.
    """
    orig = as_grayscale_slice(original)
    scaled, scale = scale_component_to_target_energy(component, target_energy=target_energy)
    if use_normalized_component_for_subtract:
        sub = safe_minmax_normalize(scaled, clip=False)
    else:
        sub = scaled
    processed = (orig.astype(np.float64) - sub.astype(np.float64)).astype(np.float32)
    return {
        "processed": processed,
        "scaled_component": scaled,
        "scale_factor": scale,
        "target_energy": float(target_energy),
        "scaled_energy": float(np.sum(scaled.astype(np.float64) ** 2)),
        "energy_stats": component_energy_stats(orig, scaled),
    }


def fourier_highfreq_component(
    image_2d: np.ndarray,
    *,
    cutoff_cycles_per_pixel: float,
) -> np.ndarray:
    """
    Geometry-respecting high-frequency residual via ideal Fourier high-pass.

    ``cutoff_cycles_per_pixel`` is a radial frequency threshold in cycles/pixel
    (0.5 = Nyquist). Returns the high-pass residual in the original intensity space.
    """
    img = as_grayscale_slice(image_2d).astype(np.float64)
    h, w = img.shape
    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(xx**2 + yy**2)
    mask = radius >= float(cutoff_cycles_per_pixel)
    spec = np.fft.fft2(img)
    high = np.fft.ifft2(spec * mask).real
    return high.astype(np.float32)


def match_fourier_cutoff_to_energy(
    image_2d: np.ndarray,
    *,
    target_energy: float,
    cutoff_grid: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Find a Fourier high-pass cutoff whose residual energy is closest to ``target_energy``.
    """
    if cutoff_grid is None:
        cutoff_grid = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.28, 0.35]

    best: Optional[Dict[str, Any]] = None
    for cut in cutoff_grid:
        comp = fourier_highfreq_component(image_2d, cutoff_cycles_per_pixel=float(cut))
        energy = float(np.sum(comp.astype(np.float64) ** 2))
        row = {
            "cutoff_cycles_per_pixel": float(cut),
            "component": comp,
            "energy": energy,
            "abs_energy_error": abs(energy - target_energy),
            "energy_fraction_of_original": component_energy_stats(image_2d, comp)[
                "energy_fraction_of_original"
            ],
        }
        if best is None or row["abs_energy_error"] < best["abs_energy_error"]:
            best = row
    assert best is not None
    return best


def spectral_centroid_cycles_per_pixel(component: np.ndarray) -> float:
    """Energy-weighted radial FFT centroid in cycles/pixel (Nyquist=0.5)."""
    img = np.asarray(component, dtype=np.float64)
    h, w = img.shape
    spec = np.fft.fftshift(np.abs(np.fft.fft2(img)) ** 2)
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(xx**2 + yy**2)
    total = float(np.sum(spec))
    if total <= 1e-12:
        return 0.0
    return float(np.sum(radius * spec) / total)


def simple_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Lightweight SSIM (single-window global stats; no scikit-image dependency)."""
    x = as_grayscale_slice(a).astype(np.float64)
    y = as_grayscale_slice(b).astype(np.float64)
    # Stabilize over intensity scale of the pair.
    data_range = float(max(np.max(x) - np.min(x), np.max(y) - np.min(y), 1e-6))
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mx, my = float(np.mean(x)), float(np.mean(y))
    vx, vy = float(np.var(x)), float(np.var(y))
    cov = float(np.mean((x - mx) * (y - my)))
    num = (2 * mx * my + c1) * (2 * cov + c2)
    den = (mx**2 + my**2 + c1) * (vx + vy + c2)
    return float(num / max(den, 1e-12))


def enhance_geometry_matched(
    image_2d: np.ndarray,
    mode: Literal[
        "original",
        "1d_row_imf0_raw",
        "1d_col_imf0_matched",
        "2d_bimf0_matched",
        "fft_hp_matched",
        "2d_subtract_bimf0",
        "2d_subtract_residual",
        "2d_residual_matched",
    ],
    *,
    sift_thresh: float = 1e-8,
    clip_output: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Stage-C geometry preprocess with **raw** energy-matched subtraction.

    All non-original modes subtract in raw intensity space (not Gastro IMF
    min-max normalize), then finalize with per-slice min-max to ``[0, 1]``.
    Matched modes scale their component to the **1D-row IMF0 energy** of the
    same slice before subtraction (same convention as Stage C ``2d_bimf0_matched``).

    Component-sweep extensions (true 2D EMD2D, ``max_imf=-1``):
    - ``2d_subtract_bimf0`` / ``2d_subtract_residual``: raw one-at-a-time removal
    - ``2d_residual_matched``: residual scaled to 1D-row IMF0 energy, then subtract
    - ``2d_bimf0_matched``: unchanged Stage-C path (``max_imf=1``; BIMF0 identical
      on this dataset where full decomp yields exactly one oscillatory BIMF)
    """
    orig = as_grayscale_slice(image_2d)
    meta: Dict[str, Any] = {"mode": mode}

    if mode == "original":
        out = safe_minmax_normalize(orig, clip=clip_output)
        meta["energy_fraction_removed"] = 0.0
        return out, meta

    row = decompose_1d_raster(
        orig, flatten_order="C", sift_thresh=sift_thresh, normalize_components=False
    )
    target_energy = float(np.sum(row.components[0].astype(np.float64) ** 2))
    meta["target_energy"] = target_energy
    meta["target_energy_fraction"] = component_energy_stats(orig, row.components[0])[
        "energy_fraction_of_original"
    ]

    if mode == "1d_row_imf0_raw":
        component = row.components[0]
        scale = 1.0
    elif mode == "1d_col_imf0_matched":
        col = decompose_1d_raster(
            orig, flatten_order="F", sift_thresh=sift_thresh, normalize_components=False
        )
        component, scale = scale_component_to_target_energy(
            col.components[0], target_energy=target_energy
        )
    elif mode == "2d_bimf0_matched":
        twod = decompose_emd2d_pyemd(orig, max_imf=1, normalize_components=False)
        component, scale = scale_component_to_target_energy(
            twod.components[0], target_energy=target_energy
        )
    elif mode == "2d_subtract_bimf0":
        twod = decompose_emd2d_pyemd(orig, max_imf=-1, normalize_components=False)
        if not twod.components:
            raise RuntimeError("EMD2D returned no oscillatory BIMF")
        component = twod.components[0]
        scale = 1.0
        meta["n_bimf"] = twod.n_oscillatory
        meta["natural_energy_fraction"] = component_energy_stats(orig, component)[
            "energy_fraction_of_original"
        ]
    elif mode == "2d_subtract_residual":
        twod = decompose_emd2d_pyemd(orig, max_imf=-1, normalize_components=False)
        component = twod.residual
        scale = 1.0
        meta["n_bimf"] = twod.n_oscillatory
        meta["natural_energy_fraction"] = component_energy_stats(orig, component)[
            "energy_fraction_of_original"
        ]
    elif mode == "2d_residual_matched":
        twod = decompose_emd2d_pyemd(orig, max_imf=-1, normalize_components=False)
        component, scale = scale_component_to_target_energy(
            twod.residual, target_energy=target_energy
        )
        meta["n_bimf"] = twod.n_oscillatory
        meta["natural_energy_fraction"] = component_energy_stats(orig, twod.residual)[
            "energy_fraction_of_original"
        ]
    elif mode == "fft_hp_matched":
        fft_best = match_fourier_cutoff_to_energy(orig, target_energy=target_energy)
        component, scale = scale_component_to_target_energy(
            fft_best["component"], target_energy=target_energy
        )
        meta["fft_cutoff_cycles_per_pixel"] = fft_best["cutoff_cycles_per_pixel"]
    else:
        raise ValueError(f"Unknown geometry enhance mode: {mode}")

    processed = (orig.astype(np.float64) - component.astype(np.float64)).astype(np.float32)
    out = safe_minmax_normalize(processed, clip=clip_output)
    meta["scale_factor"] = float(scale)
    meta["removed_energy_fraction"] = component_energy_stats(orig, component)[
        "energy_fraction_of_original"
    ]
    return out, meta

