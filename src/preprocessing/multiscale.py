"""
Fine-grained multiscale decompositions for the scale-removal ablation.

Three decompositions, all exact (``sum(components) + residual == image``) and
cheap enough to compute on the fly from the raw MRI (milliseconds per frame;
about 0.2 s for ``raster_emd``), so no decomposition cache is needed:

``gaussian_bands_octave``
    Difference-of-Gaussians bands at fixed octave scales. Band ``k`` holds
    structure between Gaussian sigma ``SIGMAS[k-1]`` and ``SIGMAS[k]``
    (band 0: finer than sigma 1 px). Scales are chosen, not data-driven.

``fabemd_v1``
    Fast and Adaptive BEMD (Bhuiyan, Adhami & Khan, 2008). Bidimensional EMD
    whose envelopes come from order-statistic (max/min) filters smoothed by a
    mean filter, instead of PyEMD's global cubic-RBF surfaces. The filters
    cannot overshoot, so components stay at image amplitude (PyEMD BEMD on this
    data produced cancelling BIMFs with 10x-10^6x the image energy). Window size
    is data-driven from the spacing of local extrema and forced to grow, so
    BIMF 0 is finest and each later BIMF is coarser. One sift per BIMF, as in
    the original FABEMD.

``raster_emd_v1``
    The Gastro decomposition (``external/Gastro/utils/dataloader.py``, ``emd2d``):
    the frame is flattened row by row into one long 1D signal, sifted with
    ``emd.sift.sift(sift_thresh=1e-8)``, and each IMF is reshaped back to the
    frame. IMF 0 is finest along the raster; the last column is the trend, so
    Gastro's ``[-1, -2, -3]`` removes the three slowest components. The IMF count
    varies per frame (9-10 on this data). Here the unit-normalized frame is
    sifted, so the IMFs are in image units and are removed at their real size.

Removal is amplitude-faithful: components are expressed in the same units as
the unit-normalized image and subtracted as-is, then the result is min-max
finalized. This differs from the legacy Gastro convention (min-max each
component to [0, 1], subtract from the *raw* image), which on this data changes
the model input by <0.001 and therefore removes effectively nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from src.preprocessing.emd_enhancement import as_grayscale_slice, safe_minmax_normalize

GAUSSIAN_BANDS_ID = "gaussian_bands_octave"
FABEMD_ID = "fabemd_v1"
RASTER_EMD_ID = "raster_emd_v1"

RASTER_EMD_SIFT_THRESH = 1e-8

# Band k spans sigma SIGMAS[k-1]..SIGMAS[k] (sigma 0 = the image itself).
GAUSSIAN_SIGMAS: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)

FABEMD_MAX_IMF = 5
FABEMD_MIN_EXTREMA = 4  # stop when either extrema map has fewer points


@dataclass
class MultiscaleDecomposition:
    original: np.ndarray  # unit-normalized native-FOV image, float64
    components: List[np.ndarray]  # finest first
    residual: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_components(self) -> int:
        return len(self.components)

    def reconstruction_rmse(self) -> float:
        recon = np.sum(np.stack(list(self.components) + [self.residual]), axis=0)
        return float(np.sqrt(np.mean((self.original - recon) ** 2)))


def _unit_image(image_2d: np.ndarray) -> np.ndarray:
    return safe_minmax_normalize(as_grayscale_slice(image_2d), clip=True).astype(np.float64)


def gaussian_bands(
    image_2d: np.ndarray,
    sigmas: Sequence[float] = GAUSSIAN_SIGMAS,
) -> MultiscaleDecomposition:
    """Difference-of-Gaussians band stack, finest band first."""
    unit = _unit_image(image_2d)
    levels = [unit] + [ndimage.gaussian_filter(unit, s, mode="mirror") for s in sigmas]
    bands = [levels[k] - levels[k + 1] for k in range(len(sigmas))]
    return MultiscaleDecomposition(
        original=unit,
        components=bands,
        residual=levels[-1],
        meta={"method_id": GAUSSIAN_BANDS_ID, "sigmas": list(sigmas)},
    )


_NEIGHBOURS = np.ones((3, 3), dtype=bool)
_NEIGHBOURS[1, 1] = False


def _strict_extrema(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Coordinates of strict 8-neighbour maxima and minima (plateaus excluded)."""
    maxima = s > ndimage.maximum_filter(s, footprint=_NEIGHBOURS, mode="mirror")
    minima = s < ndimage.minimum_filter(s, footprint=_NEIGHBOURS, mode="mirror")
    return np.argwhere(maxima), np.argwhere(minima)


def _median_nn_distance(points: np.ndarray) -> float:
    dist, _ = cKDTree(points).query(points, k=2)
    return float(np.median(dist[:, 1]))


def fabemd(
    image_2d: np.ndarray,
    max_imf: int = FABEMD_MAX_IMF,
    min_extrema: int = FABEMD_MIN_EXTREMA,
) -> MultiscaleDecomposition:
    """
    FABEMD with order-statistic envelopes.

    Window per BIMF = odd ceil of min(median nearest-neighbour spacing of
    maxima, of minima), at least 3 and at least the previous window + 2.
    When the residue has fewer than ``min_extrema`` maxima or minima, the
    spacing is undefined and the window doubles instead (``fallback_windows``
    in meta), so every frame yields the same BIMF count. Stops at ``max_imf``
    or when the window would exceed the image.
    """
    unit = _unit_image(image_2d)
    residue = unit.copy()
    bimfs: List[np.ndarray] = []
    windows: List[int] = []
    fallback_windows: List[int] = []
    stop_reason = "max_imf"
    prev_w = 1
    while len(bimfs) < max_imf:
        max_pts, min_pts = _strict_extrema(residue)
        if len(max_pts) < min_extrema or len(min_pts) < min_extrema:
            spacing = 2.0 * prev_w
            fallback_windows.append(len(bimfs))
        else:
            spacing = min(_median_nn_distance(max_pts), _median_nn_distance(min_pts))
        w = max(3, int(np.ceil(spacing)), prev_w + 2)
        w += 1 - w % 2  # odd
        if w > min(residue.shape):
            stop_reason = "window_exceeds_image"
            break
        upper = ndimage.uniform_filter(ndimage.maximum_filter(residue, size=w, mode="mirror"), size=w, mode="mirror")
        lower = ndimage.uniform_filter(ndimage.minimum_filter(residue, size=w, mode="mirror"), size=w, mode="mirror")
        mean_env = 0.5 * (upper + lower)
        bimfs.append(residue - mean_env)
        residue = mean_env
        windows.append(w)
        prev_w = w
    return MultiscaleDecomposition(
        original=unit,
        components=bimfs,
        residual=residue,
        meta={
            "method_id": FABEMD_ID,
            "max_imf": max_imf,
            "min_extrema": min_extrema,
            "windows": windows,
            "fallback_windows": fallback_windows,
            "stop_reason": stop_reason,
        },
    )


def raster_emd(
    image_2d: np.ndarray,
    sift_thresh: float = RASTER_EMD_SIFT_THRESH,
) -> MultiscaleDecomposition:
    """
    Gastro-style 1D EMD of the row-major raster, finest IMF first.

    ``emd.sift.sift`` returns the IMFs with the trend as its last column, and
    the columns sum exactly to the signal. All columns become components (so
    negative indices match Gastro's ``imf_index``); ``residual`` is the
    float round-off, effectively zero.
    """
    import warnings

    import emd

    unit = _unit_image(image_2d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # emd's np.log10(where=...) UserWarning, once per sift
        imfs = emd.sift.sift(unit.reshape(-1), sift_thresh=sift_thresh)
    components = [imfs[:, i].reshape(unit.shape) for i in range(imfs.shape[1])]
    return MultiscaleDecomposition(
        original=unit,
        components=components,
        residual=unit - np.sum(np.stack(components), axis=0),
        meta={
            "method_id": RASTER_EMD_ID,
            "flatten_order": "C",
            "sift_thresh": sift_thresh,
            "n_imfs": len(components),
        },
    )


DECOMPOSERS = {
    GAUSSIAN_BANDS_ID: gaussian_bands,
    FABEMD_ID: fabemd,
    RASTER_EMD_ID: raster_emd,
}


def decompose(method_id: str, image_2d: np.ndarray) -> MultiscaleDecomposition:
    try:
        fn = DECOMPOSERS[method_id]
    except KeyError as exc:
        raise ValueError(f"Unknown multiscale method {method_id!r}; known: {sorted(DECOMPOSERS)}") from exc
    return fn(image_2d)


def subtract_components(
    original: np.ndarray,
    components: Sequence[np.ndarray],
    indices: Sequence[int],
    *,
    clip_output: bool = True,
) -> np.ndarray:
    """Amplitude-faithful removal: ``original - sum(components[indices])``, then min-max."""
    if not indices:
        raise ValueError("subtract requires at least one component index")
    removed = np.zeros_like(original, dtype=np.float64)
    for idx in indices:
        if idx >= len(components) or idx < -len(components):
            raise IndexError(f"Component index {idx} out of range for {len(components)} components")
        removed += components[idx]
    return safe_minmax_normalize(original - removed, clip=clip_output)
