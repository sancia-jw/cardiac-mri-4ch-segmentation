"""
2D Empirical Mode Decomposition (EMD) enhancement for cardiac MRI slices.

Adapted from ``external/Gastro/utils/dataloader.py`` for grayscale CINE/LGE
frames used in **segmentation**. This module only transforms image intensities;
segmentation masks must be kept as integer class labels and must not be passed
through these functions.

Supported image shapes
----------------------
- ``(H, W)``       grayscale MRI slice
- ``(H, W, 1)``    grayscale slice with explicit channel dimension

Typical outputs
---------------
- ``original``, ``imf_only``, ``subtract`` → ``(H, W)`` float32
- ``concat`` → ``(H, W, C)`` float32 with ``C = 1 + len(imf_indices)`` when
  each selected IMF is emitted as its own channel, or ``C = 2`` when the
  selected IMFs are summed into one reconstruction channel (default).

IMF index guide (``emd`` SIFT ordering)
---------------------------------------
- ``[0]``        : highest-frequency detail / fine texture
- ``[1]``, ``[2]``: mid-scale structures, often useful edges
- ``[-1]``, ``[-2]``, ``[-3]`` : lowest-frequency trend / bias components

Examples for cardiac MRI
------------------------
- ``mode="original"`` : baseline, no EMD
- ``mode="imf_only", imf_indices=[1]`` : emphasize mid-frequency structure
- ``mode="subtract", imf_indices=[-1, -2, -3]`` : Gastro-style trend removal
- ``mode="concat", imf_indices=[1]`` : 2-channel U-Net input (image + IMF)

Requires: ``pip install emd``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Literal, Sequence, Tuple, Union

import numpy as np

try:
    import emd
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'emd' package is required for EMD enhancement. Install with: pip install emd"
    ) from exc


EnhancementMode = Literal["original", "imf_only", "subtract", "concat"]
FlattenOrder = Literal["C", "F"]  # C = row-major (legacy), F = column-major control


@dataclass
class EMEnhancementConfig:
    """
    EMD settings for grayscale cardiac MRI slices.

    Attributes
    ----------
    mode:
        - ``original``  : normalized input only (no EMD)
        - ``imf_only``  : sum of selected IMFs, then normalized
        - ``subtract``  : original minus sum of selected IMFs, then normalized
        - ``concat``    : stack normalized original + IMF channel(s)
    imf_indices:
        IMF indices to use for non-``original`` modes, e.g. ``[0]``, ``[1]``,
        ``[0, 1]``, ``[1, 2]``, ``[-1, -2, -3]``.
    normalize_imfs:
        If True, min-max normalize each IMF after decomposition (Gastro default).
        Subtraction uses these normalized IMF values against the raw slice.
    sift_thresh:
        Stopping threshold for ``emd.sift.sift``.
    clip_output:
        If True, clip final image intensities to ``[0, 1]`` after normalization.
    concat_sum_imfs:
        If True (default), ``concat`` mode uses one summed IMF reconstruction
        channel. If False, each selected IMF is normalized and stacked separately.
    flatten_order:
        NumPy reshape order for 1D raster EMD. ``\"C\"`` (default) is row-major
        (legacy Gastro / completed ablations). ``\"F\"`` is column-major control.
        Ignored by true 2D backends (see ``src.preprocessing.emd_geometry``).
    """

    mode: EnhancementMode = "subtract"
    imf_indices: List[int] = field(default_factory=lambda: [-1, -2, -3])
    normalize_imfs: bool = True
    sift_thresh: float = 1e-8
    clip_output: bool = True
    concat_sum_imfs: bool = True
    flatten_order: FlattenOrder = "C"


def as_grayscale_slice(image: np.ndarray) -> np.ndarray:
    """
    Validate and return a 2D float32 grayscale MRI slice.

    Accepts ``(H, W)`` or ``(H, W, 1)`` only.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0].astype(np.float32, copy=False)
    raise ValueError(
        f"Expected grayscale MRI slice with shape (H, W) or (H, W, 1), got {arr.shape}"
    )


def safe_minmax_normalize(
    array: np.ndarray,
    *,
    clip: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Per-slice min-max normalization with division-by-zero protection.

    Flat or constant arrays map to zeros. Optionally clips to ``[0, 1]``.
    """
    arr = np.asarray(array, dtype=np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) <= eps:
        out = np.zeros_like(arr, dtype=np.float32)
    else:
        out = (arr - vmin) / (vmax - vmin + eps)
        out = out.astype(np.float32)
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out


def compute_imfs(
    image_2d: np.ndarray,
    *,
    normalize_imfs: bool = True,
    sift_thresh: float = 1e-8,
    clip_imfs: bool = False,
    flatten_order: FlattenOrder = "C",
) -> List[np.ndarray]:
    """
    Decompose a grayscale slice via **1D raster EMD** (not bidimensional EMD).

    The slice is flattened with ``flatten_order`` (``C`` = row-major legacy,
    ``F`` = column-major), passed to ``emd.sift.sift``, and each IMF is reshaped
    back to ``(H, W)`` with the same order.
    """
    slice_2d = as_grayscale_slice(image_2d)
    signal_1d = slice_2d.reshape(-1, order=flatten_order)

    all_imfs = emd.sift.sift(signal_1d, sift_thresh=sift_thresh)
    n_imfs = int(all_imfs.shape[1])

    imfs: List[np.ndarray] = []
    for i in range(n_imfs):
        imf_2d = all_imfs[:, i].reshape(slice_2d.shape, order=flatten_order).astype(np.float32)
        if normalize_imfs:
            imf_2d = safe_minmax_normalize(imf_2d, clip=clip_imfs)
        imfs.append(imf_2d)
    return imfs


def select_imfs(imfs: Sequence[np.ndarray], imf_indices: Iterable[int]) -> List[np.ndarray]:
    """Select IMF components by index (supports negative indices)."""
    if not imfs:
        raise ValueError("IMF list is empty.")
    if not imf_indices:
        raise ValueError("imf_indices must contain at least one index for EMD modes.")

    selected: List[np.ndarray] = []
    for idx in imf_indices:
        if idx >= len(imfs) or idx < -len(imfs):
            raise IndexError(
                f"IMF index {idx} out of range for {len(imfs)} available IMFs "
                f"(valid: 0..{len(imfs) - 1} or negative indices)."
            )
        selected.append(np.asarray(imfs[idx], dtype=np.float32))
    return selected


def sum_imfs(imfs: Sequence[np.ndarray]) -> np.ndarray:
    """Sum one or more 2D IMF arrays."""
    if not imfs:
        raise ValueError("Cannot sum an empty IMF list.")
    total = np.zeros_like(imfs[0], dtype=np.float32)
    for imf in imfs:
        total = total + imf
    return total


def reconstruct_from_imfs(
    image_2d: np.ndarray,
    imf_indices: Sequence[int],
    *,
    normalize_imfs: bool = True,
    sift_thresh: float = 1e-8,
    flatten_order: FlattenOrder = "C",
    imfs: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Return the sum of selected IMFs as a ``(H, W)`` float32 image."""
    slice_2d = as_grayscale_slice(image_2d)
    imf_list = list(imfs) if imfs is not None else compute_imfs(
        slice_2d,
        normalize_imfs=normalize_imfs,
        sift_thresh=sift_thresh,
        flatten_order=flatten_order,
    )
    return sum_imfs(select_imfs(imf_list, imf_indices))


def subtract_imfs_from_image(
    image_2d: np.ndarray,
    imf_indices: Sequence[int],
    *,
    normalize_imfs: bool = True,
    sift_thresh: float = 1e-8,
    flatten_order: FlattenOrder = "C",
    imfs: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """
    Return ``original - sum(selected IMFs)`` as a ``(H, W)`` float32 image.

    With ``imf_indices=[-1, -2, -3]``, this removes low-frequency trend
    components (Gastro-style enhancement).
    """
    slice_2d = as_grayscale_slice(image_2d)
    imf_list = list(imfs) if imfs is not None else compute_imfs(
        slice_2d,
        normalize_imfs=normalize_imfs,
        sift_thresh=sift_thresh,
        flatten_order=flatten_order,
    )
    removed = sum_imfs(select_imfs(imf_list, imf_indices))
    return (slice_2d - removed).astype(np.float32)


def _finalize_output(
    image_2d: np.ndarray,
    *,
    clip: bool,
) -> np.ndarray:
    """Apply safe normalization for model input."""
    return safe_minmax_normalize(image_2d, clip=clip)


def enhance_mri_slice(
    image: np.ndarray,
    config: EMEnhancementConfig | None = None,
) -> np.ndarray:
    """
    Enhance a grayscale cardiac MRI slice for segmentation model input.

    **Masks are not modified by this function.** Apply it only to image
    intensities; keep segmentation masks as integer class labels.

    Parameters
    ----------
    image:
        Grayscale MRI slice, shape ``(H, W)`` or ``(H, W, 1)``.
    config:
        :class:`EMEnhancementConfig`.

    Returns
    -------
    np.ndarray
        - ``original`` / ``imf_only`` / ``subtract`` → ``(H, W)`` float32
        - ``concat`` → ``(H, W, C)`` float32

    Examples
    --------
    ::

        cfg = EMEnhancementConfig(mode="imf_only", imf_indices=[1])
        x = enhance_mri_slice(slice_2d, cfg)          # (H, W)

        cfg = EMEnhancementConfig(mode="concat", imf_indices=[0, 1])
        x = enhance_mri_slice(slice_2d, cfg)          # (H, W, 2) if concat_sum_imfs
    """
    cfg = config or EMEnhancementConfig()
    slice_2d = as_grayscale_slice(image)

    if cfg.mode == "original":
        return _finalize_output(slice_2d, clip=cfg.clip_output)

    imf_list = compute_imfs(
        slice_2d,
        normalize_imfs=cfg.normalize_imfs,
        sift_thresh=cfg.sift_thresh,
        flatten_order=cfg.flatten_order,
    )

    if cfg.mode == "imf_only":
        enhanced = reconstruct_from_imfs(
            slice_2d,
            cfg.imf_indices,
            imfs=imf_list,
            normalize_imfs=cfg.normalize_imfs,
            sift_thresh=cfg.sift_thresh,
            flatten_order=cfg.flatten_order,
        )
        return _finalize_output(enhanced, clip=cfg.clip_output)

    if cfg.mode == "subtract":
        enhanced = subtract_imfs_from_image(
            slice_2d,
            cfg.imf_indices,
            imfs=imf_list,
            normalize_imfs=cfg.normalize_imfs,
            sift_thresh=cfg.sift_thresh,
            flatten_order=cfg.flatten_order,
        )
        return _finalize_output(enhanced, clip=cfg.clip_output)

    if cfg.mode == "concat":
        original_ch = _finalize_output(slice_2d, clip=cfg.clip_output)
        chosen = select_imfs(imf_list, cfg.imf_indices)

        if cfg.concat_sum_imfs:
            imf_ch = _finalize_output(sum_imfs(chosen), clip=cfg.clip_output)
            return np.stack([original_ch, imf_ch], axis=-1)

        imf_channels = [_finalize_output(imf, clip=cfg.clip_output) for imf in chosen]
        return np.stack([original_ch, *imf_channels], axis=-1)

    raise ValueError(f"Unknown enhancement mode: {cfg.mode}")


def normalize_slice_for_unet(image_2d: np.ndarray, *, clip: bool = True) -> np.ndarray:
    """Normalize a 2D slice to float32 ``(H, W)`` in ``[0, 1]``."""
    return safe_minmax_normalize(as_grayscale_slice(image_2d), clip=clip)


def enhance_slice_for_unet(
    image_2d: np.ndarray,
    config: EMEnhancementConfig | None = None,
) -> np.ndarray:
    """
    Backward-compatible helper returning a single-channel ``(H, W)`` U-Net input.

    For ``concat`` mode, use :func:`enhance_mri_slice` instead to obtain a
    multi-channel tensor.
    """
    cfg = config or EMEnhancementConfig()
    if cfg.mode == "concat":
        raise ValueError(
            "concat mode produces multi-channel output; use enhance_mri_slice() instead."
        )
    return enhance_mri_slice(image_2d, cfg)


def output_channels(config: EMEnhancementConfig) -> int:
    """Return the number of image channels produced by ``enhance_mri_slice``."""
    if config.mode in ("original", "imf_only", "subtract"):
        return 1
    if config.mode == "concat":
        if config.concat_sum_imfs:
            return 2
        return 1 + len(config.imf_indices)
    raise ValueError(f"Unknown mode: {config.mode}")


def output_shape(
    height: int,
    width: int,
    config: EMEnhancementConfig,
) -> Tuple[int, ...]:
    """Return the spatial shape produced by :func:`enhance_mri_slice`."""
    channels = output_channels(config)
    if channels == 1:
        return (height, width)
    return (height, width, channels)
