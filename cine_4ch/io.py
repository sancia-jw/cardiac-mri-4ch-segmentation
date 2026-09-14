"""NIfTI discovery, pairing, and loading for 4CH_TR."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from cine_4ch.config import DATA_ROOT, LFS_HINT, NII_EXTENSIONS, VIEW


@dataclass(frozen=True)
class CasePair:
    """One paired image / annotation case."""

    stem: str
    case_id: str
    image_path: Path
    anno_path: Path


def strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def extract_case_id(stem: str) -> str:
    """Extract numeric case ID from filenames like CINE_4CH_001."""
    match = re.search(r"(\d{3,})$", stem)
    if match:
        return match.group(1)
    return stem


def collect_nii_paths(directory: Path) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    if not directory.is_dir():
        return paths
    for extension in NII_EXTENSIONS:
        for path in sorted(directory.glob(f"*{extension}")):
            paths[strip_nii_suffix(path.name)] = path
    return paths


def get_view_dirs(data_root: Path = DATA_ROOT) -> Tuple[Path, Path]:
    view_root = data_root / VIEW
    return view_root / "image", view_root / "anno"


def discover_cases(data_root: Path = DATA_ROOT) -> Tuple[List[CasePair], List[str], List[str]]:
    """
    Return paired cases plus stems missing image or annotation.

    Raises FileNotFoundError when image/anno directories are absent.
    """
    image_dir, anno_dir = get_view_dirs(data_root)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}\n{LFS_HINT}")
    if not anno_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {anno_dir}\n{LFS_HINT}")

    image_map = collect_nii_paths(image_dir)
    anno_map = collect_nii_paths(anno_dir)

    shared = sorted(set(image_map) & set(anno_map))
    missing_image = sorted(set(anno_map) - set(image_map))
    missing_anno = sorted(set(image_map) - set(anno_map))

    pairs = [
        CasePair(
            stem=stem,
            case_id=extract_case_id(stem),
            image_path=image_map[stem],
            anno_path=anno_map[stem],
        )
        for stem in shared
    ]
    return pairs, missing_image, missing_anno


def load_volume(path: Path) -> Tuple[np.ndarray, Tuple[float, ...]]:
    """Load a NIfTI volume and squeeze singleton dimensions."""
    try:
        nii = nib.load(str(path))
        data = np.asarray(nii.get_fdata(dtype=np.float32))
        data = np.squeeze(data)
        if data.size == 0:
            raise ValueError("empty volume")
        spacing = tuple(float(v) for v in nii.header.get_zooms()[: data.ndim])
        return data, spacing
    except Exception as exc:
        raise RuntimeError(f"Failed to load {path}: {exc}") from exc


def load_pair(case: CasePair) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]:
    image, spacing = load_volume(case.image_path)
    label, _ = load_volume(case.anno_path)
    label = np.rint(label).astype(np.int64)
    return image, label, spacing


def choose_representative_frame(label: np.ndarray) -> int:
    """Pick the temporal frame with the most foreground voxels."""
    if label.ndim < 3:
        return 0
    foreground_per_frame = (label > 0).reshape(-1, label.shape[-1]).sum(axis=0)
    if foreground_per_frame.max() > 0:
        return int(np.argmax(foreground_per_frame))
    return label.shape[-1] // 2


def extract_frame(volume: np.ndarray, frame_idx: int) -> np.ndarray:
    if volume.ndim == 2:
        return volume
    if volume.ndim == 3:
        return volume[..., frame_idx]
    raise ValueError(f"Expected 2D or 3D volume, got shape {volume.shape}")


def normalize_image_slice(image_2d: np.ndarray) -> np.ndarray:
    """Per-slice min-max normalization to [0, 1]."""
    vmin = float(image_2d.min())
    vmax = float(image_2d.max())
    if vmax <= vmin:
        return np.zeros_like(image_2d, dtype=np.float32)
    return ((image_2d - vmin) / (vmax - vmin)).astype(np.float32)
