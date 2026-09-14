"""PyTorch Dataset for 2D slices from CINE 4CH volumes."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from cine_4ch.config import DEFAULT_IMAGE_SIZE, OUTPUTS_DIR
from cine_4ch.io import CasePair, discover_cases, extract_frame, load_pair, normalize_image_slice
from src.preprocessing.emd_enhancement import EMEnhancementConfig, enhance_mri_slice

EMD_CACHE_ROOT = OUTPUTS_DIR / "emd_cache"


def load_split_cases(splits_csv: Path, split_name: str, data_root: Optional[Path] = None) -> List[CasePair]:
    """Load CasePair objects for one split from splits_4ch.csv."""
    df = pd.read_csv(splits_csv)
    split_df = df[df["split"] == split_name]
    all_cases, _, _ = discover_cases(data_root) if data_root else discover_cases()
    case_map = {case.case_id: case for case in all_cases}
    cases: List[CasePair] = []
    for raw_id in split_df["case_id"]:
        case_id = str(raw_id).zfill(3)
        if case_id in case_map:
            cases.append(case_map[case_id])
    return cases


def _resize_slice(image_2d: np.ndarray, label_2d: np.ndarray, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Resize image/label. Images stay HWC (or HW); labels stay HW integer maps."""
    label_t = torch.from_numpy(label_2d).float().unsqueeze(0).unsqueeze(0)
    label_r = F.interpolate(label_t, size=size, mode="nearest")

    if image_2d.ndim == 2:
        image_t = torch.from_numpy(image_2d).float().unsqueeze(0).unsqueeze(0)
        image_r = F.interpolate(image_t, size=size, mode="bilinear", align_corners=False)
        return image_r.squeeze().numpy(), label_r.squeeze().numpy().astype(np.int64)

    if image_2d.ndim != 3:
        raise ValueError(f"Expected image shape (H, W) or (H, W, C), got {image_2d.shape}")

    # Multi-channel enhanced input (H, W, C): interpolate in NCHW, return HWC.
    image_t = torch.from_numpy(image_2d).float().permute(2, 0, 1).unsqueeze(0)
    image_r = F.interpolate(image_t, size=size, mode="bilinear", align_corners=False)
    hwc = image_r.squeeze(0).permute(1, 2, 0).contiguous().numpy()
    return hwc, label_r.squeeze().numpy().astype(np.int64)


def _preprocess_image_slice(image_2d: np.ndarray, emd_config: EMEnhancementConfig | None) -> np.ndarray:
    if emd_config is None:
        return normalize_image_slice(image_2d)
    return enhance_mri_slice(image_2d, emd_config)


def _emd_config_cache_key(emd_config: EMEnhancementConfig | None) -> str:
    if emd_config is None:
        payload = {"mode": "normalize_only"}
    else:
        payload = {
            "mode": emd_config.mode,
            "imf_indices": list(emd_config.imf_indices),
            "normalize_imfs": emd_config.normalize_imfs,
            "sift_thresh": emd_config.sift_thresh,
            "clip_output": emd_config.clip_output,
            "concat_sum_imfs": emd_config.concat_sum_imfs,
        }
        # Keep legacy cache hashes stable for row-major (C); only key non-default order.
        flatten_order = getattr(emd_config, "flatten_order", "C")
        if flatten_order != "C":
            payload["flatten_order"] = flatten_order
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    mode = payload["mode"]
    return f"{mode}_{digest}"


def _cache_path(case_stem: str, frame_idx: int, config_key: str) -> Path:
    return EMD_CACHE_ROOT / config_key / case_stem / f"frame_{frame_idx:03d}.npy"


def _to_image_tensor(image_2d: np.ndarray) -> torch.Tensor:
    """Convert HW or HWC numpy image to CHW float tensor."""
    if image_2d.ndim == 2:
        return torch.from_numpy(image_2d).float().unsqueeze(0)
    if image_2d.ndim == 3:
        # Expect channel-last (H, W, C) from enhance_mri_slice / _resize_slice.
        return torch.from_numpy(np.ascontiguousarray(image_2d)).float().permute(2, 0, 1).contiguous()
    raise ValueError(f"Expected 2D or 3D image array, got shape {image_2d.shape}")


def _worker_preprocess_frame(args: tuple) -> tuple:
    """Process-pool worker: preprocess one frame and write cache. Returns (idx, path)."""
    idx, image_path, frame_idx, emd_dict, cache_file = args
    from cine_4ch.io import extract_frame, load_volume, normalize_image_slice
    from src.preprocessing.emd_enhancement import EMEnhancementConfig, enhance_mri_slice

    cache_path = Path(cache_file)
    if cache_path.exists():
        return idx, str(cache_path)

    image, _ = load_volume(Path(image_path))
    image_2d = extract_frame(image, frame_idx)
    if emd_dict is None:
        out = normalize_image_slice(image_2d)
    else:
        out = enhance_mri_slice(image_2d, EMEnhancementConfig(**emd_dict))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, out.astype(np.float32, copy=False))
    return idx, str(cache_path)


class Cine4CHSliceDataset(Dataset):
    """
    Each sample is one 2D frame from a 3D CINE volume.

    All temporal frames are included; frames with no foreground labels are kept
    so the model also learns background-only phases. Volumes are preloaded into
    memory once to avoid repeated NIfTI reads during training.
    """

    def __init__(
        self,
        cases: Sequence[CasePair],
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        augment: bool = False,
        emd_config: EMEnhancementConfig | None = None,
        precompute_preprocessing: bool = True,
        precompute_desc: str | None = None,
        use_disk_cache: bool = True,
        num_preprocess_workers: int = 0,
    ) -> None:
        self.cases = list(cases)
        self.image_size = image_size
        self.augment = augment
        self.emd_config = emd_config
        self.index: List[Tuple[int, int]] = []
        self._images: List[np.ndarray] = []
        self._labels: List[np.ndarray] = []
        self._preprocessed: List[np.ndarray] = []

        for case_idx, case in enumerate(self.cases):
            image, label, _ = load_pair(case)
            self._images.append(image)
            self._labels.append(label)
            num_frames = 1 if label.ndim < 3 else label.shape[-1]
            for frame_idx in range(num_frames):
                self.index.append((case_idx, frame_idx))

        if precompute_preprocessing:
            self._precompute(
                desc=precompute_desc or "preprocess",
                use_disk_cache=use_disk_cache,
                num_workers=num_preprocess_workers,
            )
        else:
            self._preprocessed = []

    def _precompute(self, desc: str, use_disk_cache: bool, num_workers: int) -> None:
        config_key = _emd_config_cache_key(self.emd_config)
        emd_dict = None if self.emd_config is None else {
            "mode": self.emd_config.mode,
            "imf_indices": list(self.emd_config.imf_indices),
            "normalize_imfs": self.emd_config.normalize_imfs,
            "sift_thresh": self.emd_config.sift_thresh,
            "clip_output": self.emd_config.clip_output,
            "concat_sum_imfs": self.emd_config.concat_sum_imfs,
            "flatten_order": getattr(self.emd_config, "flatten_order", "C"),
        }

        # Prefer disk cache hits (fast path on resume / re-eval).
        cached: List[Optional[np.ndarray]] = [None] * len(self.index)
        missing: List[tuple] = []
        for idx, (case_idx, frame_idx) in enumerate(self.index):
            case = self.cases[case_idx]
            if use_disk_cache:
                path = _cache_path(case.stem, frame_idx, config_key)
                if path.exists():
                    cached[idx] = np.load(path)
                    continue
            missing.append((idx, case_idx, frame_idx))

        if missing and num_workers > 0 and self.emd_config is not None and self.emd_config.mode != "original":
            jobs = []
            for idx, case_idx, frame_idx in missing:
                case = self.cases[case_idx]
                cache_file = str(_cache_path(case.stem, frame_idx, config_key))
                jobs.append((idx, str(case.image_path), frame_idx, emd_dict, cache_file))

            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                futures = [pool.submit(_worker_preprocess_frame, job) for job in jobs]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, leave=False):
                    idx, path = fut.result()
                    cached[idx] = np.load(path)
        else:
            for idx, case_idx, frame_idx in tqdm(missing, desc=desc, leave=False):
                image_2d = extract_frame(self._images[case_idx], frame_idx)
                out = _preprocess_image_slice(image_2d, self.emd_config)
                cached[idx] = out
                if use_disk_cache:
                    case = self.cases[case_idx]
                    path = _cache_path(case.stem, frame_idx, config_key)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, out.astype(np.float32, copy=False))

        self._preprocessed = [arr for arr in cached]  # type: ignore[misc]
        if any(x is None for x in self._preprocessed):
            raise RuntimeError("Preprocessing cache incomplete.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        case_idx, frame_idx = self.index[idx]
        case = self.cases[case_idx]
        label = self._labels[case_idx]

        if self._preprocessed:
            image_2d = self._preprocessed[idx]
        else:
            image_2d = _preprocess_image_slice(
                extract_frame(self._images[case_idx], frame_idx),
                self.emd_config,
            )

        label_2d = extract_frame(label, frame_idx).astype(np.int64)
        image_2d, label_2d = _resize_slice(image_2d, label_2d, self.image_size)

        if self.augment and np.random.rand() < 0.5:
            if image_2d.ndim == 2:
                image_2d = np.flip(image_2d, axis=1).copy()
            else:
                image_2d = np.flip(image_2d, axis=1).copy()
            label_2d = np.flip(label_2d, axis=1).copy()

        image_tensor = _to_image_tensor(image_2d)
        label_tensor = torch.from_numpy(label_2d).long()
        return image_tensor, label_tensor, case.stem, frame_idx
