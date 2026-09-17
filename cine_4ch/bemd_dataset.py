"""Dataset that applies BEMD subtract modes from a persistent decomposition cache.

Training never runs BEMD; it only loads cached BIMFs / residual and applies the
Gastro-style subtract + finalize used by the historical 1D EMD ablation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from cine_4ch.bemd_cache import BEMD_CACHE_ROOT, is_valid_cache_entry, load_decomposition
from cine_4ch.config import DEFAULT_IMAGE_SIZE, OUTPUTS_DIR
from cine_4ch.dataset import _resize_slice, _to_image_tensor
from cine_4ch.io import CasePair, extract_frame, load_pair
from src.preprocessing.bemd_square_pad import BEMDConfig, enhance_from_bemd_decomp

ENHANCED_CACHE_ROOT = OUTPUTS_DIR / "bemd_enhanced_cache"


@dataclass(frozen=True)
class BEMDEnhanceSpec:
    """One ablation condition applied on top of cached BEMD decompositions."""

    run_id: str
    mode: str  # original | subtract
    bimf_indices: Tuple[int, ...]
    description: str = ""
    normalize_bimfs: bool = True
    clip_output: bool = True

    def cache_key(self) -> str:
        payload = {
            "run_id": self.run_id,
            "mode": self.mode,
            "bimf_indices": list(self.bimf_indices),
            "normalize_bimfs": self.normalize_bimfs,
            "clip_output": self.clip_output,
            "reconstruction": "gastro_style_minmax_bimf_then_finalize",
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        return f"{self.run_id}_{digest}"


def default_bemd_ablation_specs() -> List[BEMDEnhanceSpec]:
    """Focused BEMD ablation (subtract-only; no IMF-only / concat)."""
    return [
        BEMDEnhanceSpec(
            run_id="original",
            mode="original",
            bimf_indices=(),
            description="Normalized original MRI only (no BEMD removal).",
        ),
        BEMDEnhanceSpec(
            run_id="subtract_bimf_0",
            mode="subtract",
            bimf_indices=(0,),
            description=(
                "Original minus BIMF 0 (typically finest / highest-spatial-frequency mode)."
            ),
        ),
        BEMDEnhanceSpec(
            run_id="subtract_bimf_1",
            mode="subtract",
            bimf_indices=(1,),
            description="Original minus BIMF 1 (generally coarser than BIMF 0).",
        ),
        BEMDEnhanceSpec(
            run_id="subtract_bimf_2",
            mode="subtract",
            bimf_indices=(2,),
            description="Original minus BIMF 2 (generally coarser mid-scale mode).",
        ),
        BEMDEnhanceSpec(
            run_id="subtract_bimf_3",
            mode="subtract",
            bimf_indices=(3,),
            description="Original minus BIMF 3 (generally coarsest oscillatory BIMF).",
        ),
        BEMDEnhanceSpec(
            run_id="subtract_bimf_0_1",
            mode="subtract",
            bimf_indices=(0, 1),
            description="Original minus BIMF 0 and BIMF 1.",
        ),
    ]


def required_bimf_count(specs: Sequence[BEMDEnhanceSpec] | None = None) -> int:
    specs = list(specs) if specs is not None else default_bemd_ablation_specs()
    needed = 0
    for s in specs:
        if s.bimf_indices:
            needed = max(needed, max(s.bimf_indices) + 1)
    return needed


def _enhanced_path(case_stem: str, frame_idx: int, key: str, root: Path = ENHANCED_CACHE_ROOT) -> Path:
    return root / key / case_stem / f"frame_{frame_idx:03d}.npy"


def enhance_frame_from_cache(
    case_stem: str,
    frame_idx: int,
    spec: BEMDEnhanceSpec,
    *,
    bemd_cache_root: Path = BEMD_CACHE_ROOT,
) -> np.ndarray:
    decomp = load_decomposition(case_stem, frame_idx, cache_root=bemd_cache_root)
    return enhance_from_bemd_decomp(
        decomp,
        mode=spec.mode,
        bimf_indices=spec.bimf_indices,
        normalize_bimfs=spec.normalize_bimfs,
        clip_output=spec.clip_output,
    )


class BEMDSliceDataset(Dataset):
    """
    Same slice indexing / resize / flip-aug as ``Cine4CHSliceDataset``, but image
    intensities come from cached ``bemd_default_square_pad`` + enhance spec.
    """

    def __init__(
        self,
        cases: Sequence[CasePair],
        spec: BEMDEnhanceSpec,
        *,
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        augment: bool = False,
        bemd_cache_root: Path = BEMD_CACHE_ROOT,
        use_enhanced_disk_cache: bool = True,
        require_n_bimf: Optional[int] = None,
        precompute_desc: str | None = None,
        enhanced_cache_root: Path = ENHANCED_CACHE_ROOT,
        excluded_frames: frozenset[tuple[str, int]] = frozenset(),
    ) -> None:
        self.cases = list(cases)
        self.spec = spec
        self.image_size = image_size
        self.augment = augment
        self.bemd_cache_root = Path(bemd_cache_root)
        self.enhanced_cache_root = Path(enhanced_cache_root)
        self.excluded_frames = frozenset(excluded_frames)
        self.use_enhanced_disk_cache = use_enhanced_disk_cache
        self.require_n_bimf = require_n_bimf
        if self.require_n_bimf is None and spec.bimf_indices:
            self.require_n_bimf = max(spec.bimf_indices) + 1

        self.index: List[Tuple[int, int]] = []
        self._labels: List[np.ndarray] = []
        self._preprocessed: List[np.ndarray] = []

        missing: List[Tuple[str, int]] = []
        for case_idx, case in enumerate(self.cases):
            _, label, _ = load_pair(case)
            self._labels.append(label)
            n_frames = 1 if label.ndim < 3 else int(label.shape[-1])
            for frame_idx in range(n_frames):
                if (case.stem, frame_idx) in self.excluded_frames:
                    continue
                self.index.append((case_idx, frame_idx))
                if not is_valid_cache_entry(
                    case.stem,
                    frame_idx,
                    cache_root=self.bemd_cache_root,
                    require_n_bimf=self.require_n_bimf,
                ):
                    missing.append((case.stem, frame_idx))

        if missing:
            preview = ", ".join(f"{s}/f{f}" for s, f in missing[:8])
            more = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
            raise FileNotFoundError(
                f"Missing/invalid BEMD cache for {len(missing)} frame(s) under "
                f"{self.bemd_cache_root} (need n_bimf>={self.require_n_bimf}). "
                f"Examples: {preview}{more}. Run scripts/preprocess_bemd.py first."
            )

        self._precompute(desc=precompute_desc or f"bemd {spec.run_id}")

    def _precompute(self, desc: str) -> None:
        key = self.spec.cache_key()
        out: List[np.ndarray] = []
        for case_idx, frame_idx in tqdm(self.index, desc=desc, leave=False):
            case = self.cases[case_idx]
            path = _enhanced_path(case.stem, frame_idx, key, self.enhanced_cache_root)
            if self.use_enhanced_disk_cache and path.is_file():
                out.append(np.load(path))
                continue
            arr = enhance_frame_from_cache(
                case.stem,
                frame_idx,
                self.spec,
                bemd_cache_root=self.bemd_cache_root,
            )
            if self.use_enhanced_disk_cache:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, arr.astype(np.float32, copy=False))
            out.append(arr)
        self._preprocessed = out

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        case_idx, frame_idx = self.index[idx]
        case = self.cases[case_idx]
        image_2d = self._preprocessed[idx]
        label_2d = extract_frame(self._labels[case_idx], frame_idx).astype(np.int64)
        image_2d, label_2d = _resize_slice(image_2d, label_2d, self.image_size)

        if self.augment and np.random.rand() < 0.5:
            image_2d = np.flip(image_2d, axis=1).copy()
            label_2d = np.flip(label_2d, axis=1).copy()

        return _to_image_tensor(image_2d), torch.from_numpy(label_2d).long(), case.stem, frame_idx


def spec_to_dict(spec: BEMDEnhanceSpec) -> dict:
    return asdict(spec)


def bemd_solver_config_dict(cfg: BEMDConfig | None = None) -> dict:
    return asdict(cfg or BEMDConfig())
