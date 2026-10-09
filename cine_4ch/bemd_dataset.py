"""Dataset that applies component-removal conditions to 4CH frames.

``source="bemd_cache"`` (legacy): training never runs BEMD; it only loads cached
BIMFs / residual and applies the Gastro-style subtract + finalize used by the
historical 1D EMD ablation.

Other sources (``src.preprocessing.multiscale``: Gaussian bands, FABEMD, raster
EMD) are decomposed on the fly from the raw frame and removed
amplitude-faithfully; the derived inputs are still written to the enhanced
cache, in parallel across cases when many frames are missing.
"""

from __future__ import annotations

import hashlib
import json
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from cine_4ch.bemd_cache import BEMD_CACHE_ROOT, is_valid_cache_entry, load_decomposition
from cine_4ch.config import DEFAULT_IMAGE_SIZE, OUTPUTS_DIR
from cine_4ch.dataset import _resize_slice, _to_image_tensor
from cine_4ch.io import CasePair, extract_frame, load_pair, load_volume
from src.preprocessing import multiscale
from src.preprocessing.bemd_square_pad import METHOD_ID as BEMD_METHOD_ID
from src.preprocessing.bemd_square_pad import BEMDConfig, enhance_from_bemd_decomp
from src.preprocessing.emd_enhancement import as_grayscale_slice, safe_minmax_normalize

ENHANCED_CACHE_ROOT = OUTPUTS_DIR / "bemd_enhanced_cache"

BEMD_CACHE_SOURCE = "bemd_cache"
LEGACY_RECONSTRUCTION = "gastro_style_minmax_bimf_then_finalize"
AMPLITUDE_RECONSTRUCTION = "amplitude_subtract_then_finalize"


@dataclass(frozen=True)
class BEMDEnhanceSpec:
    """One ablation condition: which components to remove, and from which decomposition."""

    run_id: str
    mode: str  # original | subtract
    bimf_indices: Tuple[int, ...]
    description: str = ""
    normalize_bimfs: bool = True
    clip_output: bool = True
    # bemd_cache | a src.preprocessing.multiscale method id
    source: str = BEMD_CACHE_SOURCE
    # Also remove every component from this index to the last (on-the-fly sources
    # only); for decompositions whose component count varies per frame.
    tail_from: Optional[int] = None

    @property
    def uses_bemd_cache(self) -> bool:
        return self.source == BEMD_CACHE_SOURCE

    @property
    def decomposition_method(self) -> str:
        return BEMD_METHOD_ID if self.uses_bemd_cache else self.source

    @property
    def reconstruction(self) -> str:
        return LEGACY_RECONSTRUCTION if self.uses_bemd_cache else AMPLITUDE_RECONSTRUCTION

    def cache_key(self) -> str:
        payload = {
            "run_id": self.run_id,
            "mode": self.mode,
            "bimf_indices": list(self.bimf_indices),
            "normalize_bimfs": self.normalize_bimfs,
            "clip_output": self.clip_output,
            "reconstruction": self.reconstruction,
        }
        # Only non-legacy sources add the key, so existing enhanced caches stay valid.
        if not self.uses_bemd_cache:
            payload["source"] = self.source
        if self.tail_from is not None:
            payload["tail_from"] = self.tail_from
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


RAW_SOURCE = "raw"  # original-mode baseline read straight from the MRI, no cache


def multiscale_ablation_specs() -> List[BEMDEnhanceSpec]:
    """Fine-scale removal: Gaussian octave bands and FABEMD BIMFs (amplitude-faithful)."""
    sigmas = multiscale.GAUSSIAN_SIGMAS
    edges = (0.0,) + tuple(sigmas)
    specs = [
        BEMDEnhanceSpec(
            run_id="original",
            mode="original",
            bimf_indices=(),
            description="Normalized original MRI only (no component removal).",
            source=RAW_SOURCE,
        )
    ]
    for k in range(len(sigmas)):
        specs.append(
            BEMDEnhanceSpec(
                run_id=f"subtract_gband_{k}",
                mode="subtract",
                bimf_indices=(k,),
                description=f"Original minus Gaussian band {k} (sigma {edges[k]:g}-{edges[k + 1]:g} px).",
                source=multiscale.GAUSSIAN_BANDS_ID,
            )
        )
    # BIMF 4 is excluded: ~36% of frames reach it only via the window-doubling
    # fallback and 26 frames are too small for it.
    typical_windows = (3, 7, 17, 33)
    for k, w in enumerate(typical_windows):
        specs.append(
            BEMDEnhanceSpec(
                run_id=f"subtract_fabemd_{k}",
                mode="subtract",
                bimf_indices=(k,),
                description=f"Original minus FABEMD BIMF {k} (median envelope window {w} px).",
                source=multiscale.FABEMD_ID,
            )
        )
    return specs


def raster_emd_ablation_specs() -> List[BEMDEnhanceSpec]:
    """Gastro 1D raster EMD (``external/Gastro/utils``), IMFs removed amplitude-faithfully."""
    raw_original = multiscale_ablation_specs()[0]
    conditions = [
        ("subtract_remd_0", (0,), "IMF 0 (finest along the raster)"),
        ("subtract_remd_1", (1,), "IMF 1"),
        ("subtract_remd_0_1", (0, 1), "IMFs 0 and 1"),
        ("subtract_remd_trend", (-1, -2, -3), "the 3 slowest IMFs, trend included (Gastro's setting)"),
        # Lower-frequency follow-up. Raster period on real frames (rows are ~155-169 px):
        # IMF 2 ~43 px, IMF 3 ~100 px, IMF 4 ~220 px (1-2 rows), IMF 5+ several rows.
        ("subtract_remd_2", (2,), "IMF 2 (~40 px along the raster)"),
        ("subtract_remd_3", (3,), "IMF 3 (~100 px, about one image row)"),
        ("subtract_remd_4", (4,), "IMF 4 (~200 px, 1-2 image rows)"),
    ]
    specs = [raw_original] + [
        BEMDEnhanceSpec(
            run_id=run_id,
            mode="subtract",
            bimf_indices=indices,
            description=f"Original minus raster-EMD {what}.",
            source=multiscale.RASTER_EMD_ID,
        )
        for run_id, indices, what in conditions
    ]
    # 9-10 IMFs per frame, so "IMF 5 to the last" can't be a fixed index list.
    specs.append(
        BEMDEnhanceSpec(
            run_id="subtract_remd_5plus",
            mode="subtract",
            bimf_indices=(),
            description="Original minus raster-EMD IMFs 5 to last (structure spanning several rows, trend included).",
            source=multiscale.RASTER_EMD_ID,
            tail_from=5,
        )
    )
    return specs


CATALOGS = {
    "bemd": default_bemd_ablation_specs,
    "multiscale": multiscale_ablation_specs,
    "raster_emd": raster_emd_ablation_specs,
}


def required_bimf_count(specs: Sequence[BEMDEnhanceSpec] | None = None) -> int:
    """BIMFs the BEMD cache must hold; on-the-fly sources need none."""
    specs = list(specs) if specs is not None else default_bemd_ablation_specs()
    needed = 0
    for s in specs:
        if s.bimf_indices and s.uses_bemd_cache:
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


def enhance_frame_from_raw(raw_2d: np.ndarray, spec: BEMDEnhanceSpec) -> np.ndarray:
    """On-the-fly decomposition + amplitude-faithful removal for non-cache sources."""
    if spec.mode == "original":
        return safe_minmax_normalize(as_grayscale_slice(raw_2d), clip=spec.clip_output)
    if spec.mode != "subtract":
        raise ValueError(f"Unsupported enhance mode: {spec.mode}")
    decomp = multiscale.decompose(spec.source, raw_2d)
    indices = list(spec.bimf_indices)
    if spec.tail_from is not None:
        indices += [k for k in range(spec.tail_from, decomp.n_components) if k not in indices]
    return multiscale.subtract_components(
        decomp.original, decomp.components, indices, clip_output=spec.clip_output
    )


PARALLEL_MIN_FRAMES = 64


def _enhance_case_to_cache(case: CasePair, frame_idxs: Sequence[int], spec: BEMDEnhanceSpec, key: str, root: Path) -> None:
    raw_volume, _ = load_volume(case.image_path)
    for frame_idx in frame_idxs:
        arr = enhance_frame_from_raw(extract_frame(raw_volume, frame_idx), spec)
        path = _enhanced_path(case.stem, frame_idx, key, root)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, arr.astype(np.float32, copy=False))
        os.replace(tmp, path)


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
                if spec.uses_bemd_cache and not is_valid_cache_entry(
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
        if not self.spec.uses_bemd_cache and self.spec.mode != "original":
            self._precompute_parallel(key)
        out: List[np.ndarray] = []
        raw_case_idx, raw_volume = None, None  # index is case-ordered: load each volume once
        for case_idx, frame_idx in tqdm(self.index, desc=desc, leave=False):
            case = self.cases[case_idx]
            path = _enhanced_path(case.stem, frame_idx, key, self.enhanced_cache_root)
            if self.use_enhanced_disk_cache and path.is_file():
                out.append(np.load(path))
                continue
            if self.spec.uses_bemd_cache:
                arr = enhance_frame_from_cache(
                    case.stem,
                    frame_idx,
                    self.spec,
                    bemd_cache_root=self.bemd_cache_root,
                )
            else:
                if raw_case_idx != case_idx:
                    raw_volume, _ = load_volume(case.image_path)
                    raw_case_idx = case_idx
                arr = enhance_frame_from_raw(extract_frame(raw_volume, frame_idx), self.spec)
            if self.use_enhanced_disk_cache:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, arr.astype(np.float32, copy=False))
            out.append(arr)
        self._preprocessed = out

    def _precompute_parallel(self, key: str) -> None:
        """Fill the enhanced cache for uncached frames across processes, one case per task.

        Only pays off for slow on-the-fly sources (raster EMD: ~0.2 s/frame), so it
        runs only when the disk cache is on and enough frames are missing; outputs
        are identical to the serial path, which then just loads them.
        """
        if not self.use_enhanced_disk_cache:
            return
        workers = int(os.environ.get("ONTHEFLY_WORKERS", min(4, os.cpu_count() or 1)))
        todo: Dict[int, List[int]] = {}
        for case_idx, frame_idx in self.index:
            path = _enhanced_path(self.cases[case_idx].stem, frame_idx, key, self.enhanced_cache_root)
            if not path.is_file():
                todo.setdefault(case_idx, []).append(frame_idx)
        if workers < 2 or sum(len(v) for v in todo.values()) < PARALLEL_MIN_FRAMES:
            return
        # spawn, not fork: the parent already runs torch threads (and Windows has no fork)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            jobs = [
                pool.submit(_enhance_case_to_cache, self.cases[c], frames, self.spec, key, self.enhanced_cache_root)
                for c, frames in todo.items()
            ]
            for job in tqdm(as_completed(jobs), total=len(jobs), desc=f"{self.spec.run_id} decompose x{workers}", leave=False):
                job.result()

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
