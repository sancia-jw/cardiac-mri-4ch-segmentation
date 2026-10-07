"""Read-only cache preflight; never runs decomposition or constructs a model."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

from cine_4ch.bemd_cache import METHOD_ID, is_valid_cache_entry


def load_exclusions(path: Path | None) -> frozenset[tuple[str, int]]:
    if path is None:
        return frozenset()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("method_id") != METHOD_ID:
        raise ValueError(f"Wrong decomposition method in {path}")
    result = set()
    for row in payload["excluded_frames"]:
        key = (row["case_stem"], row["frame_idx"])
        if not isinstance(key[0], str) or type(key[1]) is not int or key[1] < 0 or not row.get("reason"):
            raise ValueError(f"Invalid exclusion: {row}")
        if key in result:
            raise ValueError(f"Duplicate exclusion: {key}")
        result.add(key)
    return frozenset(result)


def audit_raw(cases, excluded_frames=frozenset(), *, progress=False) -> dict:
    """Header-only frame count / shape preflight for conditions that need no BEMD cache."""
    seen = set()
    for case in tqdm(cases, desc="raw MRI audit", disable=not progress):
        shape = tuple(n for n in nib.load(str(case.anno_path)).shape if n != 1)
        image_shape = tuple(n for n in nib.load(str(case.image_path)).shape if n != 1)
        if len(shape) not in (2, 3) or image_shape != shape:
            raise ValueError(f"Unexpected image/annotation shapes for {case.stem}: {image_shape}, {shape}")
        n_frames = 1 if len(shape) == 2 else shape[-1]
        seen.update((case.stem, frame) for frame in range(n_frames))
    unknown = excluded_frames - seen
    if unknown:
        raise ValueError(f"Exclusions refer to frames outside the supplied dataset: {sorted(unknown)}")
    return {"total_frames": len(seen), "excluded_frames": len(excluded_frames),
            "usable_frames": len(seen - excluded_frames), "issues": []}


def audit_cache(cases, cache_root: Path, excluded_frames=frozenset(), required_n_bimf=0, *, progress=False) -> dict:
    """Validate metadata, array headers/sizes and component coverage against raw headers.

    Reconstruction values are from the original per-frame validation. This is
    not a new full-array numerical reconstruction audit.
    """
    seen = set()
    available = Counter({str(i): 0 for i in range(4)})
    distribution = Counter()
    statuses = Counter()
    issues = []
    valid = 0
    rmse = []
    for case in tqdm(cases, desc="BEMD cache audit", disable=not progress):
        shape = tuple(n for n in nib.load(str(case.anno_path)).shape if n != 1)
        image_shape = tuple(n for n in nib.load(str(case.image_path)).shape if n != 1)
        if len(shape) not in (2, 3) or image_shape != shape:
            raise ValueError(f"Unexpected image/annotation shapes for {case.stem}: {image_shape}, {shape}")
        n_frames = 1 if len(shape) == 2 else shape[-1]
        for frame in range(n_frames):
            key = (case.stem, frame)
            seen.add(key)
            if key in excluded_frames:
                continue
            try:
                directory = cache_root / case.stem / f"frame_{frame:03d}"
                meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
                error = float(meta["reconstruction"]["rmse"])
                if not np.isfinite(error) or not is_valid_cache_entry(case.stem, frame, cache_root=cache_root):
                    raise ValueError("failed existing cache/reconstruction gate")
                if meta["case_stem"] != case.stem or meta["frame_idx"] != frame:
                    raise ValueError("metadata frame identity mismatch")
                n = int(meta["n_bimf"])
                for name in [f"bimf_{i}" for i in range(n)] + ["original", "residual"]:
                    path = directory / f"{name}.npy"
                    # Read only the header; mapping full arrays is unnecessary
                    # for availability checks and costly on some Windows disks.
                    with path.open("rb") as stream:
                        version = np.lib.format.read_magic(stream)
                        if version == (1, 0):
                            array_shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                        elif version == (2, 0):
                            array_shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                        else:
                            raise ValueError(f"Unsupported NPY header version {version}: {path}")
                        offset = stream.tell()
                    if array_shape != shape[:2] or dtype != np.dtype("float64"):
                        raise ValueError(f"unexpected shape/dtype: {path}")
                    if path.stat().st_size != offset + int(np.prod(array_shape)) * dtype.itemsize:
                        raise ValueError(f"unexpected array file size: {path}")
                if n < required_n_bimf:
                    raise ValueError(f"need {required_n_bimf} BIMFs, have {n}")
                valid += 1
                distribution[str(n)] += 1
                statuses[meta["status"]] += 1
                rmse.append(error)
                for i in range(min(n, 4)):
                    available[str(i)] += 1
            except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
                issues.append({"case_stem": case.stem, "frame_idx": frame, "reason": str(exc)})
    unknown = excluded_frames - seen
    if unknown:
        raise ValueError(f"Exclusions refer to frames outside the supplied dataset: {sorted(unknown)}")
    return {
        "total_frames": len(seen), "excluded_frames": len(excluded_frames),
        "usable_frames": valid, "bimf_availability": dict(available),
        "usable_bimf_count_distribution": dict(sorted(distribution.items())),
        "usable_statuses": dict(statuses), "issues": issues,
        "reconstruction_rmse": {"min": min(rmse), "max": max(rmse),
                                "mean": float(np.mean(rmse)), "median": float(np.median(rmse))} if rmse else {},
    }
