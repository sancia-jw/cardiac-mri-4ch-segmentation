#!/usr/bin/env python3
"""Convert CMRxRecon (MICCAI 2023) long-axis cine annotations into 4CH training cases.

CMRxRecon annotates the long-axis cine (3CH, 2CH and 4CH slice groups in one
volume) with LA=1, RA=2, LV=3, RV=4 and no myocardium (Wang et al. 2024,
Sci Data). Single-coil SegmentROI/Pxxx folders hold the reconstructed images
(cine_lax_forlabel*.nii*) and labels (cine_lax_label*.nii*).

For each subject this script:
1. picks the slice group showing all four chambers (largest RA area),
2. keeps the annotated frames,
3. maps labels to this project's ids (LV cavity 1, RV 3, RA 4, LA 5; no 2),
4. resamples to --spacing mm,
5. flips/rotates (one of 8 options) so LV->LA and LV->RV point the way they
   do in CMR-MULTI 4CH (--reference-anno, mean over its cases),
6. crops --size pixels around the heart,
and writes <out>/4CH_CMRX/{image,anno}/CMRX_<subject>.nii.gz as (H, W, frames)
plus <out>/4CH_CMRX/conversion.csv. Upload <out> as a Kaggle dataset.

Usage:
  python scripts/convert_cmrxrecon.py --root .../SingleCoil/Cine/TrainingSet \
      --reference-anno CMR-MULTI/CINE_MULTI/4CH_TR/anno --out cmrx_4ch
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CMRX_TO_PROJECT = {1: 5, 2: 4, 3: 1, 4: 3}  # LA, RA, LV, RV -> project ids
LV, RV, RA, LA = 1, 3, 4, 5


def dihedral(arr: np.ndarray, k: int) -> np.ndarray:
    """One of 8 flips/rotations of the first two axes: k in 0..7."""
    out = np.rot90(arr, k % 4, axes=(0, 1))
    return np.flip(out, axis=0) if k >= 4 else out


def chamber_vectors(label: np.ndarray) -> np.ndarray | None:
    """Unit vectors LV->LA and LV->RV from label centroids of a 2D (or stacked) map; None if missing."""
    cents = {}
    for c in (LV, RV, LA):
        pts = np.argwhere(label == c)
        if len(pts) == 0:
            return None
        cents[c] = pts[:, :2].mean(0)
    vec = np.stack([cents[LA] - cents[LV], cents[RV] - cents[LV]])
    return vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8)


def reference_vectors(anno_dir: Path, n: int = 20) -> np.ndarray:
    vecs = []
    for path in sorted(anno_dir.glob("*.nii*"))[:n]:
        lab = np.rint(np.asarray(nib.load(str(path)).dataobj)).astype(np.int64).squeeze()
        v = chamber_vectors(lab)
        if v is not None:
            vecs.append(v)
    if not vecs:
        raise SystemExit(f"No usable reference labels in {anno_dir}")
    mean = np.mean(vecs, axis=0)
    return mean / np.linalg.norm(mean, axis=1, keepdims=True)


def best_orientation(label: np.ndarray, ref: np.ndarray) -> int:
    scores = []
    for k in range(8):
        v = chamber_vectors(dihedral(label, k))
        scores.append(-np.inf if v is None else float((v * ref).sum()))
    return int(np.argmax(scores))


def crop_around(image: np.ndarray, label: np.ndarray, size: tuple[int, int]):
    """Crop/pad the first two axes to ``size`` centred on the foreground bounding box."""
    fg = np.argwhere(label.max(axis=2) > 0) if label.ndim == 3 else np.argwhere(label > 0)
    centre = ((fg.min(0) + fg.max(0)) // 2)[:2]
    pad = [(s, s) for s in size] + [(0, 0)] * (image.ndim - 2)
    image = np.pad(image, pad, mode="constant", constant_values=float(image.min()))
    label = np.pad(label, pad, mode="constant")
    top, left = centre[0] + size[0] - size[0] // 2, centre[1] + size[1] - size[1] // 2
    return image[top:top + size[0], left:left + size[1]], label[top:top + size[0], left:left + size[1]]


def find(folder: Path, pattern: str, exclude: str | None = None) -> Path | None:
    hits = [p for p in sorted(folder.glob(pattern)) if not (exclude and exclude in p.name)]
    return hits[0] if hits else None


def convert_subject(img_path: Path, lab_path: Path, ref: np.ndarray, spacing: float, size):
    img_nii, lab_nii = nib.load(str(img_path)), nib.load(str(lab_path))
    image = np.asarray(img_nii.dataobj, dtype=np.float32)
    raw = np.rint(np.asarray(lab_nii.dataobj)).astype(np.int64)
    if image.ndim == 3:  # (x, y, slice group): one frame
        image, raw = image[..., None], raw[..., None]
    if image.shape != raw.shape:
        raise ValueError(f"image {image.shape} vs label {raw.shape}")
    label = np.zeros_like(raw)
    for src, dst in CMRX_TO_PROJECT.items():
        label[raw == src] = dst
    ra_area = [(label[:, :, z] == RA).sum() for z in range(label.shape[2])]
    z = int(np.argmax(ra_area))
    if ra_area[z] == 0:
        raise ValueError("no slice group shows the right atrium")
    image, label = image[:, :, z], label[:, :, z]  # (x, y, t)
    frames = [t for t in range(label.shape[2]) if {LV, RV, LA, RA} <= set(np.unique(label[:, :, t]))]
    if not frames:
        raise ValueError("no frame with all four chambers")
    image, label = image[:, :, frames], label[:, :, frames]
    zoom = np.array(img_nii.header.get_zooms()[:2], dtype=float) / spacing
    image = ndimage.zoom(image, (*zoom, 1), order=1)
    label = ndimage.zoom(label, (*zoom, 1), order=0)
    k = best_orientation(label, ref)
    image, label = dihedral(image, k), dihedral(label, k)
    image, label = crop_around(image, label, size)
    return image, label.astype(np.uint8), {"slice_group": z, "frames": len(frames), "orientation": k,
                                           **{f"area_{c}": int((label == c).sum()) for c in (LV, RV, RA, LA)}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="CMRxRecon SingleCoil/Cine/TrainingSet (or a folder of P* subject folders)")
    ap.add_argument("--reference-anno", type=Path, required=True, help="CMR-MULTI 4CH_TR/anno")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spacing", type=float, default=1.2, help="Target in-plane spacing, mm (CMR-MULTI ~1.0-1.35)")
    ap.add_argument("--size", type=int, nargs=2, default=[176, 160])
    args = ap.parse_args()

    base = args.root / "SegmentROI" if (args.root / "SegmentROI").is_dir() else args.root
    subjects = sorted(p for p in base.iterdir() if p.is_dir())
    ref = reference_vectors(args.reference_anno)
    out = args.out / "4CH_CMRX"
    (out / "image").mkdir(parents=True, exist_ok=True)
    (out / "anno").mkdir(parents=True, exist_ok=True)
    rows = []
    for subj in subjects:
        img = find(subj, "*lax*forlabel*.nii*")
        lab = find(subj, "*lax*label*.nii*", exclude="forlabel")
        if not img or not lab:
            rows.append({"subject": subj.name, "status": "missing lax image/label"})
            continue
        try:
            image, label, info = convert_subject(img, lab, ref, args.spacing, tuple(args.size))
        except ValueError as exc:
            rows.append({"subject": subj.name, "status": str(exc)})
            continue
        affine = np.diag([args.spacing, args.spacing, 1.0, 1.0])
        nib.save(nib.Nifti1Image(image.astype(np.float32), affine), str(out / "image" / f"CMRX_{subj.name}.nii.gz"))
        nib.save(nib.Nifti1Image(label, affine), str(out / "anno" / f"CMRX_{subj.name}.nii.gz"))
        rows.append({"subject": subj.name, "status": "ok", **info})
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "subject", k != "status", k))
    with open(out / "conversion.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"Converted {len(ok)}/{len(rows)} subjects -> {out}")
    if ok:
        print("Orientation used (k: count):", {k: sum(r["orientation"] == k for r in ok) for k in range(8)})
        # If CMRxRecon's LV label included the myocardium, LV/LA would match CMR-MULTI's (cavity+myo)/LA.
        ratio = np.median([r[f"area_{LV}"] / max(r[f"area_{LA}"], 1) for r in ok])
        cav, full = [], []
        for path in sorted(args.reference_anno.glob("*.nii*"))[:20]:
            lab = np.rint(np.asarray(nib.load(str(path)).dataobj)).astype(np.int64)
            la = max((lab == LA).sum(), 1)
            cav.append((lab == LV).sum() / la)
            full.append(((lab == LV) | (lab == 2)).sum() / la)
        print(f"LV/LA area ratio: CMRxRecon {ratio:.2f} | CMR-MULTI cavity {np.median(cav):.2f}, "
              f"cavity+myocardium {np.median(full):.2f}. If CMRxRecon is near the second, its LV label "
              "includes the myocardium: set external_merge to [1, 2] instead of [0, 2].")
        print("Frames per subject:", sorted({r["frames"] for r in ok}))
    for r in rows:
        if r["status"] != "ok":
            print(f"  {r['subject']}: {r['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
