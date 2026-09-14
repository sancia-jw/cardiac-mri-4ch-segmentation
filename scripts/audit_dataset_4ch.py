#!/usr/bin/env python3
"""
Audit CMR-MULTI CINE_MULTI/4CH_TR image/annotation pairs.

Writes a per-case summary to outputs/dataset_audit_4ch.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as: python scripts/audit_dataset_4ch.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cine_4ch.config import (
    DATA_ROOT,
    EXPECTED_FOREGROUND_LABELS,
    LABEL_NAMES,
    NUM_CLASSES,
    OUTPUTS_DIR,
)
from cine_4ch.io import discover_cases, load_pair


def audit_case(case) -> dict:
    row = {
        "stem": case.stem,
        "case_id": case.case_id,
        "image_path": str(case.image_path),
        "anno_path": str(case.anno_path),
        "corrupt": False,
        "shape_match": False,
        "image_shape": "",
        "label_shape": "",
        "unique_labels": "",
        "missing_foreground_labels": "",
        "unknown_labels": "",
        "flags": "",
    }

    try:
        image, label, spacing = load_pair(case)
        row["image_shape"] = str(tuple(image.shape))
        row["label_shape"] = str(tuple(label.shape))
        row["spacing_mm"] = str(tuple(spacing))
        row["shape_match"] = image.shape == label.shape

        unique = sorted(int(v) for v in np.unique(label))
        row["unique_labels"] = ",".join(str(v) for v in unique)

        for label_id in range(NUM_CLASSES):
            row[f"label_{label_id}_voxels"] = int((label == label_id).sum())

        missing_fg = [lid for lid in EXPECTED_FOREGROUND_LABELS if lid not in unique]
        unknown = [v for v in unique if v < 0 or v >= NUM_CLASSES]
        row["missing_foreground_labels"] = ",".join(str(v) for v in missing_fg)
        row["unknown_labels"] = ",".join(str(v) for v in unknown)

        flags = []
        if not row["shape_match"]:
            flags.append("shape_mismatch")
        if missing_fg:
            flags.append("missing_foreground_labels")
        if unknown:
            flags.append("unknown_labels")
        if label.ndim not in (2, 3):
            flags.append("unexpected_ndim")
        row["flags"] = ";".join(flags)
    except Exception as exc:
        row["corrupt"] = True
        row["flags"] = f"load_error:{exc}"

    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CINE_MULTI/4CH_TR dataset.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "dataset_audit_4ch.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        pairs, missing_image, missing_anno = discover_cases(args.data_root)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Image files found:   {len(missing_anno) + len(pairs)}")
    print(f"Annotation files:    {len(missing_image) + len(pairs)}")
    print(f"Paired cases:        {len(pairs)}")
    print(f"Missing image:       {len(missing_image)}")
    print(f"Missing annotation:  {len(missing_anno)}")
    if missing_image:
        print(f"  anno without image (first 5): {missing_image[:5]}")
    if missing_anno:
        print(f"  image without anno (first 5): {missing_anno[:5]}")
    print()

    rows = [audit_case(case) for case in pairs]
    df = pd.DataFrame(rows)

    # Print shape summary
    shape_counts = df["image_shape"].value_counts()
    print("Image shape distribution:")
    for shape, count in shape_counts.items():
        print(f"  {shape}: {count}")
    print()

    flagged = df[(df["corrupt"]) | (df["flags"].astype(str) != "")]
    print(f"Flagged cases: {len(flagged)}")
    if len(flagged):
        print(flagged[["stem", "flags", "corrupt"]].to_string(index=False))
    print()

    print("Label voxel totals across dataset:")
    for label_id in range(NUM_CLASSES):
        col = f"label_{label_id}_voxels"
        if col in df.columns:
            total = df[col].fillna(0).sum()
            print(f"  {label_id} ({LABEL_NAMES[label_id]}): {int(total):,}")
    print()

    df.to_csv(args.output, index=False)
    print(f"Saved audit CSV to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
