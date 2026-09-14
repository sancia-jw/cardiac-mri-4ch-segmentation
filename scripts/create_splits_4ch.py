#!/usr/bin/env python3
"""
Create case-level train/val/test splits for CINE 4CH_TR.

Each CINE_4CH_XXX file is treated as one case (no shared patient IDs in filenames).
Splits: 70% train, 15% validation, 15% test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cine_4ch.config import DATA_ROOT, OUTPUTS_DIR
from cine_4ch.io import discover_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create case-level splits for 4CH_TR.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "splits_4ch.csv",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < 0:
        print("train_ratio + val_ratio must be <= 1.0", file=sys.stderr)
        return 1

    try:
        pairs, _, _ = discover_cases(args.data_root)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    case_ids = sorted({case.case_id for case in pairs})
    rng = np.random.default_rng(args.seed)
    shuffled = case_ids.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * args.train_ratio))
    n_val = int(round(n * args.val_ratio))
    # Remaining cases go to test so every case is assigned exactly once.
    n_test = n - n_train - n_val

    train_ids = set(shuffled[:n_train])
    val_ids = set(shuffled[n_train : n_train + n_val])
    test_ids = set(shuffled[n_train + n_val :])

    rows = []
    for case in sorted(pairs, key=lambda c: c.case_id):
        if case.case_id in train_ids:
            split = "train"
        elif case.case_id in val_ids:
            split = "val"
        else:
            split = "test"
        rows.append(
            {
                "case_id": case.case_id,
                "stem": case.stem,
                "split": split,
                "image_path": str(case.image_path),
                "anno_path": str(case.anno_path),
            }
        )

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Total cases: {n}")
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}  Test: {len(test_ids)}")
    print(f"Saved splits to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
