#!/usr/bin/env python3
"""
Randomly visualize CINE 4CH cases (raw image, mask, overlay).

Saves figures to outputs/visual_checks/.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cine_4ch.config import DATA_ROOT, OUTPUTS_DIR
from cine_4ch.io import choose_representative_frame, discover_cases, extract_frame, load_pair
from cine_4ch.viz import save_case_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual QC for CINE 4CH_TR cases.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "visual_checks",
    )
    parser.add_argument("--num-cases", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible reruns.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        pairs, _, _ = discover_cases(args.data_root)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    chosen = rng.sample(pairs, k=min(args.num_cases, len(pairs)))

    print(f"Saving {len(chosen)} figures to {args.output_dir} (seed={args.seed})")
    for case in sorted(chosen, key=lambda c: c.stem):
        image, label, _ = load_pair(case)
        frame_idx = choose_representative_frame(label)
        image_slice = extract_frame(image, frame_idx)
        label_slice = extract_frame(label, frame_idx)
        unique_labels = sorted(int(v) for v in np.unique(label_slice))

        out_path = args.output_dir / f"{case.stem}.png"
        save_case_figure(image_slice, label_slice, case.stem, frame_idx, out_path, unique_labels)
        print(f"  {case.stem} -> frame {frame_idx} -> {out_path.name}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
