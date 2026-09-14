#!/usr/bin/env python3
"""
Small fixed-subset screen for BEMD ablation conditions.

Uses the SAME deterministic case subset for every condition. Screening is only
to catch broken/unpromising configs — final science comes from full 15-epoch
runs on the fixed 74/16/15 split.

Requires BEMD cache. Does not launch full training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cine_4ch.ablation import AblationHyperparams
from cine_4ch.bemd_ablation import (
    default_bemd_ablation_specs,
    fixed_subset,
    resolve_device,
    train_bemd_run,
    write_bemd_summary,
)
from cine_4ch.bemd_cache import BEMD_CACHE_ROOT
from cine_4ch.config import OUTPUTS_DIR
from cine_4ch.dataset import load_split_cases

SCREEN_ROOT = OUTPUTS_DIR / "bemd_ablation_screen"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BEMD ablation subset screen.")
    p.add_argument("--output-root", type=Path, default=SCREEN_ROOT)
    p.add_argument("--cache-root", type=Path, default=BEMD_CACHE_ROOT)
    p.add_argument("--splits-csv", type=Path, default=OUTPUTS_DIR / "splits_4ch.csv")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-cases", type=int, default=8)
    p.add_argument("--max-val-cases", type=int, default=4)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--runs", type=str, nargs="*", default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.splits_csv.exists():
        print(f"Missing {args.splits_csv}", file=sys.stderr)
        return 1

    train_all = load_split_cases(args.splits_csv, "train")
    val_all = load_split_cases(args.splits_csv, "val")
    train_sub = fixed_subset(train_all, args.max_train_cases, args.seed)
    val_sub = fixed_subset(val_all, args.max_val_cases, args.seed + 3)

    subset_meta = {
        "note": (
            "Screening subset only. Do not treat these Dice scores as the final "
            "scientific result; use full bemd_ablation runs."
        ),
        "seed": args.seed,
        "epochs": args.epochs,
        "train_stems": [c.stem for c in train_sub],
        "val_stems": [c.stem for c in val_sub],
        "same_subset_all_conditions": True,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "subset_cases.json").write_text(
        json.dumps(subset_meta, indent=2), encoding="utf-8"
    )
    print("Screen subset (shared across all conditions):")
    print(f"  train ({len(train_sub)}): {subset_meta['train_stems']}")
    print(f"  val   ({len(val_sub)}): {subset_meta['val_stems']}")

    catalog = default_bemd_ablation_specs()
    if args.runs:
        selected = set(args.runs)
        specs = [s for s in catalog if s.run_id in selected]
    else:
        specs = catalog

    hyperparams = AblationHyperparams(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    device = resolve_device(args.device)
    rows = []
    for spec in specs:
        run_dir = args.output_root / spec.run_id
        val_summary = train_bemd_run(
            spec,
            hyperparams,
            args.splits_csv,
            run_dir,
            device,
            resume=args.resume,
            bemd_cache_root=Path(args.cache_root),
            train_cases=train_sub,
            val_cases=val_sub,
        )
        rows.append(
            {
                "run_id": spec.run_id,
                "bimf_indices": str(list(spec.bimf_indices)),
                "best_epoch": val_summary.get("best_epoch", ""),
                "val_dice_mean_foreground": f"{val_summary.get('val_dice_mean_foreground', 0):.6f}",
                "val_dice_mean": f"{val_summary.get('val_dice_mean', 0):.6f}",
                "seed": args.seed,
                "epochs": args.epochs,
                "screen_only": True,
            }
        )

    write_bemd_summary(rows, args.output_root / "screening_summary.csv")
    (args.output_root / "screening_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(f"Screen summary: {args.output_root / 'screening_summary.csv'}")
    print("Reminder: full-run comparison lives under outputs/bemd_ablation/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
