#!/usr/bin/env python3
"""
EMD preprocessing ablation: original U-Net vs EMD-enhanced inputs.

Uses fixed hyperparameters and the existing train/val/test split. Only
preprocessing mode changes between runs.

Results are saved under outputs/emd_ablation/<run_id>/ with:
  - config.json, training_log.csv, val_curves.png
  - checkpoints/best.pt
  - test_metrics.csv, test_predictions/, qualitative_overlays/

A final comparison table is written to outputs/emd_ablation/ablation_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cine_4ch.ablation import (
    AblationHyperparams,
    default_ablation_runs,
    evaluate_test_and_visualize,
    load_val_summary_from_run,
    resolve_device,
    train_run,
    write_ablation_summary,
)
from cine_4ch.config import LABEL_NAMES, NUM_CLASSES, OUTPUTS_DIR


def merge_summary_rows(summary_path: Path, new_rows: list[dict]) -> list[dict]:
    """Update existing ablation_summary.csv in place for the runs just completed."""
    by_id: dict[str, dict] = {}
    if summary_path.exists():
        with open(summary_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                by_id[row["run_id"]] = row
    for row in new_rows:
        by_id[row["run_id"]] = row
    # Prefer catalog order, then any extras.
    ordered_ids = [r.run_id for r in default_ablation_runs()]
    extras = [rid for rid in by_id if rid not in ordered_ids]
    return [by_id[rid] for rid in ordered_ids + extras if rid in by_id]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EMD preprocessing ablation experiments.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUTS_DIR / "emd_ablation",
    )
    parser.add_argument(
        "--splits-csv",
        type=Path,
        default=OUTPUTS_DIR / "splits_4ch.csv",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--runs",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of run_ids (default: all).",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training; only run test eval/visualization for existing runs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last completed epoch in training_log.csv.",
    )
    parser.add_argument("--num-overlays", type=int, default=10)
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=None,
        help="Parallel workers for EMD disk-cache precompute (default: min(8, cpu-1)).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.splits_csv.exists():
        print(f"Split file not found: {args.splits_csv}", file=sys.stderr)
        return 1

    hyperparams = AblationHyperparams(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Fixed hyperparameters: {hyperparams}")

    all_runs = default_ablation_runs()
    if args.runs:
        selected = set(args.runs)
        runs = [r for r in all_runs if r.run_id in selected]
        missing = selected - {r.run_id for r in runs}
        if missing:
            print(f"Unknown run ids: {sorted(missing)}", file=sys.stderr)
            return 1
    else:
        runs = all_runs

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for spec in runs:
        run_dir = args.output_root / spec.run_id

        if not args.skip_train:
            val_summary = train_run(
                spec,
                hyperparams,
                args.splits_csv,
                run_dir,
                device,
                resume=args.resume,
                preprocess_workers=args.preprocess_workers,
            )
        else:
            val_summary = load_val_summary_from_run(run_dir)

        test_metrics = evaluate_test_and_visualize(
            spec,
            hyperparams,
            args.splits_csv,
            run_dir,
            device,
            num_overlays=args.num_overlays,
            seed=args.seed,
            preprocess_workers=args.preprocess_workers,
        )

        row = {
            "run_id": spec.run_id,
            "preprocessing_mode": spec.emd_config.mode,
            "imf_indices": str(spec.emd_config.imf_indices),
            "description": spec.description,
            "epochs": hyperparams.epochs,
            "batch_size": hyperparams.batch_size,
            "lr": hyperparams.lr,
            "seed": hyperparams.seed,
            "best_epoch": val_summary.get("best_epoch", ""),
            "val_dice_mean": f"{val_summary.get('val_dice_mean', 0):.6f}" if val_summary.get("val_dice_mean") is not None and val_summary.get("val_dice_mean") != "" else "",
            "val_dice_mean_foreground": f"{val_summary.get('val_dice_mean_foreground', 0):.6f}" if val_summary.get("val_dice_mean_foreground") is not None and val_summary.get("val_dice_mean_foreground") != "" else "",
            "test_dice_mean": f"{test_metrics['dice_mean']:.6f}",
            "test_dice_mean_foreground": f"{test_metrics['dice_mean_foreground']:.6f}",
        }
        for i in range(NUM_CLASSES):
            row[f"test_dice_class_{i}_{LABEL_NAMES[i]}"] = f"{test_metrics[f'dice_class_{i}']:.6f}"
        summary_rows.append(row)

    summary_path = args.output_root / "ablation_summary.csv"
    merged = merge_summary_rows(summary_path, summary_rows)
    write_ablation_summary(merged, summary_path)
    print(f"\nAblation summary saved to {summary_path} ({len(merged)} runs)")

    # Direct comparison vs original baseline when both are present.
    by_id = {r["run_id"]: r for r in merged}
    if "original" in by_id and summary_rows:
        orig_fg = float(by_id["original"]["test_dice_mean_foreground"])
        for row in summary_rows:
            if row["run_id"] == "original":
                continue
            run_fg = float(row["test_dice_mean_foreground"])
            delta = run_fg - orig_fg
            print(
                f"Comparison vs original: {row['run_id']} test_fg={run_fg:.6f} "
                f"original={orig_fg:.6f} delta={delta:+.6f}"
            )
            cmp_path = args.output_root / row["run_id"] / "comparison_vs_original.txt"
            cmp_path.write_text(
                (
                    f"run_id: {row['run_id']}\n"
                    f"test_dice_mean_foreground: {run_fg:.6f}\n"
                    f"original_test_dice_mean_foreground: {orig_fg:.6f}\n"
                    f"delta_vs_original: {delta:+.6f}\n"
                    f"beats_original: {run_fg > orig_fg}\n"
                    f"best_epoch: {row.get('best_epoch', '')}\n"
                ),
                encoding="utf-8",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
