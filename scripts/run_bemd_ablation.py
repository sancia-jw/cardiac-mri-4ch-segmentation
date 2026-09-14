#!/usr/bin/env python3
"""
BEMD (square-pad) preprocessing ablation for 4CH cardiac MRI segmentation.

Conditions (independent variable = which BIMF(s) removed):
  original, subtract_bimf_0, subtract_bimf_1, subtract_bimf_2,
  subtract_bimf_3, subtract_bimf_0_1

Requires a completed BEMD cache (scripts/preprocess_bemd.py). Does NOT run BEMD
during training. Does NOT overwrite outputs/emd_ablation/.

Results: outputs/bemd_ablation/<run_id>/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cine_4ch.ablation import AblationHyperparams
from cine_4ch.bemd_ablation import (
    BEMD_ABLATION_ROOT,
    default_bemd_ablation_specs,
    evaluate_bemd_test,
    load_val_summary_from_run,
    resolve_device,
    train_bemd_run,
    write_bemd_summary,
)
from cine_4ch.bemd_cache import BEMD_CACHE_ROOT
from cine_4ch.config import LABEL_NAMES, NUM_CLASSES, OUTPUTS_DIR


def merge_summary_rows(summary_path: Path, new_rows: list[dict], catalog_ids: list[str]) -> list[dict]:
    by_id: dict[str, dict] = {}
    if summary_path.exists():
        with open(summary_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                by_id[row["run_id"]] = row
    for row in new_rows:
        by_id[row["run_id"]] = row
    extras = [rid for rid in by_id if rid not in catalog_ids]
    return [by_id[rid] for rid in catalog_ids + extras if rid in by_id]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run BEMD BIMF-removal ablation.")
    p.add_argument("--config", type=Path, default=None, help="Optional YAML (runs / hyperparams).")
    p.add_argument("--output-root", type=Path, default=BEMD_ABLATION_ROOT)
    p.add_argument("--cache-root", type=Path, default=BEMD_CACHE_ROOT)
    p.add_argument("--splits-csv", type=Path, default=OUTPUTS_DIR / "splits_4ch.csv")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--runs", type=str, nargs="*", default=None)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--num-overlays", type=int, default=10)
    return p.parse_args()


def _apply_yaml(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML required for --config") from exc
    data = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    train = data.get("train", data)
    for key, attr in [
        ("output_root", "output_root"),
        ("cache_root", "cache_root"),
        ("splits_csv", "splits_csv"),
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("lr", "lr"),
        ("seed", "seed"),
        ("device", "device"),
        ("runs", "runs"),
    ]:
        if key in train and train[key] is not None:
            val = train[key]
            if attr in ("output_root", "cache_root", "splits_csv"):
                val = Path(val)
            setattr(args, attr, val)
    return args


def main() -> int:
    args = _apply_yaml(parse_args())
    if not args.splits_csv.exists():
        print(f"Split file not found: {args.splits_csv}", file=sys.stderr)
        return 1
    if not Path(args.cache_root).is_dir():
        print(
            f"BEMD cache not found: {args.cache_root}\n"
            "Run: python scripts/preprocess_bemd.py",
            file=sys.stderr,
        )
        return 1

    hyperparams = AblationHyperparams(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Hyperparams: {hyperparams}")

    catalog = default_bemd_ablation_specs()
    if args.runs:
        selected = set(args.runs)
        specs = [s for s in catalog if s.run_id in selected]
        missing = selected - {s.run_id for s in specs}
        if missing:
            print(f"Unknown run ids: {sorted(missing)}", file=sys.stderr)
            return 1
    else:
        specs = catalog

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for spec in specs:
        run_dir = args.output_root / spec.run_id
        if not args.skip_train:
            val_summary = train_bemd_run(
                spec,
                hyperparams,
                args.splits_csv,
                run_dir,
                device,
                resume=args.resume,
                bemd_cache_root=Path(args.cache_root),
            )
        else:
            val_summary = load_val_summary_from_run(run_dir)

        test_metrics = evaluate_bemd_test(
            spec,
            hyperparams,
            args.splits_csv,
            run_dir,
            device,
            num_overlays=args.num_overlays,
            seed=args.seed,
            bemd_cache_root=Path(args.cache_root),
        )

        row = {
            "run_id": spec.run_id,
            "condition": spec.run_id,
            "preprocessing_mode": spec.mode,
            "bimf_indices": str(list(spec.bimf_indices)),
            "description": spec.description,
            "decomposition_method": "bemd_default_square_pad",
            "reconstruction": "gastro_style_minmax_bimf_then_finalize",
            "epochs": hyperparams.epochs,
            "batch_size": hyperparams.batch_size,
            "lr": hyperparams.lr,
            "seed": hyperparams.seed,
            "best_epoch": val_summary.get("best_epoch", ""),
            "selected_checkpoint": str(run_dir / "checkpoints" / "best.pt"),
            "val_dice_mean": _fmt(val_summary.get("val_dice_mean")),
            "val_dice_mean_foreground": _fmt(val_summary.get("val_dice_mean_foreground")),
            "test_dice_mean": f"{test_metrics['dice_mean']:.6f}",
            "test_dice_mean_foreground": f"{test_metrics['dice_mean_foreground']:.6f}",
        }
        for i in range(NUM_CLASSES):
            row[f"test_dice_class_{i}_{LABEL_NAMES[i]}"] = f"{test_metrics[f'dice_class_{i}']:.6f}"
        summary_rows.append(row)

    summary_path = args.output_root / "ablation_summary.csv"
    merged = merge_summary_rows(summary_path, summary_rows, [s.run_id for s in catalog])

    # Delta vs original
    by_id = {r["run_id"]: r for r in merged}
    if "original" in by_id:
        try:
            orig_val = float(by_id["original"]["val_dice_mean_foreground"])
            orig_test = float(by_id["original"]["test_dice_mean_foreground"])
        except (KeyError, TypeError, ValueError):
            orig_val = orig_test = None
        if orig_test is not None:
            for r in merged:
                try:
                    r["delta_val_fg_vs_original"] = f"{float(r['val_dice_mean_foreground']) - orig_val:.6f}"
                    r["delta_test_fg_vs_original"] = f"{float(r['test_dice_mean_foreground']) - orig_test:.6f}"
                except (KeyError, TypeError, ValueError):
                    r["delta_val_fg_vs_original"] = ""
                    r["delta_test_fg_vs_original"] = ""

    write_bemd_summary(merged, summary_path)
    # Machine-readable JSON twin
    import json

    (args.output_root / "ablation_summary.json").write_text(
        json.dumps(merged, indent=2), encoding="utf-8"
    )
    print(f"\nBEMD ablation summary: {summary_path} ({len(merged)} runs)")
    return 0


def _fmt(v) -> str:
    if v is None or v == "":
        return ""
    try:
        return f"{float(v):.6f}"
    except (TypeError, ValueError):
        return str(v)


if __name__ == "__main__":
    raise SystemExit(main())
