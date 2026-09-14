#!/usr/bin/env python3
"""Aggregate BEMD ablation run metrics into one comparison table (CSV + JSON)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cine_4ch.bemd_ablation import BEMD_ABLATION_ROOT, default_bemd_ablation_specs, write_bemd_summary
from cine_4ch.config import LABEL_NAMES, NUM_CLASSES


def _read_metric_csv(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out[row["metric"]] = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def _best_from_log(run_dir: Path) -> dict:
    log = run_dir / "training_log.csv"
    best = {}
    if not log.exists():
        return best
    best_fg = -1.0
    with open(log, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("epoch"):
                continue
            fg = float(row["val_dice_mean_foreground"])
            if fg > best_fg:
                best_fg = fg
                best = {
                    "best_epoch": int(row["epoch"]),
                    "val_dice_mean": float(row["val_dice_mean"]),
                    "val_dice_mean_foreground": fg,
                }
                for i in range(NUM_CLASSES):
                    key = f"val_dice_class_{i}"
                    if key in row:
                        best[key] = float(row[key])
    return best


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate bemd_ablation metrics.")
    p.add_argument("--output-root", type=Path, default=BEMD_ABLATION_ROOT)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    catalog = default_bemd_ablation_specs()
    rows = []
    for spec in catalog:
        run_dir = args.output_root / spec.run_id
        if not run_dir.is_dir():
            continue
        val = _best_from_log(run_dir)
        test = _read_metric_csv(run_dir / "test_metrics.csv")
        cfg = {}
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        row = {
            "run_id": spec.run_id,
            "condition": spec.run_id,
            "bimf_indices": str(list(spec.bimf_indices)),
            "description": spec.description,
            "seed": (cfg.get("hyperparams") or {}).get("seed", args.seed),
            "selected_checkpoint": str(run_dir / "checkpoints" / "best.pt"),
            "best_epoch": val.get("best_epoch", ""),
            "val_dice_mean_foreground": val.get("val_dice_mean_foreground", ""),
            "val_dice_mean": val.get("val_dice_mean", ""),
            "test_dice_mean_foreground": test.get("test_dice_mean_foreground", ""),
            "test_dice_mean": test.get("test_dice_mean", ""),
            "config_path": str(cfg_path) if cfg_path.exists() else "",
        }
        for i in range(NUM_CLASSES):
            row[f"test_dice_class_{i}_{LABEL_NAMES[i]}"] = test.get(f"test_dice_class_{i}", "")
        rows.append(row)

    by_id = {r["run_id"]: r for r in rows}
    if "original" in by_id:
        try:
            ov = float(by_id["original"]["val_dice_mean_foreground"])
            ot = float(by_id["original"]["test_dice_mean_foreground"])
        except (TypeError, ValueError):
            ov = ot = None
        if ot is not None:
            for r in rows:
                try:
                    r["delta_val_fg_vs_original"] = float(r["val_dice_mean_foreground"]) - ov
                    r["delta_test_fg_vs_original"] = float(r["test_dice_mean_foreground"]) - ot
                except (TypeError, ValueError):
                    r["delta_val_fg_vs_original"] = ""
                    r["delta_test_fg_vs_original"] = ""

    out_csv = args.output_root / "ablation_summary.csv"
    out_json = args.output_root / "ablation_summary.json"
    write_bemd_summary(rows, out_csv)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out_csv} ({len(rows)} runs)")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
