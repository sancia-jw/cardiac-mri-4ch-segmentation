#!/usr/bin/env python3
"""Train and test the training-recipe conditions in configs/improve.yaml.

Each condition x seed writes <output-root>/<condition>/seed_<seed>/ with
train_log.csv, final.pt, test_metrics.json and test_per_case.csv. When every
seed of a condition is done, the seed models are also tested as one ensemble
(mean softmax) in <condition>/ensemble/. Finished runs are skipped, so the
script can be re-run after an interruption. One row per run is appended to
<output-root>/summary.csv.

Usage:
  python scripts/train_improve.py --config configs/improve.yaml --data-root DATA \
      --output-root OUT [--extra-root EXTRA] [--seeds 42 43 44] [--runs a b] [--validate-only]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cine_4ch import improve  # noqa: E402
from cine_4ch.io import CasePair, collect_nii_paths, extract_case_id, load_volume  # noqa: E402

SUMMARY_FIELDS = ["condition", "seed", "fg", "fg_lcc", "legacy_fg"] + [
    f"dice_{c}" for c in range(1, 6)] + [f"dice_lcc_{c}" for c in range(1, 6)] + ["train_cases", "minutes"]


def discover_view(root: Path, view: str) -> list[CasePair]:
    images = collect_nii_paths(root / view / "image")
    annos = collect_nii_paths(root / view / "anno")
    return [CasePair(s, extract_case_id(s), images[s], annos[s]) for s in sorted(set(images) & set(annos))]


def split_cases(splits_csv: Path, data_root: Path) -> dict[str, list[CasePair]]:
    import pandas as pd

    by_id = {c.case_id: c for c in discover_view(data_root, "4CH_TR")}
    df = pd.read_csv(splits_csv, dtype={"case_id": str})
    out = {}
    for name in ("train", "val", "test"):
        ids = [str(i).zfill(3) for i in df[df["split"] == name]["case_id"]]
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise FileNotFoundError(f"{name}: cases {missing} not found under {data_root}/4CH_TR")
        out[name] = [by_id[i] for i in ids]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/improve.yaml")
    ap.add_argument("--data-root", type=Path, required=True, help="Folder containing 4CH_TR/")
    ap.add_argument("--extra-root", type=Path, default=None, help="Folder containing extra views, e.g. 4CH_VAL/")
    ap.add_argument("--splits-csv", type=Path, default=PROJECT_ROOT / "outputs/splits_4ch.csv")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--runs", nargs="+", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--budget-hours", type=float, default=None,
                    help="Don't start a run that is not expected to finish within this many hours of launch.")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--max-cases", type=int, default=None, help="Smoke test: cap cases per split.")
    args = ap.parse_args()

    raw = yaml.safe_load(args.config.read_text())
    seeds = args.seeds or raw["seeds"]
    conds = {name: improve.run_config_from_dict(name, spec) for name, spec in raw["conditions"].items()}
    names = args.runs or list(conds)
    unknown = [n for n in names if n not in conds]
    if unknown:
        raise SystemExit(f"Unknown conditions: {unknown}")

    splits = split_cases(args.splits_csv, args.data_root)
    extra_views = sorted({v for n in names for v in conds[n].extra_train})
    extra = {}
    for view in extra_views:
        found = discover_view(args.extra_root, view) if args.extra_root else []
        if not found:
            skipped = [n for n in names if view in conds[n].extra_train]
            print(f"WARNING: no {view} cases under --extra-root {args.extra_root}; skipping {skipped}")
            names = [n for n in names if n not in skipped]
        else:
            extra[view] = found
    if args.max_cases:
        splits = {k: v[:args.max_cases] for k, v in splits.items()}
        extra = {k: v[:args.max_cases] for k, v in extra.items()}

    shapes = []
    for case in [c for v in splits.values() for c in v] + [c for v in extra.values() for c in v]:
        shape = load_volume(case.image_path)[0].shape
        shapes.append(shape[:2])
    canvas = improve.canvas_shape(shapes)
    print(f"Cases: " + ", ".join(f"{k}={len(v)}" for k, v in {**splits, **extra}.items()))
    print(f"Native sizes: {sorted(set(shapes))} -> canvas {canvas}")
    print(f"Conditions: {names}\nSeeds: {seeds}")
    for n in names:
        c = conds[n]
        print(f"  {n}: preset={c.preset} model={c.model} context={c.context} aug={c.aug} epochs={c.epochs} "
              f"steps={c.steps_per_epoch} bs={c.batch_size} lr={c.lr} extra={c.extra_train} "
              f"train_on_val={c.train_on_val} checkpoint={c.checkpoint} upsample={c.upsample}")
    if args.validate_only:
        from collections import Counter

        loops = Counter()
        for case in [c for v in splits.values() for c in v] + [c for v in extra.values() for c in v]:
            image = load_volume(case.image_path)[0]
            loops[tuple(e - s for s, e in improve.slice_blocks(image))] += 1
        print("Cine loops per case (frames per slice position): "
              + ", ".join(f"{k} x{n}" for k, n in loops.most_common()))
        print("Validation only; nothing trained.")
        return 0

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "summary.csv"
    launch = time.time()
    cache: dict = {}
    durations: dict[str, float] = {}

    def volumes(cfg, split_names, views=()):
        key = (cfg.preset, cfg.context, cfg.upsample, tuple(split_names), tuple(views))
        if key not in cache:
            cases = [c for s in split_names for c in splits[s]] + [c for v in views for c in extra[v]]
            grid = improve.canvas_shape([improve.scaled_size(hw, cfg.upsample) for hw in shapes])
            cache[key] = improve.load_cases(cases, cfg.preset, cfg.context, grid, cfg.upsample)
        return cache[key]

    for name in names:
        cfg = conds[name]
        cond_dir = args.output_root / name
        models = []
        for seed in seeds:
            run_dir = cond_dir / f"seed_{seed}"
            if (run_dir / "test_metrics.json").exists():
                print(f"Skip {name} seed {seed}: done")
                continue
            if args.budget_hours and name in durations:
                left = args.budget_hours * 3600 - (time.time() - launch)
                if left < durations[name] * 1.15:
                    print(f"Budget: {left / 60:.0f} min left, {name} needs ~{durations[name] / 60:.0f}; stopping.")
                    return 0
            train_splits = ["train", "val"] if cfg.train_on_val else ["train"]
            train_vols = volumes(cfg, train_splits, cfg.extra_train)
            val_vols = [] if cfg.train_on_val else volumes(cfg, ["val"])
            test_vols = volumes(cfg, ["test"])
            print(f"\n=== {name} seed {seed}: {len(train_vols)} train cases, "
                  f"{sum(v.image.shape[0] for v in train_vols)} frames ===")
            t0 = time.time()
            run_dir.mkdir(parents=True, exist_ok=True)
            improve.write_json(run_dir / "config.json", {**asdict(cfg), "seed": seed, "canvas": list(train_vols[0].image.shape[1:]),
                                                         "train_cases": [v.stem for v in train_vols]})
            model = improve.train(cfg, train_vols, val_vols, device, seed, run_dir)
            metrics, rows = improve.evaluate([model], test_vols, cfg.preset, device)
            minutes = (time.time() - t0) / 60
            durations[name] = time.time() - t0
            improve.write_json(run_dir / "test_metrics.json", {**metrics, "minutes": minutes})
            with open(run_dir / "test_per_case.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            new = not summary_path.exists()
            with open(summary_path, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerow({"condition": name, "seed": seed, **{k: f"{v:.5f}" for k, v in metrics.items()},
                            "train_cases": len(train_vols), "minutes": f"{minutes:.1f}"})
            print(f"  TEST {name} seed {seed}: fg {metrics['fg']:.4f} (lcc {metrics['fg_lcc']:.4f}, "
                  f"legacy {metrics['legacy_fg']:.4f}) myo {metrics['dice_2']:.4f}  [{minutes:.1f} min]")
            del model
            torch.cuda.empty_cache()

        ens_dir = cond_dir / "ensemble"
        weights = [cond_dir / f"seed_{s}" / ("best_val.pt" if cfg.checkpoint == "best_val" else "final.pt")
                   for s in seeds]
        if len(seeds) > 1 and all(p.exists() for p in weights) and not (ens_dir / "test_metrics.json").exists():
            for p in weights:
                m = improve.build_model(cfg.model, cfg.context).to(device)
                m.load_state_dict(torch.load(p, map_location=device))
                models.append(m)
            metrics, rows = improve.evaluate(models, volumes(cfg, ["test"]), cfg.preset, device)
            ens_dir.mkdir(parents=True, exist_ok=True)
            improve.write_json(ens_dir / "test_metrics.json", metrics)
            with open(summary_path, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
                w.writerow({"condition": name, "seed": "ensemble", **{k: f"{v:.5f}" for k, v in metrics.items()}})
            print(f"  TEST {name} ensemble of {len(models)}: fg {metrics['fg']:.4f} (lcc {metrics['fg_lcc']:.4f})")
            del models
            torch.cuda.empty_cache()
    print(f"All done in {(time.time() - launch) / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
