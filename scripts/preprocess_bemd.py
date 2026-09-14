#!/usr/bin/env python3
"""
Precompute persistent ``bemd_default_square_pad`` decompositions.

Writes incrementally under outputs/bemd_cache/bemd_default_square_pad/ and an
audit CSV/JSON summary. Safe to resume (skips valid cache entries).

Training never calls this; run it once (or resume) before bemd ablation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cine_4ch.bemd_cache import BEMD_CACHE_ROOT, default_workers, preprocess_cases
from cine_4ch.config import DATA_ROOT, OUTPUTS_DIR
from cine_4ch.dataset import load_split_cases
from cine_4ch.io import discover_cases
from src.preprocessing.bemd_square_pad import BEMDConfig, METHOD_ID


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache BEMD (square-pad) decompositions.")
    p.add_argument("--config", type=Path, default=None, help="Optional YAML override (keys mirror flags).")
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--cache-root", type=Path, default=BEMD_CACHE_ROOT)
    p.add_argument("--splits-csv", type=Path, default=OUTPUTS_DIR / "splits_4ch.csv")
    p.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "val", "test"],
        help="Which split(s) to preprocess (default: all paired cases).",
    )
    p.add_argument("--max-frames", type=int, default=None, help="Cap total frames (smoke tests).")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--end-index", type=int, default=None)
    p.add_argument("--workers", type=int, default=None, help="Process workers (default: conservative).")
    p.add_argument("--force", action="store_true", help="Recompute even if valid cache exists.")
    p.add_argument("--resume", action="store_true", default=True, help="Skip valid cache (default).")
    p.add_argument("--no-resume", action="store_true", help="Alias for --force.")
    p.add_argument(
        "--audit-csv",
        type=Path,
        default=None,
        help="Default: <cache-root>/preprocess_audit.csv",
    )
    p.add_argument("--max-imf", type=int, default=4)
    p.add_argument("--mean-thr", type=float, default=0.01)
    p.add_argument("--mse-thr", type=float, default=0.01)
    p.add_argument("--FIXE", type=int, default=1)
    p.add_argument("--FIXE-H", type=int, default=0)
    return p.parse_args()


def _apply_yaml(args: argparse.Namespace) -> argparse.Namespace:
    if args.config is None:
        return args
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML required for --config. pip install pyyaml") from exc
    data = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    # Nested preprocess: section or flat keys
    section = data.get("preprocess", data)
    mapping = {
        "data_root": "data_root",
        "cache_root": "cache_root",
        "splits_csv": "splits_csv",
        "split": "split",
        "max_frames": "max_frames",
        "start_index": "start_index",
        "end_index": "end_index",
        "workers": "workers",
        "force": "force",
        "max_imf": "max_imf",
        "mean_thr": "mean_thr",
        "mse_thr": "mse_thr",
        "FIXE": "FIXE",
        "FIXE_H": "FIXE_H",
    }
    for yaml_key, attr in mapping.items():
        if yaml_key in section and section[yaml_key] is not None:
            val = section[yaml_key]
            if attr in ("data_root", "cache_root", "splits_csv", "audit_csv") and not isinstance(val, Path):
                val = Path(val)
            setattr(args, attr, val)
    return args


def main() -> int:
    args = _apply_yaml(parse_args())
    if args.no_resume:
        args.force = True
    workers = default_workers() if args.workers is None else max(1, int(args.workers))
    audit_csv = args.audit_csv or (Path(args.cache_root) / "preprocess_audit.csv")

    cfg = BEMDConfig(
        max_imf=args.max_imf,
        mean_thr=args.mean_thr,
        mse_thr=args.mse_thr,
        FIXE=args.FIXE,
        FIXE_H=args.FIXE_H,
    )

    if args.split == "all":
        cases, _, _ = discover_cases(args.data_root)
    else:
        if not args.splits_csv.exists():
            print(f"Missing splits CSV: {args.splits_csv}", file=sys.stderr)
            return 1
        cases = load_split_cases(args.splits_csv, args.split, data_root=args.data_root)

    print(f"Method: {METHOD_ID}")
    print(f"Cases: {len(cases)}  workers={workers}  force={args.force}")
    print(f"Cache root: {args.cache_root}")

    summary = preprocess_cases(
        cases,
        cache_root=Path(args.cache_root),
        config=cfg,
        force=bool(args.force),
        workers=workers,
        max_frames=args.max_frames,
        start_index=args.start_index,
        end_index=args.end_index,
        audit_csv=Path(audit_csv),
    )

    print(json.dumps({k: summary[k] for k in summary if k not in ("frames_with_lt_4_bimfs", "errors")}, indent=2))
    if summary.get("n_bimf_lt_4"):
        print(
            f"WARNING: {summary['n_bimf_lt_4']} frame(s) produced fewer than 4 BIMFs. "
            f"See {args.cache_root}/preprocess_summary.json",
            file=sys.stderr,
        )
    if summary.get("n_errors"):
        print(f"ERROR: {summary['n_errors']} frame(s) failed. See preprocess_summary.json", file=sys.stderr)
        return 2
    if summary.get("n_bad_recon"):
        print(f"ERROR: {summary['n_bad_recon']} frame(s) failed reconstruction gate.", file=sys.stderr)
        return 3
    print(f"Audit CSV: {audit_csv}")
    print(f"Summary JSON: {Path(args.cache_root) / 'preprocess_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
