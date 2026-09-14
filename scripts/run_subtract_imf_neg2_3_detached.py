#!/usr/bin/env python3
"""
Detached runner for subtract_imf_neg2_3: keeps resuming until 15 epochs + test metrics exist.

Logs to outputs/emd_ablation/subtract_imf_neg2_3/detached_runner.log
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "outputs" / "emd_ablation" / "subtract_imf_neg2_3"
LOG_CSV = RUN_DIR / "training_log.csv"
TEST_CSV = RUN_DIR / "test_metrics.csv"
RUNNER_LOG = RUN_DIR / "detached_runner.log"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
TARGET_EPOCHS = 15
MAX_ATTEMPTS = 40
SLEEP_BETWEEN_ATTEMPTS_SEC = 30


def log(msg: str) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(RUNNER_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def epochs_done() -> int:
    if not LOG_CSV.exists():
        return 0
    n = 0
    with open(LOG_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("epoch"):
                n = max(n, int(row["epoch"]))
    return n


def is_complete() -> bool:
    return epochs_done() >= TARGET_EPOCHS and TEST_CSV.exists()


def main() -> int:
    if not PYTHON.exists():
        log(f"ERROR: venv python not found at {PYTHON}")
        return 1

    log(f"Detached runner started. cwd={ROOT}")
    log(f"Current epochs={epochs_done()} test_metrics={TEST_CSV.exists()}")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if is_complete():
            log("COMPLETE: 15 epochs + test_metrics present. Exiting.")
            return 0

        done = epochs_done()
        log(
            f"Attempt {attempt}/{MAX_ATTEMPTS}: resume from epoch {done + 1} "
            f"(have {done}/{TARGET_EPOCHS}, test={TEST_CSV.exists()})"
        )

        cmd = [
            str(PYTHON),
            str(ROOT / "scripts" / "run_emd_ablation.py"),
            "--runs",
            "subtract_imf_neg2_3",
            "--resume",
            "--epochs",
            str(TARGET_EPOCHS),
            "--batch-size",
            "4",
            "--lr",
            "0.001",
            "--seed",
            "42",
            "--device",
            "cpu",
            "--preprocess-workers",
            "4",
            "--num-overlays",
            "10",
        ]

        with open(RUNNER_LOG, "a", encoding="utf-8") as out:
            out.write(f"\n----- attempt {attempt} command: {' '.join(cmd)}\n")
            out.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=out,
                stderr=subprocess.STDOUT,
            )

        log(f"Attempt {attempt} exited with code {proc.returncode}; epochs={epochs_done()}")

        if is_complete():
            log("COMPLETE after successful attempt.")
            return 0

        log(f"Not complete yet. Sleeping {SLEEP_BETWEEN_ATTEMPTS_SEC}s before resume...")
        time.sleep(SLEEP_BETWEEN_ATTEMPTS_SEC)

    log(f"ERROR: gave up after {MAX_ATTEMPTS} attempts. epochs={epochs_done()}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
