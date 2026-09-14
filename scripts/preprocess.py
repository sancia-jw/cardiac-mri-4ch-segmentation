#!/usr/bin/env python3
"""Portable preprocess entry point → BEMD square-pad cache (see preprocess_bemd.py)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "preprocess_bemd.py"
    # Preserve argv so flags like --config configs/bemd_ablation.yaml work.
    runpy.run_path(str(target), run_name="__main__")
