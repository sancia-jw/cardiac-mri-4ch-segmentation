"""Paths and label definitions for CINE_MULTI/4CH_TR."""

from pathlib import Path
from typing import Dict, Tuple

# Repository layout: segmentation/CMR-MULTI/CINE_MULTI/4CH_TR/{image,anno}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "CMR-MULTI" / "CINE_MULTI"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VIEW = "4CH_TR"

NUM_CLASSES = 6  # background + 5 foreground structures

# Foreground label IDs expected in a complete 4CH annotation.
EXPECTED_FOREGROUND_LABELS = (1, 2, 3, 4, 5)

LABEL_NAMES: Dict[int, str] = {
    0: "background",
    1: "Left Ventricle Cavity",
    2: "Left Ventricle Myocardium",
    3: "Right Ventricle Cavity",
    4: "Right Atrium",
    5: "Left Atrium",
}

LABEL_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 102, 255),
    4: (255, 215, 0),
    5: (255, 0, 255),
}

NII_EXTENSIONS = (".nii.gz", ".nii")

LFS_HINT = (
    "No paired NIfTI files found. If you cloned CMR-MULTI from HuggingFace, "
    "request access and run from the CMR-MULTI directory:\n"
    "  git lfs install\n"
    "  git lfs pull"
)

# Default spatial size for the 2D UNet baseline (H, W vary slightly across cases).
DEFAULT_IMAGE_SIZE = (160, 160)
