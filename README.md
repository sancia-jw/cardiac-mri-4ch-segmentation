# CMR-MULTI CINE 4CH Segmentation Baseline

Baseline pipeline for **CINE_MULTI/4CH_TR** cardiac MRI segmentation using the [CMR-MULTI](https://huggingface.co/datasets/TaipingQu/CMR-MULTI) dataset.

## Label map (4CH_TR)

| ID | Structure |
|----|-----------|
| 0 | background |
| 1 | Left Ventricle Cavity |
| 2 | Left Ventricle Myocardium |
| 3 | Right Ventricle Cavity |
| 4 | Right Atrium |
| 5 | Left Atrium |

## Setup

```bash
cd c:\SanciaMJW\projects\segmentation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Ensure NIfTI data is present under `CMR-MULTI/CINE_MULTI/4CH_TR/{image,anno}/`. If the clone only has metadata, run `git lfs pull` inside `CMR-MULTI/`.

## Run scripts in order

### 1. Dataset audit

Scans all paired cases, checks shapes/labels, and writes a CSV summary.

```bash
python scripts/audit_dataset_4ch.py
```

Output: `outputs/dataset_audit_4ch.csv`

### 2. Visual quality checks

Randomly saves 20 case figures (image, mask, overlay). Rerun with the same `--seed` for reproducibility.

```bash
python scripts/visualize_cases_4ch.py
python scripts/visualize_cases_4ch.py --seed 42 --num-cases 20
```

Output: `outputs/visual_checks/*.png`

### 3. Train / val / test splits

Case-level split (70% / 15% / 15%). Each `CINE_4CH_XXX` file is one case; no case appears in more than one split.

```bash
python scripts/create_splits_4ch.py
```

Output: `outputs/splits_4ch.csv`

### 4. Baseline UNet training

Trains a simple 2D multiclass UNet on all temporal frames from the training split. Uses CrossEntropy + Dice loss and logs per-class validation Dice.

```bash
python scripts/train_unet_4ch.py
python scripts/train_unet_4ch.py --epochs 30 --batch-size 8 --device cuda
```

Training on CPU is supported but slow (~5k slices/epoch). A GPU is recommended for full runs.

Outputs:
- `outputs/checkpoints/unet_4ch_best.pt`
- `outputs/checkpoints/unet_4ch_epoch_XXX.pt`
- `outputs/training_log.csv`

## Project layout

```text
segmentation/
  CMR-MULTI/CINE_MULTI/4CH_TR/   # dataset (image/, anno/)
  cine_4ch/                      # shared config, io, model, dataset
  configs/                       # BEMD ablation YAML configs
  scripts/                       # audit, visualize, splits, train, BEMD ablation
  outputs/                       # generated artifacts
  explore_cmr_multi_4ch.py       # single-case explorer
```

## BEMD BIMF ablation (next experiment)

True-2D backend after the EMD2D audit: `bemd_default_square_pad`. See `outputs/bemd_ablation/README.md`.

For the validated five-condition primary experiment, cache exclusions, and
private Kaggle dataset transfer, see [the Kaggle handoff](docs/bemd_kaggle.md).
Use `--validate-only` to check the cache without training or evaluation.

```bash
python scripts/run_bemd_ablation.py --config configs/bemd_ablation.yaml --exclusions configs/bemd_exclusions.json --validate-only
```

Do not overwrite `outputs/emd_ablation/` (historical 1D Holman results).

## Notes

- CINE volumes are 3D `(H, W, time)`; training treats each temporal frame as a 2D slice.
- Spatial size is resized to `160×160` for the baseline UNet.
- Images are normalized per-slice to `[0, 1]` before training.
