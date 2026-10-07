# BEMD cache transfer and primary Kaggle ablation

No local training or model evaluation was run during preparation.

## Cache validation and exclusions

The local run attempted all 8,285 frames from 105 cases. Of these, 8,282
pass the unchanged reconstruction gate (native-intensity RMSE <= 0.001).
The three failing frames were recomputed individually with their original
PyEMD 1.10.0 settings. Every resulting array was identical to its original
cached counterpart; their failures are reproducible, not incomplete/corrupt
cache entries. Arrays are finite float64, shapes are correct, saved originals
match raw MRI exactly, and loaded reconstruction errors match the metadata.

| Case | Zero-based frame | Original and recomputed RMSE | Largest component magnitude |
| --- | ---: | ---: | ---: |
| CINE_4CH_023 | 16 | 0.046689965481377965 | 4.02e15 |
| CINE_4CH_051 | 65 | 0.136424190191389 | 9.63e15 |
| CINE_4CH_085 | 65 | 0.21324565081617805 | 1.54e16 |

Large opposing BIMF3/residual values cause float64 cancellation error. The
solver, normalization, residual definition, and reconstruction gate were not
changed. Original entries remain intact; only these three frames were
recomputed into `outputs/bemd_cache/run_logs/repair_20260917/recomputed/`.
They are explicitly excluded by `configs/bemd_exclusions.json`, also provided
as `exclusions.json` inside the upload directory. The same exclusions apply
to every primary condition, including original, across train/validation/test.
No missing component is invented or substituted.

| Component | Available among usable frames | Unavailable |
| --- | ---: | ---: |
| BIMF 0 | 8,282 | 0 |
| BIMF 1 | 8,282 | 0 |
| BIMF 2 | 8,282 | 0 |
| BIMF 3 | 7,385 | 897 |

The usable-frame distribution is 897 with 3 BIMFs and 7,385 with 4 BIMFs.
All-frame distribution (including exclusions) is 897 with 3 and 7,388 with 4.
The availability audit checks expected raw-volume frame counts, metadata,
array headers/shapes/dtypes, and file sizes. Reconstruction statistics reuse
the existing per-frame numerical validation; only the three failures were
numerically rechecked/recomputed. Soft warnings remain usable under the
existing gate and are counted separately in the manifest.

## Private dataset upload

Upload the **contents** of this existing directory as a private Kaggle Dataset:

```text
C:\SanciaMJW\projects\segmentation\outputs\bemd_cache\bemd_default_square_pad\
```

Size is approximately 10.26 GB (9.55 GiB). No second full copy or archive is
needed. Include the per-case folders, `exclusions.json`, `cache_manifest.json`,
and the existing `preprocess_summary.json` / `preprocess_audit.csv`.
Keep the excluded frame folders for provenance; the explicit exclusions prevent
their use. The historical summary/audit describe the original full run and
remain unchanged. The manifest supplements them with the transfer audit and
references their hashes rather than duplicating all per-frame records.

Suggested dataset slug: `cmr-multi-bemd-square-pad`. Uploading the contents
directly gives this layout:

```text
/kaggle/input/cmr-multi-bemd-square-pad/
    cache_manifest.json
    exclusions.json
    CINE_4CH_001/frame_000/{original.npy,residual.npy,bimf_0.npy,...,metadata.json}
    ...
```

If Kaggle preserves a `bemd_default_square_pad/` wrapper folder, append it to
`CACHE` below. `CACHE` must directly contain `CINE_4CH_001/` and `exclusions.json`.
The actual dataset slug depends on the name assigned during upload.
MRI stays in a separate private input dataset; `DATA` must contain `4CH_TR/`.
Cache/MRI arrays, checkpoints, and generated caches remain ignored by Git.

## Primary experiment and fixed controls

`configs/bemd_ablation.yaml` selects exactly:

1. `original`
2. `subtract_bimf_0`
3. `subtract_bimf_1`
4. `subtract_bimf_2`
5. `subtract_bimf_0_1`

Preserve the repository's `outputs/splits_4ch.csv`; do not regenerate it.
Case membership remains 74/16/15, with 5,883/1,245/1,154 usable frames after
exclusions. The loader resolves cases against `--data-root`, ignoring the
historical Windows paths in the CSV. Fixed controls remain seed 42, 15 epochs
in the config (`notebooks/bemd_kaggle_primary_50ep.ipynb` passes `--epochs 50` via its `EPOCHS` setting),
batch size 4, LR 1e-3, the same one-channel UNet2D at 160x160, per-slice
normalization, horizontal-flip augmentation, Adam, combined loss, Dice metrics,
and best validation foreground-Dice checkpoint selection. No model code,
optimizer/loss/metrics, or decomposition settings were changed.

## Kaggle commands (run later, not locally)

Clone the prepared repository into `/kaggle/working/cardiac-mri-4ch-segmentation`, install its
`requirements.txt` if needed, and attach both private datasets. From the repo
root, use a Kaggle notebook `%%bash` cell or a shell. Set the paths to the
actual mounted dataset directories:

```bash
DATA=/kaggle/input/cmr-multi-4ch/CINE_MULTI
CACHE=/kaggle/input/cmr-multi-bemd-square-pad
OUT=/kaggle/working/bemd_ablation_primary

# Safe preflight: reads raw headers/cache; no model, training or evaluation.
python scripts/run_bemd_ablation.py \
  --config configs/bemd_ablation.yaml \
  --data-root "$DATA" --cache-root "$CACHE" \
  --exclusions "$CACHE/exclusions.json" \
  --splits-csv outputs/splits_4ch.csv \
  --output-root "$OUT" --validate-only
```

Only later, when ready to train on Kaggle with a GPU, run:

```bash
python scripts/run_bemd_ablation.py \
  --config configs/bemd_ablation.yaml \
  --data-root "$DATA" --cache-root "$CACHE" \
  --exclusions "$CACHE/exclusions.json" \
  --splits-csv outputs/splits_4ch.csv \
  --output-root "$OUT" --device cuda
```

This command executes the five selected conditions and their existing test
evaluation. Source BEMD arrays are read only. Derived image caches default to
`$OUT/bemd_enhanced_cache`; checkpoints, metrics, predictions, and provenance
also stay under `$OUT`. Optionally set `--enhanced-cache-root` explicitly.
These path options and `exclusions` are also accepted in YAML's `train:`
section; explicit CLI flags take precedence. There is no need to edit source.
Use a fresh output directory for a new input cache to avoid stale derived images.
`--skip-train` is not a validation mode: it evaluates existing checkpoints.
Use `--validate-only` for checks without model execution.

## Secondary BIMF3 comparison

`subtract_bimf_3` remains supported in the catalog and its individual YAML.
It is deliberately absent from the primary five-run config. A future comparison
must define the common valid subset with BIMF3 and train/evaluate BOTH original
and subtract_bimf_3 with exactly the same subset rules. At frame level, the
current eligible pool has 7,385 frames; a whole-case rule would require a
separate case-level eligibility audit. Retain original split assignments and
report per-split sample counts. Do not compare a subset BIMF3 run against the
full-frame original baseline. No secondary experiment was run or configured
to silently drop the 897 missing-component frames.

## Verification performed

- Syntax compilation and imports of changed modules and CLI help.
- Three targeted recomputations only; no existing decomposition overwritten.
- Full read-only component availability/header/size audit.
- Synthetic dataset tests for all five transforms, consistent exclusions,
  external read-only inputs, writable derived caches, missing BIMF rejection,
  YAML/CLI precedence, and validate-only guards against train/evaluation calls.
- No U-Net instantiation, optimizer step, model evaluation, or training run.
