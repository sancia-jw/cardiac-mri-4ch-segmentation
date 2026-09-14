# BEMD BIMF-removal ablation

Scientific question: which spatial modes (BIMFs) from `bemd_default_square_pad`
are useful or harmful for 4CH cardiac MRI segmentation?

## Backend

After the EMD2D audit (`outputs/emd_geometry_followup/decomposition_granularity/`),
PyEMD `EMD2D` is **not** used for multiscale BIMF ablation (only BIMF0+residual).

Decomposition protocol: **`bemd_default_square_pad`**

1. Zero-pad native (H,W) to square `max(H,W)` (bottom/right)
2. Run default-ish PyEMD `BEMD` (`FIXE=1`, `mean_thr=0.01`, `mse_thr=0.01`, `max_imf=4`)
3. Crop each BIMF + residual back to native FOV

## Reconstruction (matches historical Gastro-style EMD ablation)

- `normalize_bimfs=True`: each selected BIMF is min-max normalized before subtraction
- `subtract_bimf_k = finalize(original - minmax(BIMF[k]))`
- `subtract_bimf_0_1 = finalize(original - minmax(BIMF[0]) - minmax(BIMF[1]))`
- `finalize` = per-slice min-max to `[0,1]` with clipping

Do **not** interpret BIMFs as perfect Fourier bands. BIMF 0 is typically the
finest / highest-spatial-frequency mode; later BIMFs are generally coarser.
Spectral centroids usually decrease but are not guaranteed monotonic.

## Conditions

| run_id | meaning |
|--------|---------|
| original | normalized MRI only |
| subtract_bimf_0 | remove BIMF 0 |
| subtract_bimf_1 | remove BIMF 1 |
| subtract_bimf_2 | remove BIMF 2 |
| subtract_bimf_3 | remove BIMF 3 |
| subtract_bimf_0_1 | remove BIMF 0 and 1 |

Controls held fixed vs the prior full EMD ablation: split 74/16/15, seed 42,
U-Net, 160×160, aug, Adam 1e-3, batch 4, 15 epochs, combined loss, FG Dice
checkpoint selection.

## Cache layout

```
outputs/bemd_cache/bemd_default_square_pad/<case_stem>/frame_XXX/
  bimf_0.npy ... bimf_K.npy
  residual.npy
  original.npy
  metadata.json
```

Valid entries are never auto-deleted. Resume skips valid cache. Use `--force` to recompute.

## Commands (Kaggle / full)

See `README.md` project root or the final report from the implementation chat.
Historical 1D Holman results under `outputs/emd_ablation/` are left untouched.
