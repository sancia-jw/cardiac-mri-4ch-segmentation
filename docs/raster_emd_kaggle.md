# Gastro decomposition (raster EMD) ablation

## What it is

The decomposition in [mou58/Gastro `utils/dataloader.py`](https://github.com/mou58/Gastro/tree/master/utils)
(`emd2d`, `imfs2im`) is not a 2D EMD. It flattens the image row by row into
one long 1D signal, runs `emd.sift.sift(sift_thresh=1e-8)`, and reshapes each
IMF back to the image. IMF 0 is the finest variation along the rows; the last
column is the slow trend. Gastro's main setting subtracts IMFs `[-1, -2, -3]`.

The repo already had this method (`src/preprocessing/emd_enhancement.py`,
trained in `outputs/emd_ablation`), but with Gastro's scaling: each IMF is
min-max scaled to [0, 1] and subtracted from the raw MRI (range ~700-5,500).
On real frames that moves the model input by about 0.0001, so those runs
trained on near-identical images, as with the BEMD primary run.

## What runs

`raster_emd` in `src/preprocessing/multiscale.py` sifts the unit-normalized
frame, so IMFs are in image units, and removes them at their real size, like
the Gaussian and FABEMD conditions. On real frames there are 9-10 IMFs, the
reconstruction is exact (error ~1e-17), and removal changes the input by
0.05-0.25.

| Condition | Removed |
| --- | --- |
| `original` | nothing |
| `subtract_remd_0` | IMF 0 (finest) |
| `subtract_remd_1` | IMF 1 |
| `subtract_remd_0_1` | IMFs 0 and 1 |
| `subtract_remd_trend` | IMFs -1, -2, -3 (Gastro's setting) |

Each condition runs at seeds 42, 43 and 44. Controls match the multiscale run.
The decomposition takes ~0.2 s per frame, so it is computed on the fly in
parallel across cases (`ONTHEFLY_WORKERS`, default 4) and cached once in
`/tmp` for all seeds.

## Running it

Import `notebooks/raster_emd_kaggle.ipynb`, attach only the raw MRI dataset,
set `COMMIT` to the pushed commit, and run the cells in order. Config:
`configs/raster_emd_ablation.yaml` (`--catalog raster_emd`).

Local checks:

```powershell
.\.venv\Scripts\python.exe scripts/run_bemd_ablation.py --config configs/raster_emd_ablation.yaml --data-root CMR-MULTI/CINE_MULTI --output-root outputs/raster_emd_preflight_unused --validate-only
.\.venv\Scripts\python.exe -m unittest tests.test_bemd_portability tests.test_multiscale tests.test_raster_emd
```
