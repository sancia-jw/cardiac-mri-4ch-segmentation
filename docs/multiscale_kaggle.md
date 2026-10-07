# Multiscale removal ablation (follow-up to the BEMD primary run)

## Why

The BEMD primary run (`notebooks/bemd_kaggle_primary.ipynb`, 2026-10-06) found
no condition meaningfully different from `original`: test foreground Dice
ranged from -0.0075 to +0.0006 against the baseline, and validation and test
ranked the conditions differently. Two problems explain this.

1. **The subtraction removed effectively nothing.** The Gastro-style convention
   min-max scales each BIMF to [0, 1] and subtracts it from the *raw* MRI, whose
   intensity range is about 700-5,500. Measured on 40 cached frames, the final
   model input moved by a mean of 0.00004-0.00009 and a maximum of 0.0013 on a
   [0, 1] scale, which is less than one 8-bit grey level. The five conditions
   trained on near-identical images. The legacy 1D EMD path
   (`src/preprocessing/emd_enhancement.py`) uses the same convention.
2. **PyEMD BEMD components aren't usable scale layers.** Its envelopes are
   global cubic RBF surfaces through every extremum, and these overshoot. On
   cached frames, BIMF 1 holds about 11x the image energy, BIMF 2 about 33x,
   and BIMF 3 up to 8x10^5x. The components cancel each other out instead of
   forming bands, and the three excluded frames are the extreme case. More
   sifting makes this worse: with `FIXE=5, max_imf=8`, two test frames stopped
   at 2 BIMFs, with BIMF 0 at 260x the image energy and the residual up to
   10^9x. Removing the existing BIMFs at their true amplitude changes the input
   by 0.31-0.39, mostly from the cancelling terms. Rebuilding the cache with
   PyEMD therefore wasn't pursued.

## What runs instead

`src/preprocessing/multiscale.py` provides two exact, bounded decompositions,
computed on the fly from the raw MRI in 13-33 ms per frame, with no cache:

| Condition family | Method | Components used |
| --- | --- | --- |
| `subtract_gband_0..4` | Difference of Gaussians, sigma 1, 2, 4, 8, 16 px | 5 octave bands |
| `subtract_fabemd_0..3` | FABEMD (Bhuiyan et al., 2008): max/min-filter envelopes plus mean filter | BIMFs with median windows 3, 7, 17, 33 px |

Across all 8,285 frames, reconstruction RMSE is at most 1e-17. FABEMD BIMFs 0-3
exist for every frame (the window-doubling fallback is used for BIMF 3 on only
62 frames). BIMF 4 is left out because about 36% of frames reach it only
through the fallback and 26 frames are too small for it.

Removal is amplitude-faithful: `minmax(unit - sum(components))`, where `unit`
is the min-max normalized frame. Individual components carry 1-5% of the image
energy, and removing one changes the input by a mean of 0.03-0.09.

Controls are unchanged: the 74/16/15 split, seed 42, 15 epochs, batch size 4,
lr 1e-3, `UNet2D`, the loss, the metrics, and the same three exclusions
(`configs/bemd_exclusions.json`), giving 8,282 frames.

## Running it

Attach only the raw MRI dataset to `notebooks/multiscale_kaggle.ipynb`, set
`COMMIT` to the pushed commit, and run the cells in order. The config is
`configs/multiscale_ablation.yaml`; on the command line, use
`--catalog multiscale`. The derived-image cache is written to `/tmp`, so it
isn't saved with the notebook output.

Read-only local check (no training):

```powershell
.\.venv\Scripts\python.exe scripts/run_bemd_ablation.py --config configs/multiscale_ablation.yaml --data-root CMR-MULTI/CINE_MULTI --output-root outputs/multiscale_preflight_unused --validate-only
.\.venv\Scripts\python.exe -m unittest tests.test_bemd_portability tests.test_multiscale
```

With a single seed, differences below about 0.005 Dice are within run-to-run
noise. A positive result should be repeated across seeds before it's reported.
