# Step 1 — Audit of current `PyEMD.EMD2D`

## Code path
- Package: `EMD-signal` / `PyEMD` experimental `EMD2d.EMD2D`
- Normalize image to `[0,1]`, sift IMFs, append residual if nonzero, rescale

## Extrema
- 3x3 local max/min (`scipy.ndimage.maximum_filter`)
- Outer loop continues only if `n_min > 4` AND `n_max > 4`

## Envelopes
- Mirror-pad 3x3, `SmoothBivariateSpline` on extrema

## Defaults
| Param | Default |
|---|---:|
| mean_thr | 0.01 |
| mse_thr | 0.01 |
| FIXE | 0 |
| FIXE_H | 0 |
| MAX_ITERATION | 1000 |

## Why MRI yields 1 BIMF
On mentor frame after BIMF0, residue extrema count is **0 / 0**.
Outer loop stops; leftover becomes residual (~40-50% energy, smooth trend).

Root cause: **permissive proto-IMF stop + extrema detector** → broad first BIMF →
featureless residual. Not a hard max_imf=1 limit.
