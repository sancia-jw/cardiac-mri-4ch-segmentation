# Training-recipe experiments (improve_kaggle.ipynb)

The EMD/BEMD/multiscale ablations all left the U-Net where it was (~0.877 test
foreground Dice). This experiment changes how the network is trained instead of
what it sees.

## What was limiting the old baseline

- 15 epochs at a constant learning rate, batch 4, best-validation checkpoint.
- Only left-right flips as augmentation. Flipping a 4-chamber view mirrors the
  anatomy (left heart on the wrong side), so it is not a realistic variation.
- Every frame resized to 160x160 (native 169x155, so a slight squash) and
  min-max scaled per frame, which depends on the brightest pixel.
- A 3-level U-Net (1.9 M parameters).
- 4CH_VAL (cases 106-120, 15 annotated cases) was never used.
- Each 4CH volume is 3 slice positions x 25-30 cine phases, but every frame
  was segmented on its own.

## Conditions (configs/improve.yaml)

| condition | change |
|---|---|
| legacy | the old recipe, re-trained so every row is scored the same way |
| recipe | native grid (zero-padded to 192x160), per-case 0.5-99.5 percentile clip + z-score, rotation +-25 deg / scale 0.8-1.25 / shift / light elastic, brightness, contrast, gamma, blur, low-res and noise (all on the GPU), no mirroring; 6-level residual U-Net (16.6 M); AdamW 1e-3 + 3-epoch warm-up + cosine; batch 16; 80 x 250 steps; final weights |
| recipe_t5 | + phases t-2..t+2 of the same slice as 5 input channels (wrapping round the cine loop) |
| t5_extra | + the 15 4CH_VAL cases in training |
| t5_full | + the 16 validation cases in training (no model selection uses them) |
| smp_r34_t3 | recipe with an ImageNet-pretrained ResNet-34 U-Net, phases t-1..t+1 as RGB |

Each condition runs at seeds 42, 43, 44; the three models of a condition are
also scored as an ensemble (mean softmax).

## Metric

Logits are mapped back to the native grid (cropped from the canvas, or
upsampled for `legacy`). Dice for each foreground class is pooled over all
frames of a test case, then averaged over the 15 test cases; `fg` is the mean
of the five classes. `fg_lcc` keeps only the largest connected blob of each
class per frame. `legacy_fg` is the old style (Dice per batch of 4 consecutive
frames, averaged), for comparison with earlier numbers.

## Running

Attach `cmr-multi-cine-4ch` (4CH_TR) and `cmr-multi-cine-4ch-val` (4CH_VAL),
enable GPU and Internet, set COMMIT, and run all cells. Results go to
`/kaggle/working/improve/` (summary.csv, seed_table.csv, ensemble_table.csv,
one folder per condition and seed with train_log.csv, test_per_case.csv and
final.pt). Runs that are already finished are skipped on a re-run.

Local smoke test (CPU, tiny):
`python scripts/train_improve.py --data-root CMR-MULTI/CINE_MULTI --output-root /tmp/x --runs recipe --seeds 1 --max-cases 2 --validate-only`
