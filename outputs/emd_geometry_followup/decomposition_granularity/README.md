# Decomposition granularity study

## Answer
**Use emd_default_square_pad (PyEMD BEMD + square zero-pad + crop-back).**

EMD2D (default or tuned) cannot separate multiple spatial scales on these images:
after BIMF0 the residue has **0 extrema**, so the algorithm always stops at 1 BIMF + residual.

BEMD produces **4 BIMFs** on synthetic and on square-padded MRI (8/8 probes), with excellent
reconstruction. Spectral centroids usually decrease but are not perfectly monotonic.

## Why EMD2D collapses
See AUDIT_EMD2D.md and extrema_stop_diagnosis.json.

## BEMD MRI workaround
Native non-square frames crash (IndexError). Zero-pad to max(H,W) square, decompose, crop.

Evidence: emd_square_pad_mri.json.

## Next step
If doing a BIMF-k segmentation screen, implement this BEMD square-pad backend.
Do **not** expect EMD2D tuning to unlock BIMF1+.

No U-Net training was run in this stage.
