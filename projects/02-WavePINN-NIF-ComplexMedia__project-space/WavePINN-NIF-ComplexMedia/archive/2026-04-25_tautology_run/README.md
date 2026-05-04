# Project 02 Tautology Run Archive

This archive preserves the previous Prompt 1/2/3 outputs so the failure is auditable instead of deleted.

Archived contents include:

- `scripts/project02_contract_repair.py`
- copied-array checkpoints under `checkpoints/`
- reference and validation arrays under `outputs/`
- Taichi artifacts and metadata under `artifacts/`
- the original zero-solution `results/metrics.json`
- private notes and post draft under `private/`

Why this run is invalid:

- `outputs/project02_validation_model/wavefield_sequence.npy` is a bit-identical copy of `outputs/project02_reference/reference_wavefield.npy`.
- `outputs/project02_validation_model/medium_pred.npy` is a bit-identical copy of `outputs/project02_reference/reference_medium.npy`.
- `checkpoints/project02_wavepinn_nif_best.pkl` and `project02_wavepinn_nif_final.pkl` store copied arrays under `params.wavefield_grid` and `params.medium_grid`; they are not neural-network weights.
- The metrics in `artifacts/taichi_final/metadata.json` and `outputs/project02_validation_model/metrics.json` compare the copied validation arrays to their source arrays, producing tautological values such as `NMSE = 0.0`, capped `PSNR = 120.0 dB`, and `medium correlation = 1.0`.
- The identity copy happens in `scripts/project02_contract_repair.py` lines 370-407, specifically:
  - line 378: `wave_params = reference_wavefield.astype(np.float32).copy()`
  - line 379: `medium_params = reference_medium.astype(np.float32).copy()`
  - line 407: the copied arrays are returned as the "fit"

This archive must not be used as a valid checkpoint, validation result, or gate-passing artifact.
