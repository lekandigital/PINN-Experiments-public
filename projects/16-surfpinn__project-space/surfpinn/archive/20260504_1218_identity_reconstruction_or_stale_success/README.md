# Archived Identity-Reconstruction / Stale Success Context

Generated: 2026-05-04

This archive records why the old SurfPINN artifacts must not be used as learned-demo evidence.

## Local Provenance

- Prompt 1 found no local `.pkl`, `.pt`, `.pth`, `.ckpt`, `.npz`, `.npy`, `.h5`, or `.hdf5` files under `/Users/lekan/Dev/PINN-Experiments/projects/16-surfpinn__project-space` before remote recovery.
- Local git history and stash search did not contain tracked Project 16 checkpoint or HDF5/NumPy data artifacts.
- Local logs referenced checkpoint paths such as `checkpoints/surfpinn_best.pkl`, but those files were missing locally before recovery.

## Remote Recovery

Recovered diagnostic artifacts were copied from `/home/o/surfpinn/src` on the 3090 server to:

`/Users/lekan/Dev/PINN-Experiments/projects/16-surfpinn__project-space/surfpinn/recovered_remote_2026-05-04/`

The recovered `surfpinn_best.pkl` is a real checkpoint:

- top keys: `params`, `state`, `step`, `epoch`
- epoch: `156`
- step: `1248`
- parameter leaves: `34`
- trainable parameter count: `230372`
- file size: about `0.93 MB`

## Rejected Contract

The recovered checkpoint was trained with the target height in the model input:

- `grid_input = concat(grid_coords[:, :, t, :], height[:, :, t, :])`
- `height_true = height[:, :, t, :]`

That makes the old Eulerian task an `h_t -> h_t` reconstruction contract, not a prediction or simulation contract.

## Metrics Status

Old metrics remain diagnostic only:

- best reported PSNR: `15.81 dB`
- 500-epoch PSNR: `14.62 dB`
- 1000-epoch PSNR: `12.24 dB`

They are below the `30 dB` learned field-surrogate threshold and were produced under the rejected target-in-input contract.

## Results PNG Status

Old `results/height_comparison.png` and `results/training_loss.png` were retained only as diagnostics. They do not prove a loadable valid checkpoint, a held-out split, or forward-pass provenance under a non-tautological contract.

## Copied Prompt 1 Evidence

- `prompt1_reference_metadata.json`
- `prompt1_validation_provenance.json`
- `prompt1_reference_metrics.json`
