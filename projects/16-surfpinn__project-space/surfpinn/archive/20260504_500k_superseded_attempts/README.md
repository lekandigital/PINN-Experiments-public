# Archived Superseded Prompt 5 Attempts

Generated: 2026-05-04

These checkpoints are retained for diagnosis only and must not be used as the Project 16 learned-demo checkpoint.

- `surfpinn_best_torch_family_a_smoke.pt`: early local/remote smoke checkpoint from the first high-accuracy script validation path.
- `surfpinn_best_torch_family_a_500k_configured_time_superseded.pt`: superseded 500k launch where the target held-out interval setup was too strict globally before the final target-only time-holdout configuration.

The active learned checkpoint is:

`../../checkpoints/surfpinn_best_torch_family_a_500k_target_timeheldout.pt`

Use the active checkpoint with:

- `../../outputs/reference/`
- `../../outputs/validation_model/`
- `../../../private/project16_surfpinn_gate_report.json`
- `../../../private/project16_surfpinn_rebuild_notes.md`
