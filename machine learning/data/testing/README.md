# Testing data — start here

| Folder | Purpose |
|---|---|
| `tree_counter/` | Original nine-image labelled tree test split; known site/marker limitations |
| `neon_benchmark/` | External labelled NEON images and derived stress variants |
| `synthetic_trees/` | Artificial image/label fixtures for checking software, not real-world accuracy |
| `tiler_fixtures/` | Small example images and expected tiles for testing image splitting |
| `tree_challenges/` | Composite and negative images for robustness diagnostics |
| `marker_cleanup/` | Inpainted validation images for the historical marker-sensitivity experiment |

Keep original benchmark bundles intact: some contain historical manifests,
previews and saved results. Synthetic split names (`train`, `valid`, `test`)
are fixture conventions, not new independent field samples. Canonical evaluation
reports are under `../../evaluations/`.
