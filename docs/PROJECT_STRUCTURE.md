# Project structure

| Folder | Purpose |
|---|---|
| `flask-application/` | Web routes, inference adapters, templates, styles and classifier weights |
| `tree-counting-model/` | Tree inference/training implementation, evaluated weights and benchmark datasets |
| `evaluations/notebooks/` | Current evaluation entry points and detailed results |
| `evaluations/classifier_hd/`, `tree_counter_hd/` | Shared evaluators and saved evidence |
| `evaluations/new_benchmark/` | Independent label review and frozen benchmark workflow |
| `evaluations/archive/` | Historical evaluation evidence |
| `notebooks/training/` | Older research/training notebooks and training script |
| `notebooks/test_data/` | Retained tiler fixtures referenced by existing notebooks |
| `tests/` | Automated software and metric tests |
| `data/legacy/` | Local unused raster inputs, preserved together |
| `docs/` | Architecture and audit records |
| `runs/` | Local training history required by historical evaluation |

The root `yolo11n.pt` is the local training base checkpoint, not a deployed model.
Keep it and training history for reproducibility. Uploads and label-review records
are local runtime data; neither is part of the source-code cleanup.

## Runtime flow

Upload -> image validation -> unique stored image -> binary detector -> category
classifier when positive. Independently: tree tiling -> YOLO -> duplicate
suppression -> crown overlay. Results and partial-failure notices render together.
Application startup loads saved models only; training is an explicit offline task.
