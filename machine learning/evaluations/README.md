> **Current layout:** Mourilyan photos are external manual-demo inputs. Any 527-photo inventory or diagnostic results below are historical snapshots. See [current data availability](CURRENT_DATA_LAYOUT.md).

# Model Evaluation Guide

Start with [00_evaluation_overview.ipynb](notebooks/00_evaluation_overview.ipynb).
It reads saved evidence through shared code, separates historical and deployed
configurations, and links to the detailed notebooks. It does not rerun inference.

## Current evidence

The binary detector, category classifier and tree counter are integrated features.
Independent classifier accuracy and full HD compliance remain unestablished.
Tree results have dataset limitations documented in the overview and detailed
notebook. Software checks confirm application behaviour, not field accuracy.

- [Binary evaluation](notebooks/Binary_Mangrove_Evaluation.ipynb): source/checkpoint
  audits, diagnostics, application checks and missing-label requirements.
- [Category evaluation](notebooks/Mangrove_Species_Evaluation.ipynb): the same
  shared evaluation framework, including the unresolved class-mapping provenance.
- [Tree evaluation](notebooks/YOLO_Tree_Counter_Evaluation.ipynb): application-path
  detection/counting metrics, validation experiments, robustness and rubric audit.
  The candidate confidence 0.10 is now deployed; see [deployment checks](tree_counter_hd/DEPLOYMENT.md).
- [Combined pipeline](notebooks/evaluate_combined_pipeline.ipynb): readiness only;
  no independently scored end-to-end dataset.
- [Tree dataset](notebooks/evaluate_tree_counter_dataset.ipynb) and
  [image tiler](notebooks/evaluate_image_tiler.ipynb): dataset and preprocessing checks.

Principal results are collected in the overview; detailed results remain in their
component notebooks. The overview reads saved JSON artifacts rather than copied
metric constants. Each execution includes source hashes. Read it as a snapshot,
not a live health check or a fresh model evaluation.

## Historical evidence

- [Original binary training notebook](archive/evaluate_binary_detector_historical.ipynb)
  is preserved with presentation-only heading alignment. Its unavailable Windows data paths and inconsistent
  saved test counts cannot be repaired without the original data. It is not a
  supported current evaluation runner. The [old entry point](notebooks/evaluate_binary_detector.ipynb)
  now safely loads the shared audit and links to the current evaluation.
- [Direct-YOLO notebook](notebooks/evaluate_tree_counter_model.ipynb) preserves
  historical results from different inference settings. Do not substitute these
  for the deployed application-path results in the overview.
- [Legacy category readiness notebook](notebooks/evaluate_multiclass_classifier.ipynb)
  remains available; use the detailed category notebook for current evidence.
- [Recovery audit](classifier_hd/DATA_RECOVERY.md) explains why the old training
  data and reported results cannot be independently reconstructed.

## What remains outside notebooks

Shared evaluation/metric code and saved predictions remain in `classifier_hd/`
and `tree_counter_hd/`. [External benchmark preparation](../tree-counting-model/evaluation_pack/README.md)
retains source provenance and derived stress cases. [The review workflow](new_benchmark/README.md)
manages independent labels and frozen test sets; private review records stay local.
Automated software tests remain in `tests/`. These are supporting evidence and
implementation files, not separate unpublished model accuracy claims.

## Safe refresh

From the repository root:

Commands below run from the **repository root**, with its virtual environment activated. See the root README for installation.

```bash
python "machine learning/evaluations/overview/build.py"
```

This executes only the overview and binary compatibility entry, and exports their
HTML previews to `evaluations/overview/`. It reads existing evidence without
training, modifying checkpoints, starting servers or touching annotation records.
Do not run every historical notebook with a wildcard: some contain training code.

For new measured results, follow the separate [classifier](classifier_hd/README.md)
and [tree](tree_counter_hd/README.md) reproduction instructions, then refresh the
overview. Those workflows require their stated data, weights and environment.
The [replacement benchmark workflow](new_benchmark/README.md) is the next step
for independently labelled unseen surveys.
