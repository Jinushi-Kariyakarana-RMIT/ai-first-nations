# Tree-counter specification audit and executable evaluation

Open `../notebooks/YOLO_Tree_Counter_Evaluation.ipynb`. The companion
`YOLO_Tree_Counter_Evaluation.html` is a read-only rendered copy with saved figures.
The notebook follows all 11 sections of the supplied specification and distinguishes
implemented evaluation methods from unresolved data requirements. It does not
claim an academic mark or full HD compliance.

## Reproduce

Use the existing project Python environment and local checkpoints. Install missing
evaluation dependencies from `requirements-lock.txt` if needed. The original
`dataset1` images/labels and prepared NEON pack must be present. Regenerate the
public pack with `tree-counting-model/prepare_evaluation_pack.py` if necessary.

```bash
.venv/bin/python evaluations/tree_counter_hd/run_notebook.py
```

This runner explicitly uses its own Python executable as the kernel; a global
Anaconda/Jupyter installation may otherwise choose a different interpreter.
The notebook defaults to recomputing metrics and HTTP tests, without retraining
or changing the app's saved settings. It starts a temporary Flask server on
127.0.0.1:5052 and closes it afterward. Do not occupy that port during execution.

The screenshot step uses locally installed Google Chrome in headless mode when
available. On another OS, capture the same actual application result through a
supported browser and save `flask_result.png`; no screenshot is fabricated.
If no screenshot exists, that presentation cell needs the actual capture first.

## Evidence

- `environment.json`, `requirements-lock.txt`: hardware, versions, seed, model
  checksum and inference settings. No weights are downloaded or trained.
- `split_manifest.csv`, `split_audit.json`: image IDs, source groups, labels,
  checksums and limitations. Historical train/test prefixes overlap.
- `training_history/`: preserved original training settings and 30-epoch history.
- `experiment_protocol.json`, `validation_experiments.csv`, `locked_settings.json`:
  confidence and NMS studies on validation only; frozen candidate before new tests.
- `baseline_*` and `candidate_*`: full-image predictions, pre-dedup boxes, per-tile
  diagnostics, operational metrics and official COCO AP@300/PR curves.
- `marker_diagnostic/`: validation-only sensitivity to small red source-image marks.
- `challenge_manifest.json`, `negative_*`, `overlap_experiment.csv`: source-derived
  negative regions and labelled photographic composites. They are not new surveys.
- `flask_acceptance.csv`, `flask_latency_memory.csv`, `flask_summary.json`: actual
  HTTP tests, parity, failure handling, timings and sampled process RSS.
- `specification_audit.csv`: each rubric criterion, evidence and remaining gaps.
- `notebook_execution.json`, `automated_tests.txt`: actual execution and test records.

## Critical findings

The original source images contain visible red cross marks near crowns in inspected
train, validation and test examples. Validation-only red-region suppression causes
a large performance change. This is evidence of sensitivity to a potential target
cue, not proof that every mark was removed or that inpainting restores real imagery.
Obtain the original unmarked photographs and independently reviewed labels.

The external NEON source is different from the local training source, but its
results have already been inspected. It is reused benchmark evidence, not a newly
blinded final holdout. The original nine-image split has site-prefix overlap with
training. Neither limitation can be fixed merely by reorganising notebooks.

Mourilyan remains unlabelled. Running a fresh image through Flask demonstrates
functionality and runtime only. No counting-accuracy value is assigned to it.
Site-level sums in this package are image annotation totals, not unique-tree
survey inventories; overlapping frames can contain the same tree.

## What still prevents full compliance

1. Clean source imagery and source/site/date-separated data established before tiling.
2. Independently reviewed mangrove crown ground truth and a genuinely untouched survey holdout.
3. An agreed image-level-only scope or a valid method to reconcile unique trees across survey frames.
4. Reviewed difficult-condition labels and broader full-field boundary tests.
5. Hardware-specific latency/memory limits and additional full-resolution repetitions.

The deployment decision is **Not Ready for unattended mangrove inventory**.
Poor model scores do not by themselves prevent an HD-quality evaluation, but these
unfulfilled non-negotiable evidence requirements prevent claiming complete compliance.
The broader project-wide notebook reorganisation is deliberately deferred in
`../DEFERRED_NOTEBOOK_PLAN.md`.

## Deployed settings

See [deployment update](DEPLOYMENT.md). Confidence 0.10 is now the application
default. Historical baseline tests explicitly use 0.15 in an isolated process.
Run `.venv/bin/python evaluations/tree_counter_hd/flask_validation.py --current`
for current-application checks, saved separately in `deployment_validation/`.
Temporary files removed are listed in `cleanup_record.json`; original TIFF download
copies are fetched again by `prepare_evaluation_pack.py` if regeneration is needed.
