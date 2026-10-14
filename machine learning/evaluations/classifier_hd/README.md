> **Current layout:** Mourilyan photos are external manual-demo inputs. Any 527-photo inventory or diagnostic results below are historical snapshots. See [current data availability](../CURRENT_DATA_LAYOUT.md).

# Binary and category classifier evaluation

Open the executed notebooks:

- [Binary_Mangrove_Evaluation.ipynb](../notebooks/Binary_Mangrove_Evaluation.ipynb)
- [Mangrove_Species_Evaluation.ipynb](../notebooks/Mangrove_Species_Evaluation.ipynb)

Matching HTML exports are in this directory. Both follow the ten sections of the
supplied rubrics. **Full HD compliance is not achieved:** independent reference
labels, training-source membership and category-mapping provenance are missing.
The notebooks deliberately show these as blocked, not as zero accuracy or passes.

## Reproduce

Use the project virtual environment and existing local model weights:

Commands below run from the **repository root**, with its virtual environment activated. See the root README for installation.

```bash
python "machine learning/evaluations/classifier_hd/run_notebooks.py" --refresh
```

When approved internal classifier sources are present, this inventories those images and runs actual-weight diagnostics and HTTP
checks on port 5053, verifies code/weight hashes, runs automated tests and executes
both notebooks using the same interpreter. It does not train, download weights,
change inference thresholds, upload imagery to a cloud service or stop the user's
server on port 5001. Derived fixtures, upload copies and browser profiles use an
automatically cleaned temporary directory. The screenshot uses local headless
Chrome when present; other environments should supply an actual observed capture.

Without `--refresh`, the runner validates saved evidence against checkpoint/code
hashes and re-executes the notebooks. This does not rerun model inference. Missing
or changed inputs require a fresh run. Install compatible pinned dependencies from
`requirements-lock.txt`; weights and survey images remain local dependencies.

## Evidence

- `source_inventory.csv`: original image IDs, paths, grouping hints, dimensions,
  SHA-256 and perceptual hashes. Source folders are not verified independent sites.
- `label_manifest.csv`: a persistent annotation sheet, initially empty label
  columns. Label provenance, reviewer and training exclusion must be supplied.
  Do not derive labels from either model's own predictions. The evaluation script
  does not overwrite an existing annotation sheet.
- `data_audit.json`, `near_duplicate_candidates.csv`: missing-label/split status
  and 13 candidate similar pairs requiring review, not asserted duplicates.
- `environment.json`, `requirements-lock.txt`: exact model/configuration/code
  hashes and environment. Category label order is a serving fallback, unverified
  against original checkpoint training provenance.
- `unlabelled_predictions.*`, `tile_predictions.json`: all diagnostic source/tile
  predictions and lineage. These are not scored accuracy examples.
- `tiler_coverage.csv`: original-pixel coverage for 5 image dimensions.
- `application_checks.csv`: 67 passing technical checks, with actual and simulated
  branches distinguished. Numerical parity tolerance is 1e-6.
- `latency_memory.csv`: warm full-application response measurements; reference
  repeated 3 times, other cases once. Full-resolution case is 4032x3024.
- `metrics.py`: tested classification/calibration metric calculations; no model
  accuracy results are generated without independent ground truth.
- `experiment_protocol.json`: two proposed validation-only comparisons per model.
  These predictive improvement experiments remain unexecuted due to missing labels.
- `repairs.json`: code audit and two fixes, not claims of improved checkpoint accuracy.
- `previous_evaluation_audit.json`: historical binary count inconsistency and split
  concerns; prior category evaluation was only an artifact/readiness check.
- `specification_audit.csv`: criterion-by-criterion status, without self-awarded marks.
- `automated_tests.txt`, `verification.json`, `notebook_execution.json`: verification
  and execution records. Known-answer tests are software checks, not model accuracy.

## Repairs

The app no longer attempts category classification if its binary detector is
unavailable. Training/validation subsets now use independent transform objects;
this prevents validation settings from overwriting training augmentation. No
existing model was retrained, and historic data leakage is not repaired by this fix.

## Required next inputs

1. Independently reviewed binary labels and Orange/Red/Yellow labels, with annotation
   provenance and confirmation that those categories are meaningful.
2. Original training-source registry and verified checkpoint class-index metadata.
3. Source/site-separated labelled validation and untouched final-test images, including
   realistic negatives and difficult conditions, with group overlap reviewed.

With those inputs, score the exact model with `metrics.py`, execute the two registered
validation-only comparisons, freeze settings before final testing, and add genuine
confusion matrices, calibration/ROC/PR plots, site analysis and group-level uncertainty.
Do not reuse the present diagnostics as a newly blinded holdout: their outputs have
already been inspected.

## Historical provenance recovery

See [dataset and checkpoint trace](DATA_RECOVERY.md). Both saved models were
matched byte-for-byte to original commits. Historical category metrics were found
in the prior-team report, but conflicting sample counts prevent treating them as
verified current accuracy. The missing original labelled drone datasets remain unrecovered.

## Alternative adopted

The original dataset cannot be recovered. Use the [new benchmark workflow](../new_benchmark/README.md) to collect reviewed labels, import unseen survey sources, freeze a final test and score the existing model. Historical provenance remains documented; missing old files no longer prevent building new evidence. Verified reference labels are still required.
