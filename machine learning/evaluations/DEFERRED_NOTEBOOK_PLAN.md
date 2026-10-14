# Notebook consistency plan — implemented

Executed at the user's request on 11 October 2026, as a presentation and entry-point
consolidation. Application code, deployed weights, private labels and saved model
measurements were not changed or rerun.

- Added `notebooks/00_evaluation_overview.ipynb` as the common entry point.
- Retained detailed component notebooks and their shared evaluation implementations.
- Added `overview/evidence.py` to load summary tables from saved source artifacts,
  with source hashes and a deployment/configuration consistency check.
- Separated historical direct-YOLO records from application-path measurements.
- Included external benchmark provenance and separate application control results.
- Replaced the unsafe legacy binary entry with a read-only audit/redirect; preserved
  its historical content in `archive/evaluate_binary_detector_historical.ipynb`.
- Removed outdated readiness scores from the guide. Original binary dataset paths
  and contradictory counts remain unresolved historical evidence, not silently
  repaired or promoted to current accuracy. Current notebooks use relative paths.
- Kept independent classifier and combined-pipeline accuracy marked not evaluated.
- Kept software tests outside notebooks and distinct from model accuracy.

Refresh only the lightweight entry points with:
`.venv/bin/python evaluations/overview/build.py`.

Outstanding scientific requirements (reviewed labels, independent surveys and
category mapping provenance) belong to the replacement benchmark workflow;
completing this notebook organisation does not establish full HD compliance.
