# Pipeline review — 14 October 2026

## Confirmed issues fixed

- Web startup could call training after a missing/corrupt classifier checkpoint.
  It now loads the saved classifier only and reports unavailability on failure.
- Repeated upload names overwrote earlier images. A corrupt repeated upload could
  then delete the previous valid file. Each upload now gets a unique stored name;
  overlay names inherit that unique identifier.
- Partial model failure could display “completed successfully”. Results now show
  a warning while retaining successful component results and mangrove error details.
- Image verification now handles Pillow's excessive-image-size exception as an
  invalid upload. Analysis rejects directories and unsafe route filenames.
- Benchmark CSV handles were left open; context managers now close them.
- Root documentation described an old threshold and historical results as current.
  It now links to the maintained overview.

## Verification

30 automated tests pass, including five new upload/startup regression checks.
Ten additional checks pass with real local weights: page routes, upload/result
rendering, model loading, overlay creation/dimensions/serving and exact count
parity with direct inference. See `pipeline_verification.json`.
This is a functional check on one real test image, not a new accuracy benchmark,
load test or validation of every possible deployment environment.

## Organisation and cleanup

Legacy research files moved together into `notebooks/training/`. The unused z20
raster and label pair moved together into local `data/legacy/` (about 2.7 GB), with
no image bytes discarded. Existing fixtures, active checkpoints, survey images,
training history, evaluation records and private reviews retain their locations.
Only regenerable Python/notebook caches were removed. Temporary test uploads were
automatically removed. Application and evaluation paths were checked after moves.

## Remaining limits

- Independent field accuracy and original category mapping still need labelled
  evidence. Functioning software does not resolve these model-evaluation gaps.
- Recent history is process-local; persisted history, bounded upload retention,
  concurrent inference/load testing and a job queue remain deployment work.
- Saved HD results describe their recorded code hashes. This runtime repair does
  not rewrite those historical hashes or claim a fresh HD evaluation. The classifier
  runner will require `--refresh` because `app.py` changed; follow its README when
  regenerating that evidence. Tree HTTP verification now follows unique upload URLs.
- An already running server needs a restart to load these source changes. This
  review did not stop the user's application or annotation server.
