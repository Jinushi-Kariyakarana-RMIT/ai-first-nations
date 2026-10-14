### Alternative adopted: create a new benchmark

The original labelled drone dataset cannot be recovered. A new local [benchmark review workflow](../new_benchmark/README.md) now replaces the recovery dependency.

Run `.venv/bin/python evaluations/new_benchmark/benchmark.py serve` and open **http://127.0.0.1:5054**. The 527 inspected Mourilyan images are development candidates, not a newly blinded test set. Reviewers supply labels and their evidence without seeing model predictions; Unknown/Uncertain/Mixed remain explicit.

Import unseen validation/test surveys separately. The workflow rejects exact duplicate images and cross-split site reuse, requires a documented overlap/provenance review, freezes final-test membership and checkpoint hashes, and scores only reviewed sources. Category scoring additionally requires evidence verifying the saved class mapping; otherwise a new category model must be trained from independently labelled data.

**Current independent accuracy remains unmeasured.** The tool is ready, but zero reference labels have been supplied. Completing the HD rubric still requires the reviewed benchmark and the planned validation-only experiments, subgroup/uncertainty analysis and final test. Software checks are not substitutes for that evidence.
