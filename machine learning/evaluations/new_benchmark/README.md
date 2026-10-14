> **Current layout:** Mourilyan photos are external manual-demo inputs. Any 527-photo inventory or diagnostic results below are historical snapshots. See [current data availability](../CURRENT_DATA_LAYOUT.md).

# Alternative: build a new benchmark

The historical drone dataset cannot be recovered. This workflow creates a new,
reviewed evaluation dataset instead. It does not replace missing labels with model
predictions and does not claim full HD compliance before the required evidence exists.

## Review locally

Commands below run from the **repository root**, with its virtual environment activated. See the root README for installation.

```bash
python "machine learning/evaluations/new_benchmark/benchmark.py" serve
```

Open http://127.0.0.1:5054. The historical 527 Mourilyan records are retained but excluded from the active review queue because their sources are no longer internal. Import an approved internal manifest before reviewing new images. The reviewer sees
no model prediction, enters a label and supporting evidence, and can use Uncertain,
Unknown or Mixed. A knowledgeable reviewer must confirm labels; arbitrary guesses
cannot establish ground truth. Orange/Red/Yellow must not be inferred from colour alone.

Reviews and their history are stored locally in `reviews.sqlite3`, excluded from Git.
`prepare` preserves previous reviews. Image hashes detect changed source files.
Export the complete labelled/provenance manifest with:

```bash
python "machine learning/evaluations/new_benchmark/benchmark.py" export
```

The export is also local by default. No private source photos or reviewer records
are uploaded by this workflow. The existing application remains on port 5001.

## New validation and final-test sources

Use `import_manifest.example.csv` as the schema for **new** images placed inside the
workspace. Supply site/survey IDs, source provenance and evidence excluding training
use. Import separate validation and test sites:

```bash
python "machine learning/evaluations/new_benchmark/benchmark.py" import --manifest PATH.csv
```

The importer rejects repeated image hashes and sites assigned to another split.
A different filename or route alone is not sufficient independence. Review geographic
and temporal overlap and near-duplicates, including overlap with development images.
A reviewer must record this evidence in `holdout_review.json`, using its example.
Class-balanced coverage requires real non-mangrove negatives (water, mud, other
vegetation/trees and structures) as well as mangroves. No defensible fixed sample size
can be promised without class/site variability and the desired confidence interval.
Start with multiple independent surveys and expand coverage based on uncertainty.

After review, freeze the test manifest and existing model hashes **before** scoring:

```bash
python "machine learning/evaluations/new_benchmark/benchmark.py" freeze
python "machine learning/evaluations/new_benchmark/benchmark.py" score --split test
```

Frozen labels cannot be edited. The evaluator verifies source/model hashes and
membership and will not overwrite a previous results folder. Test results must not
be used to tune thresholds. Development and validation can be scored separately:

```bash
python "machine learning/evaluations/new_benchmark/benchmark.py" score --split development
python "machine learning/evaluations/new_benchmark/benchmark.py" score --split validation
```

Both binary classes and reviewed provenance are required. Uncertain binary labels
are excluded and their number reported. Results contain per-source predictions,
confusion matrices, class metrics, ROC/PR points, Brier/log loss/ECE and reliability
bins. These are measured only when real reviewed labels exist. This scorer evaluates
the fixed inherited baseline; the two rubric improvement experiments, subgroup
analysis, plots and group-bootstrap confidence intervals remain subsequent work once
an independent validation/test dataset exists.

## Category classifier decision

Category evaluation stays blocked unless `category_mapping.json` verifies the exact
checkpoint label order with reviewer evidence. Copy the example only when that
verification exists. All three categories must be represented in eligible reviewed
mangrove images. Unknown/Mixed examples are not given guessed categories. The score
is conditional category accuracy, not end-to-end accuracy through the binary gate.

If the inherited mapping cannot be verified, collect a clearly defined category
dataset and **train a replacement classifier**, retaining train/validation/test site
separation and saving the mapping with its weights. No replacement can be trained
honestly from the currently unlabelled images, and this workflow does not silently
replace the live model.

## Current status

- Review tool and evaluation machinery implemented and tested.
- Historical snapshot: 527 development candidates were prepared with no reviewed labels. Those external demo sources are now excluded from the active queue; import an approved internal dataset.
- No new independent accuracy claimed. Full HD completion awaits reviewed labels,
  unseen test surveys, mapping verification (or a replacement category model), and
  the remaining rubric experiments/uncertainty analysis.
