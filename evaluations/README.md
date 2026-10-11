# Model Evaluation Guide

## Additional external photo benchmark

The [tree-counter evaluation pack](../tree-counting-model/evaluation_pack/README.md)
contains 194 real NEON photographs with 6,633 published crown annotations and
970 labelled stress variants. See its [completed results](../tree-counting-model/evaluation_pack/results/RESULTS.md).
This separate external benchmark uses the application's actual tiled counting
path. It does not replace the historical notebook results below and is not a
Mourilyan-specific accuracy evaluation.

This folder is the single source of truth for Mangrove AI evaluation. It keeps
the evaluation criteria, executed evidence and known gaps together so another
team member can understand what has been tested without reading the training
code first.

## Current Readiness

| Component | Evidence | Main result | Status |
| --- | --- | --- | --- |
| Binary mangrove detector | Executed tile-level test | Accuracy 98.96%, macro F1 98.89% | Needs a flight-level external test |
| Mangrove category classifier | No labelled test data in this repository | No defensible test metric | Not evaluated |
| Tree crown detector | 9 test images, 759 crowns | Precision 78.9%, recall 73.9%, mAP50 77.9% | Experimental |
| Tree count | Same 9 test images | MAE 15.56, MAPE 15.74%, bias -14.89 trees/image | Experimental; undercounts |
| Image tiler | Synthetic and repository fixtures | No uncovered pixels; expected tiles reproduced | Preprocessing checks passed |
| Combined web pipeline | No labelled end-to-end test set | No end-to-end metric | Not evaluated |

**Overall decision:** the current application is suitable for continued
development and controlled demonstrations, but the models are not yet
validated for unattended real-world monitoring.

## Important Interpretation

The binary detector's 98.96% score is a tile-level result. Original images
were tiled before the random train/validation/test split, so neighboring tiles
from the same flight may occur in more than one split. This can make the score
more optimistic than performance on a fresh flight.

The tree-counter split contains no exact duplicate images, but all eight test
site prefixes also occur in training. Its nine-image test result therefore has
limited evidence of performance at a new geographic location.

The multi-class model file exists, but there is no executed, reproducible test
result or labelled test dataset in this repository. A saved model is not proof
of accuracy.

## Folder Contents

- `evaluation_criteria.md`: metrics and acceptance checks expected for each model.
- `notebooks/evaluate_binary_detector.ipynb`: training and recorded binary test evidence.
- `notebooks/evaluate_multiclass_classifier.ipynb`: reproducible readiness check for the missing multi-class evaluation data.
- `notebooks/evaluate_tree_counter_dataset.ipynb`: dataset structure and annotation checks.
- `notebooks/evaluate_tree_counter_model.ipynb`: tree detection and counting metrics.
- `notebooks/evaluate_image_tiler.ipynb`: preprocessing coverage and tile correctness.
- `notebooks/evaluate_combined_pipeline.ipynb`: readiness check for an end-to-end labelled evaluation set.

## Metric Meanings

- **Precision:** how many predicted objects or classes are correct.
- **Recall:** how many labelled objects or classes the model finds.
- **F1:** balance between precision and recall.
- **mAP50:** object-detection quality at an intersection-over-union threshold of 0.50.
- **mAP50-95:** stricter object localisation quality across multiple thresholds.
- **MAE:** average absolute counting error in trees per image.
- **MAPE:** average absolute counting error as a percentage of the true count.
- **Bias:** signed average error; a negative value means systematic undercounting.

## Re-running Evaluations

Run notebooks from the repository root so they can locate the application,
datasets and model weights consistently:

```bash
source .venv/bin/activate
jupyter nbconvert --execute --to notebook --inplace evaluations/notebooks/<notebook>.ipynb
```

Do not overwrite the recorded model results until the replacement evaluation:

1. uses a split grouped by flight, site and date before tiling;
2. preserves an untouched external test set;
3. records the exact model checksum and inference thresholds; and
4. evaluates the same tiled inference path used by the web application.
