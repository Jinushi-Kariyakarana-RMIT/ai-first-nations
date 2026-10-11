# Tree counter evaluation results

Frozen current model; same tiled counting path and thresholds as the application.

| Test condition | Images | Count MAE | Bias | Precision @ IoU .5 | Recall @ IoU .5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| blur | 194 | 34.06 | -34.06 | 46.2% | 0.2% |
| darker | 194 | 31.02 | -31.02 | 41.1% | 3.8% |
| flip_horizontal | 194 | 33.83 | -33.79 | 42.3% | 0.5% |
| low_resolution | 194 | 33.93 | -33.93 | 35.3% | 0.3% |
| original | 194 | 33.71 | -33.71 | 44.7% | 0.6% |
| rotate_90 | 194 | 33.91 | -33.91 | 41.8% | 0.3% |

MAE is average absolute count error per image. Negative bias means undercounting.
Precision and recall compare individual boxes with the source labels; a matching total alone is insufficient.

## Interpretation

Use the original-image result as the external benchmark. Interpret each mock condition relative to its own original parents.
Do not pool the variants into a larger independent sample or use these test images to retrain while retaining the test claim.
The site-cluster bootstrap interval in summary.json is exploratory and cannot establish Mourilyan performance.

See per_image.csv, predicted_boxes.json and worst_cases for inspectable evidence.
