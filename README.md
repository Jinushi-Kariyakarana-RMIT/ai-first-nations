# Mangrove AI

Mangrove AI is an AI First Nations prototype for analysing drone and aerial
imagery. It combines mangrove detection, mangrove-category classification and
an experimental standalone visible-tree counter in one Flask application.

## Analysis Pipeline

```text
uploaded image
      |
shared image tiler
      |
      +-- binary mangrove detector
      +-- mangrove category classifier (when mangrove is detected)
      +-- standalone visible-tree counter
      |
structured results and optional tree-crown overlay
```

The tree counter runs independently and counts all visible trees. It does not
claim that each counted tree is a mangrove and does not expose GIS coordinates.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
FLASK_RUN_PORT=5001 python -m flask --app flask-application/app.py run
```

Open `http://127.0.0.1:5001`.

The Flask application loads these local model files without downloading
ImageNet weights during inference:

- `flask-application/bestBinary.pth`
- `flask-application/best_mangrove_model.pth`
- `tree-counting-model/best_tree_counter.pt`

Tree-counter weights are generated and ignored by Git. Place the evaluated
checkpoint at the path above. If any model is missing, the remaining modules
continue running and the interface reports the unavailable component.

## Shared Image Tiler

The previous `img-tiler/img-tiler.py` was an offline crop-and-save experiment.
The shared `flask-application/image_tiler.py` works in memory and is used by all
three model paths. It supports configurable tile size and overlap, complete
edge coverage, padding, optional empty-tile skipping and tile positions.

Current defaults:

- Mangrove models: 512 x 512 tiles with 64-pixel overlap.
- Tree counter: 640 x 640 tiles with 64-pixel overlap and global duplicate
  suppression.

The executed tiler evaluation is in
`notebooks/evaluate_image_tiler.ipynb`.

## Tree Counter

Train the first baseline with:

```bash
python tree-counting-model/tree_counter.py train \
  --data tree-counting-model/dataset1/data.yaml \
  --epochs 30 \
  --device mps
```

The checked dataset contains 64 training, 18 validation and 9 test images,
with 7,526 labelled crowns. All images are 640 x 640. Structural checks found
no missing image-label pairs, malformed rows, empty labels or duplicate image
hashes. See `notebooks/evaluate_tree_counter_dataset.ipynb`.

### Experimental Baseline Results

The untouched nine-image test split produced:

| Metric | Result |
| --- | ---: |
| Precision | 0.789 |
| Recall | 0.739 |
| mAP50 | 0.779 |
| mAP50-95 | 0.324 |
| Count MAE | 15.56 trees/image |
| Count RMSE | 24.25 trees/image |
| Count MAPE | 15.74% |
| Count bias | -14.89 trees/image |

The `0.15` inference threshold was selected using validation count MAE before
test evaluation. The model undercounts dense scenes, so it is labelled
**Experimental** in the application. The full reproducible evaluation is in
`notebooks/evaluate_tree_counter_model.ipynb`.

## Testing

```bash
python -m unittest discover -s tests -v
python -m compileall -q flask-application tree-counting-model tests
```

The notebooks should also be executed top to bottom after changing datasets,
tiling behavior or model weights.

## Current Limitations

- Orange, Red and Yellow are retained as dataset categories because the
  repository does not establish that they are biological species.
- The tree dataset is small, single-resolution and may not represent other
  drone heights, locations, seasons or canopy densities.
- Tree counts are estimates and should not be treated as exact inventories.
- Recent uploads are held in process memory and reset when Flask restarts.
- GIS export, change-over-time analysis and carbon estimation are future work.
