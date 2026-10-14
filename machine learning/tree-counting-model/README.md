# Standalone Tree Counter

This module estimates all visible trees independently from mangrove detection
and classification.

```text
input image -> shared 640x640 tiler -> YOLO crown detector -> duplicate removal -> estimated count
```

Bounding boxes are used internally for overlap handling and optional annotated
images. The primary public result is the estimated count and processing
diagnostics; the module does not expose GIS coordinates.

Training is optional and separate from running the application. The example
uses Apple Silicon MPS; choose a supported device for your host.

## Train

Commands below run from the **repository root**, with its virtual environment activated. See the root README for installation.

```bash
python "machine learning/tree-counting-model/tree_counter.py" train \
  --data "machine learning/data/training/tree_counter/data.yaml" \
  --epochs 30 \
  --device mps
```

Copy the selected checkpoint to
`machine learning/tree-counting-model/best_tree_counter.pt` after evaluation.
Model weights are generated artifacts and are intentionally not committed;
each environment must place the evaluated checkpoint at that path.

## Count

```bash
python "machine learning/tree-counting-model/tree_counter.py" count IMAGE.jpg \
  --model "machine learning/tree-counting-model/best_tree_counter.pt" \
  --overlay annotated.jpg
```

The output includes `estimated_trees`, `detected_crowns`, `tiles_processed`,
`average_confidence`, and the optional `overlay_path`.

The application and command-line default confidence threshold is `0.10`, selected
on the 18-image validation split to reduce counting error. Tile size remains 640,
overlap 64, and duplicate-removal IoU 0.45. The saved weights are unchanged.

This is an experimental estimate requiring manual review: external-image
performance remains poor and original-source annotation marks/site overlap limit
the accuracy evidence. See [tree evaluation](../evaluations/notebooks/YOLO_Tree_Counter_Evaluation.ipynb)
for baseline-versus-candidate results and remaining data requirements.
