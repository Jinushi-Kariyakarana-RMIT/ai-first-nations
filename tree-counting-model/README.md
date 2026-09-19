# Standalone Tree Counter

This module estimates all visible trees independently from mangrove detection
and classification.

```text
input image -> shared 640x640 tiler -> YOLO crown detector -> duplicate removal -> estimated count
```

Bounding boxes are used internally for overlap handling and optional annotated
images. The primary public result is the estimated count and processing
diagnostics; the module does not expose GIS coordinates.

## Train

```bash
.venv/bin/python tree-counting-model/tree_counter.py train \
  --data tree-counting-model/dataset1/data.yaml \
  --epochs 30 \
  --device mps
```

Copy the selected checkpoint to
`tree-counting-model/best_tree_counter.pt` after evaluation.

## Count

```bash
.venv/bin/python tree-counting-model/tree_counter.py count IMAGE.jpg \
  --model tree-counting-model/best_tree_counter.pt \
  --overlay annotated.jpg
```

The output includes `estimated_trees`, `detected_crowns`, `tiles_processed`,
`average_confidence`, and the optional `overlay_path`.
