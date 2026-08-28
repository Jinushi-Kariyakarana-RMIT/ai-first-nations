# Standalone Tree Counter

This module counts trees in an image without connecting to the mangrove/non-mangrove detector yet.

The intended flow is:

```text
input image -> 640x640 tiles -> YOLO tree detector -> duplicate removal -> total count
```

The user-facing output is only the total count, but the module still uses object-detection boxes internally so overlapping tiles do not double-count the same tree.

## Install

From the project root:

```bash
.venv/bin/python -m pip install ultralytics
```

## Train

```bash
cd tree-counting-model
../.venv/bin/python tree_counter.py train --data dataset1/data.yaml --epochs 100
```

After training, copy the best weights to:

```text
tree-counting-model/best_tree_counter.pt
```

Ultralytics usually saves the best model under:

```text
runs/detect/tree_counter/weights/best.pt
```

## Count Trees

```bash
cd tree-counting-model
../.venv/bin/python tree_counter.py count path/to/image.jpg --model best_tree_counter.pt
```

Example output:

```json
{
  "tree_count": 184,
  "tiles_processed": 12,
  "raw_detections": 197,
  "average_confidence": 0.71
}
```
