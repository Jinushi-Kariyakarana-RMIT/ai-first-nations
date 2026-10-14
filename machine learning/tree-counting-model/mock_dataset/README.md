# Mock tree dataset

For actual photo-based model evaluation, use the larger
[evaluation pack](../evaluation_pack/README.md). This folder contains only
schematic software fixtures.

These computer-drawn tree symbols have exact labels by construction. They are
schematic test fixtures, not realistic drone photographs or expert annotations.
Use them to check upload, label loading, count comparison and evaluation output.
Do not use their scores to claim accuracy on real trees or to select production
model settings. Training on these symbols may reduce real-photo performance.

There are 30 unique images: 18 training, 6 validation and 6 test images. Each
split includes empty, single-tree, sparse (8), medium (24), dense (60), and
tile-boundary (12) cases. Most images are 640 × 640; boundary cases are
1280 × 960 to exercise the app's overlapping tiles. Splits have unique images
but share the same generator and are not independent real-world samples.

- `train/images`, `valid/images`, `test/images`: uploadable PNG images.
- Matching `labels` folders: YOLO labels, one row per synthetic crown:
  `class_id center_x center_y width height`, coordinates normalized to 0–1.
  Class 0 means tree. Empty images have empty label files.
- `counts.csv`: expected count, split and paths for every image.
- `data.yaml`: dataset configuration; regenerating updates its local root path.
- `label_preview.jpg`: test examples with ground-truth boxes.
- `evaluation.json`: current model's test predictions and count errors.
- `predictions`: the current model's boxes on the six test images.

From the project root, reproduce the data and run the existing model:

```bash
.venv/bin/python tree-counting-model/mock_dataset.py --evaluate
```

The generator uses fixed seeds and validates unique images and in-bounds boxes.
It writes only this mock dataset folder. It does not train or replace the model.
Generated PNG/JPG images are ignored by the repository's existing Git rules;
run the generator after cloning to recreate them.

For a simple app check, upload `test/images/test_0_sparse.png` and compare the
reported count against 8. A mismatch is not an app failure: the existing model
was trained on photographs and may not recognise these symbols.

For meaningful accuracy evaluation, manually label a separate selection of the
fresh Mourilyan Harbour photos, keep them out of training, and compare both
counts and individual detections against those human-reviewed labels.
