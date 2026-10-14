# Data — images and labels

- **testing/**: all evaluation inputs, including real photos, synthetic fixtures
  and diagnostic image variants.
- **training/tree_counter/train/**: original tree-model training images and labels.
- **training/tree_counter/valid/**: validation images used for selecting settings.
- **legacy/**: unused older raster inputs preserved for reference.

Training and validation images are not part of the held-out test dataset.
`training/tree_counter/test` is a shortcut to `testing/tree_counter`, retained for
YOLO's existing dataset configuration. It is not a duplicate dataset.

Old dataset locations elsewhere in the workspace are relative compatibility
shortcuts. Use this folder as the canonical place to browse data. Do not remove
those shortcuts without migrating historical notebooks and recorded manifests.
