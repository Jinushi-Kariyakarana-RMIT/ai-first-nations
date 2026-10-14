IMAGE TILER TEST SET

01_small_320x240.jpg
- Smaller than 640x640.
- Tests your behavior for images smaller than one tile.

02_exact_640x640.jpg
- Exactly one 640x640 tile.
- Expected: 1 tile.

03_large_1280x960.jpg
- Larger than tile size and not an exact multiple vertically.
- Tests full tiles + edge handling.

04_tree_counter_1920x1280.jpg
- Tree-counter-style synthetic aerial canopy image.
- Exact multiple of 640 in both dimensions.
- With non-overlapping 640x640 tiling, expected: 3 columns x 2 rows = 6 tiles.

expected_640_tiles_nonoverlap/
- Reference output for 04_tree_counter_1920x1280.jpg using 640x640 non-overlapping tiles.

NOTE:
These are synthetic test images designed to test tiler dimensions/edge behavior, not model accuracy.
