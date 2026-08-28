AIFN_Mangrove_README.md

## Image Tiler Update

The previous image tiler lived in `img-tiler/img-tiler.py`. It was a small standalone script that cropped one image into fixed-size square tiles and saved those tiles to disk. It was useful for quick experiments, but it was not connected to the Flask model pipeline and did not support overlap, padding, edge coverage, or reusable in-memory tiles.

The new shared tiler lives in `flask-application/image_tiler.py`. It is designed for model preprocessing, so both the mangrove detector and future tree counter can use the same tiling behavior. It supports configurable tile sizes, overlap between tiles, padding for small or edge tiles, optional empty-tile skipping, tile position tracking, and direct conversion into PyTorch tensor batches.

In short, the old tiler was an offline crop-and-save helper. The new tiler is a reusable preprocessing module for the web app and ML models.
