### Application deployment update — 11 October 2026

The evaluated candidate is now applied to the application and command-line default:
confidence **0.10**, NMS IoU **0.45**, tile size **640**, overlap **64**.
The saved model weights are unchanged. All **18 actual HTTP acceptance checks passed**
with the deployed settings; see `../tree_counter_hd/deployment_validation/` for
settings, parity checks and measured runtime/memory. The 9 automated tests also passed.

The baseline Flask results below remain historical evidence. The evaluation runner
explicitly restores baseline settings inside its isolated test process to reproduce
those results. Use `flask_validation.py --current` to verify the deployed settings.
The candidate accuracy tables apply to the deployed configuration, with the same
source-data limitations. This deployment does not establish reliable survey accuracy.

Temporary upload copies, duplicate downloaded TIFF sources, browser profiles,
review screenshots and caches were removed. Labels, prepared evaluation images,
notebooks, final results and reproducibility scripts were retained.
