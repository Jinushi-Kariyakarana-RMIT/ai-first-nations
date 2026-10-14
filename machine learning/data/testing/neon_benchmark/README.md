> **Current layout:** Mourilyan photos are external manual-demo inputs. Any 527-photo inventory or diagnostic results below are historical snapshots. See [current data availability](../../../evaluations/CURRENT_DATA_LAYOUT.md).

# Tree counter evaluation pack

This pack contains **1,164 labelled images**: **194 real aerial photographs**
from **22 NEON sites**, containing **6,633 published tree-crown annotations**,
plus **970 deterministic mock stress images** derived from those photographs.
It replaces the earlier circle drawings as the useful model-evaluation set.
The old `mock_dataset` remains a basic software fixture only.

## Open these first

- `label_preview.jpg`: overview of real photos with reference boxes.
- `labelled_previews/`: all 194 originals with numbered reference boxes.
- `real_test/images/`: clean photos to upload to the application.
- `real_test/labels/`: matching YOLO labels; class `0` means tree.
- `counts.csv`: expected counts with easy-to-match filenames.
- `results/RESULTS.md`: completed evaluation of the current model.
- `results/per_image.csv`: individual predictions and errors.
- `results/worst_cases/`: predicted boxes on the worst original examples.

Upload **clean photos** from `real_test/images`, not the previews with boxes.
Find the same filename in `counts.csv` to compare its expected and reported
counts. All labels are also provided as normalized YOLO bounding boxes so
individual detections can be checked, not just totals.

## What the mock images test

Each real image has five related variants, with 194 images per condition:

| Folder under `mock_stress` | Change | Reference labels |
| --- | --- | --- |
| `flip_horizontal` | Mirror left/right | Boxes mirrored exactly |
| `rotate_90` | Rotate 90 degrees counterclockwise | Boxes and image dimensions rotated |
| `darker` | Brightness multiplied by 0.65 | Same boxes and count |
| `blur` | Gaussian blur, radius 1.2 pixels | Same boxes and count |
| `low_resolution` | Reduce width/height by half, then restore size | Same boxes and count |

These are repeatable image-processing augmentations, not AI-generated scenes
or labels predicted by the model under test. Appearance degradations test
recovery of original scene annotations even when some details become unclear.
They do **not** add new locations or new trees. There are 194 original image
groups, not 1,164 statistically independent observations. Images within the
same site may also be correlated. The pack is evaluation-only; do not train
on these originals or their variants while claiming they are held-out tests.

## Source and provenance

Source: [NeonTreeEvaluation](https://github.com/weecology/NeonTreeEvaluation),
maintained by Ben Weinstein and the Weecology team.
Dataset archive: [Zenodo 5914554](https://zenodo.org/records/5914554).
Pinned Git revision: `96b29d566ca7ea9f604de3e1235c503ab101cb8a`.
Repository license: **CC0 1.0**, retained in `source/LICENSE`.

All 194 files in `evaluation/RGB` that have a matching published XML annotation
were included; unannotated photos were excluded rather than labelled as empty.
Annotations originate from the dataset's human image annotation process, not
from our tree counter. These are reference annotations, not a claim that every
source label is flawless. Standing dead trees are included under the source
definition; fallen trees are not.

Original TIFFs and XMLs are retained under `source/`, and every downloaded file
was verified against its pinned Git blob checksum. TIFFs were converted to RGB
PNG without resizing. Source XML coordinates are interpreted directly as image
box boundaries (no one-pixel offset). No source boxes needed clipping. All
labelled previews make this conversion inspectable.

`manifest.csv` records source URL, revision-linked filename, parent group,
site, condition, dimensions, expected count, image pixel checksum and label
checksum. `quality_audit.json` includes the full site distribution and audit.

## Completed checks and limits

- All 1,164 image/label pairs load; their counts, bounds and checksums agree.
- Original-image counts range from 1 to 292 crowns.
- No exact duplicate original images or source boxes were found.
- No exact decoded-image matches against the existing 91 dataset images were
  found. A simple 64-bit difference-hash screen also found no close matches at
  distance 8 or less; the nearest was 15. This screening cannot prove absence
  of every cropped, resized, geographic or pretraining overlap.
- Non-square fixture checks verify both geometric box transformations.
- Detection matching enforces one prediction per reference tree.
- Model checksum, thresholds, library versions and per-image predictions are
  recorded in `results/summary.json` and the supporting files.

This is a substantial **initial external benchmark**, not a guarantee of
statistical sufficiency for every deployment. SJER and TEAK contribute 112 of
194 images; many other sites have few examples. There are no fully tree-free
originals, so dedicated negative-scene performance remains untested. All
originals are about 400 pixels per side, so this pack does not establish
large-image seam performance. The existing tiler tests cover software behavior.

Most importantly, these are general US forest images, **not Mourilyan mangrove
ground truth**. Forest type, camera scale, illumination and source label
definitions can differ from the training data and the intended deployment.
Results must be described as performance on this source and inference setup.

## Mourilyan data

`mourilyan_inventory.csv` inventories all **527 fresh photographs across three
flight-route folders**, each 4032 × 3024. Their expected counts are blank,
not zero. They are not included in benchmark scores because no human-reviewed
labels have been supplied. The photos have not been uploaded to an external
annotation service or modified.

For new site-specific accuracy evaluation, obtain a separate approved, independently labelled survey. The external Mourilyan folder is reserved for manual application trials.

## Reproduce

From the project root:

Commands below run from the **repository root**, with its virtual environment activated. See the root README for installation.

```bash
python "machine learning/tree-counting-model/prepare_evaluation_pack.py"
python "machine learning/tree-counting-model/evaluate_prepared_pack.py"
```

Preparation downloads only the public dataset if not cached. The evaluator
uses the current `best_tree_counter.pt` without changing it. Confidence 0.15,
NMS overlap 0.45, 640-pixel tiles and 64-pixel tile overlap match the app. It
reports count MAE/RMSE/bias and one-to-one box precision/recall at IoU 0.50.
These are fixed-threshold metrics, not mAP. A site-cluster bootstrap interval
is exploratory; it does not represent a Mourilyan confidence interval.

Images are ignored by the repository's existing Git rules; preserve this
folder or regenerate it after cloning. Approximate total size: 463 MB before
the optional labelled previews. No existing training data or model weights
were replaced, and no training was run.

`source/README.md` is the preserved upstream dataset description. Its examples
and figure paths refer to the original upstream repository, not this application.
