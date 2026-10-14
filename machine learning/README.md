# Machine learning — folder guide

| Folder | What it is for |
|---|---|
| **data/testing/** | All testing images and labels — start here to find test data |
| **data/training/** | Images used for training and validation; separate from testing |
| **data/legacy/** | Older unused imagery kept for reference |
| **tests/** | Python checks that verify software works; this is test code, not a dataset |
| **runs/** | Generated training outputs: saved checkpoints, logs and learning curves |
| **notebooks/training/** | Historical interactive training experiments |
| **evaluations/notebooks/** | Readable model evaluation reports; start with `00_evaluation_overview.ipynb` |
| **evaluations/** | Evaluation scripts, metrics, evidence, rubric audits and label-review tools |
| **tree-counting-model/** | Tree-counter implementation and deployed checkpoint |
| **flask-application/** | The web application, classification code and deployed classifier weights |

The old dataset names are **compatibility shortcuts**, not extra copies. All test
image content lives in `data/testing/`; real training/validation images live in
`data/training/`. Preview figures and overlays in evaluation folders are results,
not additional input datasets. The root `yolo11n.pt` is a local base training model.

For setup and startup, use the [main README](../README.md). For server deployment,
use the [deployment guide](../docs/DEPLOYMENT.md). Evaluation-specific commands
are documented in `evaluations/README.md` and its component guides.
Open notebooks with this ML workspace as the working directory.

Mourilyan Harbour demo photos are stored separately in `../../Mourilyan_Harbour_Fresh_Test_Images/` (the DS Proejct folder), outside the repository. They are for manual application trials, not internal accuracy testing. Historical evaluation inventories retain their original paths; they are not current dataset manifests.
