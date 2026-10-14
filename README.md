# Mangrove AI — AI First Nations

Mangrove AI is a browser-based application for reviewing drone and aerial photos.
It runs on the computer or server where the project is installed.

## For business owners

Upload a JPG, JPEG or PNG image, up to 50 MB. The application provides:

| Feature | What you receive |
|---|---|
| Mangrove detection | Whether the image is classified as containing mangroves, with model confidence |
| Category classification | Orange, Red or Yellow category when mangroves are detected |
| Tree counting | Estimated visible tree crowns and an image marking detected crowns |

The three features are implemented and have passed local functional checks.
Counts are estimates of **all visible tree types**, not verified mangrove-only
inventories. Orange/Red/Yellow are dataset categories, not verified biological
species. Model confidence is not measured accuracy. Independent accuracy on new
local surveys still needs reviewed reference labels.

**To use it:** open the address supplied by your technical team, choose **Analyse**,
upload a photo, and review the predictions and crown overlay. A warning indicates
that one or more parts of the analysis could not complete.

The current release has no user accounts, GIS export, carbon estimation or
persistent analysis-history database. Upload files remain on the host computer;
the recent-history list resets when the application restarts. The application
has no built-in upload retention policy.

## For the technical team: run locally

Run these commands from the repository root. Git and Python with `venv` are
required. The locally verified environment is **Python 3.14.7 on macOS ARM64**;
other operating systems and GPU configurations require their own installation
and acceptance checks.

```bash
git clone https://github.com/Jinushi-Kariyakarana-RMIT/ai-first-nations.git
cd ai-first-nations
git switch feature/mangrove-ai-enhancements
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-app.txt
```

On Windows PowerShell, use `python` instead of `python3` and activate with
`.venv\Scripts\Activate.ps1`. The remaining `python` commands are the same.
`requirements-app.txt` contains the application dependencies; the broader
`requirements.txt` also includes historical research dependencies.

### Supply the model files

A clone alone does **not** include the tree checkpoint. Obtain the evaluated
`best_tree_counter.pt` from the project maintainer through your agreed file-transfer
channel. Do not replace it with the base `yolo11n.pt` training model.

| Model | Required location | Delivery |
|---|---|---|
| Binary detector | `machine learning/flask-application/bestBinary.pth` | Tracked in Git |
| Category classifier | `machine learning/flask-application/best_mangrove_model.pth` | Tracked in Git |
| Tree counter | `machine learning/tree-counting-model/best_tree_counter.pt` | Supplied separately; ignored by Git |

Startup loads saved models; it does not train. Missing or incompatible checkpoints
disable the affected feature. A missing original `TrainingData` folder does not
prevent inference with the existing classifiers; category provenance remains a
known evaluation limitation.

### Start the application

```bash
python -m flask --app "machine learning/flask-application/app.py" run --port 5001
```

Open **http://127.0.0.1:5001** on the same computer. Keep the terminal running;
press **Ctrl+C** to stop. After pulling changes, stop and start the application again.
Check the terminal confirms that both classifiers and the tree counter loaded.

Try one JPG/PNG upload and confirm that results and a tree overlay appear without
an unexpected warning. Photos, training datasets and notebooks are not required
for application startup. Users can upload photos from any folder they can access.

## Run on your organisation's server

`127.0.0.1` is only accessible on the host computer. Giving colleagues access
requires your technical team to configure a server address and access controls.
The Flask command above is for local development, not a production deployment.

Follow the [deployment and handover guide](docs/DEPLOYMENT.md) for a WSGI server,
HTTPS/access controls, persistent configuration, storage and acceptance checks.
Public or multi-user deployment has **not** been validated by the local checks.

## Where to find things

| Location | Purpose |
|---|---|
| `machine learning/flask-application/` | Web application and classification code |
| `machine learning/tree-counting-model/` | Tree-counter implementation and checkpoint |
| `machine learning/data/testing/` | Internal evaluation images and labels |
| `machine learning/data/training/` | Training/validation images |
| `machine learning/tests/` | Automated software tests |
| `machine learning/evaluations/` | Evaluation scripts, evidence and reports |
| `machine learning/notebooks/training/` | Historical experiments |
| `machine learning/runs/` | Local generated training outputs |

See the [complete folder guide](docs/PROJECT_STRUCTURE.md). Compatibility shortcuts
preserve old dataset paths without duplicating images. Local datasets and generated
weights are not all delivered by Git.

On the original development computer, `DS Proejct/Mourilyan_Harbour_Fresh_Test_Images/`
contains standalone manual-demo photos, and `DS Proejct/AI first nation logo.png`
is the renamed original logo. Both are **outside this repository**, are not shipped
by cloning, and are not required for startup. The demo photos are excluded from
internal evaluation and annotation workflows.

## Evidence and verification

Start with the [evaluation overview](<machine learning/evaluations/notebooks/00_evaluation_overview.ipynb>).
Saved scores retain their dataset limitations; moving files or passing software
tests does not establish new predictive accuracy. Read
[current dataset availability](<machine learning/evaluations/CURRENT_DATA_LAYOUT.md>)
before rerunning evaluations. Classifier refresh is blocked until approved internal
sources are supplied; the external demo folder is not a replacement benchmark.

The latest local software run passed **32 automated tests**. Running those tests
requires evaluation dependencies beyond the application-only environment; see
[verification instructions](docs/DEPLOYMENT.md#technical-verification).
For common startup problems, see [troubleshooting](docs/DEPLOYMENT.md#troubleshooting).
