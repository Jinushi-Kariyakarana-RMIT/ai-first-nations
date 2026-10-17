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
Both requirements files install the same application dependencies. Research dependencies are kept in the separate research archive.

### Model files

All three saved checkpoints are included in this application handover:

| Model | Location |
|---|---|
| Binary detector | `machine learning/flask-application/bestBinary.pth` |
| Category classifier | `machine learning/flask-application/best_mangrove_model.pth` |
| Tree counter | `machine learning/tree-counting-model/best_tree_counter.pt` |

Startup loads these files without training. Their SHA-256 fingerprints are in
[model-checksums.json](docs/model-checksums.json). Missing or incompatible files disable the affected feature.

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
| `machine learning/flask-application/` | Web pages, application code and classification models |
| `machine learning/tree-counting-model/` | Tree counting code and saved model |
| `machine learning/tests/` | Small application regression tests; no research datasets needed |
| `docs/` | Deployment, folder guide, model fingerprints and evaluation summary |

See the [folder guide](docs/PROJECT_STRUCTURE.md) and [evaluation summary](docs/EVALUATION_SUMMARY.md).
Research datasets, notebooks, training runs and historical reports are preserved
in a separate local archive. They are not needed to run the application.
The Mourilyan demo photos remain outside the repository; users can upload their own images.

## Verify the application

```bash
python -m unittest discover -s "machine learning/tests" -v
```

These are software checks, not measurements of model accuracy. See
[troubleshooting](docs/DEPLOYMENT.md#troubleshooting) for setup issues.
