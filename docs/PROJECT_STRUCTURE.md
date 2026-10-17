# Application folder guide

```text
ai-first-nations/
├── README.md                       Start here: features and local setup
├── requirements-app.txt            Application dependencies
├── requirements.txt                Same dependencies for standard tooling
├── docs/
│   ├── DEPLOYMENT.md                Organisation server setup and operations
│   ├── PROJECT_STRUCTURE.md         This guide
│   ├── EVALUATION_SUMMARY.md        Evidence and known limitations
│   └── model-checksums.json         Exact saved-model fingerprints
└── machine learning/
    ├── README.md                   Runtime components
    ├── flask-application/
    │   ├── *.py                    Flask, classifiers and image tiling
    │   ├── *.pth                   Two classifier checkpoints
    │   ├── templates/              Web pages
    │   ├── static/                 Styles and application logo
    │   └── uploads/                Local user photos and overlays; excluded from Git
    ├── tree-counting-model/
    │   ├── tree_counter.py         YOLO counting and duplicate removal
    │   └── best_tree_counter.pt    Included YOLO checkpoint
    └── tests/                     Application and tree-counting regression checks
```

`.venv/` is your local Python environment and is excluded from Git.

## Separate developer material

On the original development computer, these sibling folders are outside the app repository:

- `ai-first-nations-research-archive-2026-10-15/`: preserved datasets, notebooks, runs, evaluation reports, research tests and matching source snapshot. It includes the latest manual trial. Not needed by app users.
- `Mourilyan_Harbour_Fresh_Test_Images/`: original drone photos for manual demonstrations. Not an internal labelled benchmark.
- `AI first nation logo.png`: original logo; the application already includes its display asset.

The archive is local, not a remote backup. Historical research files remain in Git history; this cleanup simplifies the current checkout without rewriting history.
