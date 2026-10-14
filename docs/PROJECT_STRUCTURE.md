# Current project structure

Updated 14 October 2026. This is the current layout; older move/check records in
`docs/` describe the state at the time they were written.

```text
DS Proejct/
├── AI first nation logo.png              Renamed original logo image
├── Mourilyan_Harbour_Fresh_Test_Images/   527 photos for manual application trials
└── ai-first-nations/                     Git repository
    ├── README.md                        Project introduction and launch instructions
    ├── requirements-app.txt             Pinned application dependencies
    ├── requirements.txt                 Broader historical research dependencies
    ├── docs/                            Architecture, changes and verification records
    └── machine learning/
        ├── README.md                    Workspace guide
        ├── data/
        │   ├── testing/
        │   │   ├── tree_counter/        Original labelled tree test split
        │   │   ├── neon_benchmark/      External labelled benchmark and stress variants
        │   │   ├── synthetic_trees/     Synthetic software-check fixtures
        │   │   ├── tiler_fixtures/      Image-splitting fixtures
        │   │   ├── tree_challenges/     Composite/negative diagnostic inputs
        │   │   └── marker_cleanup/      Inpainted diagnostic inputs
        │   ├── training/tree_counter/   Original train/validation splits and YOLO config
        │   └── legacy/                  Preserved unused raster inputs
        ├── tests/                       Automated software/metric test code
        ├── runs/detect/                 Generated training checkpoints and logs
        ├── notebooks/training/          Historical training experiments
        ├── evaluations/
        │   ├── notebooks/               Detailed reports and overview notebook
        │   ├── overview/                Summary loader, builder and HTML exports
        │   ├── classifier_hd/           Saved classifier evidence and evaluation code
        │   ├── tree_counter_hd/         Saved tree evidence and evaluation code
        │   ├── new_benchmark/           Internal label-review and benchmark workflow
        │   ├── archive/                 Historical evaluation notebook
        │   └── specifications/          Rubric PDFs
        ├── tree-counting-model/         Tree-counter code and deployed tree checkpoint
        ├── flask-application/
        │   ├── templates/               HTML pages
        │   ├── static/                  Styles and existing application assets
        │   └── uploads/                 User uploads and generated overlays
        └── yolo11n.pt                   Local base checkpoint for training
```

**Mourilyan photos and the renamed logo are outside Git.** The photos are for
manually uploading into the application, not internal testing, training or
annotation candidates. There is no internal shortcut to that photo folder.
The logo rename does not replace or modify existing web assets.

The remaining dataset shortcuts are intentional, contain no duplicate content,
and preserve old notebooks/configurations:

| Shortcut inside `machine learning/` | Actual data |
|---|---|
| `tree-counting-model/dataset1` | `data/training/tree_counter` |
| `data/training/tree_counter/test` | `data/testing/tree_counter` |
| `tree-counting-model/evaluation_pack` | `data/testing/neon_benchmark` |
| `tree-counting-model/mock_dataset` | `data/testing/synthetic_trees` |
| `notebooks/test_data` | `data/testing/tiler_fixtures` |
| `evaluations/tree_counter_hd/challenge_images` | `data/testing/tree_challenges` |
| Individual marker-diagnostic PNG links | `data/testing/marker_cleanup` |

Historical inventories, notebook outputs and private review records are retained
as evidence. References to the old 527-photo pool in those snapshots are historical,
not an active dataset. The review tool excludes unavailable historical images.
Classifier evidence refresh requires approved internal classifier sources; it
stops before writing results when those sources are absent. Do not put the demo
folder back into that workflow merely to make a refresh pass.

Runtime: upload → validation → binary detection → category classification when
positive; independently, tree detection → duplicate suppression → crown count and
overlay. Startup loads saved models and does not train.

For business users and local startup, read the root README. Technical server
configuration and handover checks are in [DEPLOYMENT.md](DEPLOYMENT.md).
