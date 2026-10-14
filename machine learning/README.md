# Machine learning workspace

All application and machine-learning files are grouped here, preserving their
relative layout so model loading, datasets, review records and evaluations agree.

| Folder | Contents |
|---|---|
| `flask-application/` | Web application, classifier inference and saved classifier weights |
| `tree-counting-model/` | Tree model, checkpoint, training dataset and external benchmark |
| `evaluations/` | Evaluation notebooks, shared evaluators and saved results |
| `notebooks/` | Historical training experiments and tiler fixtures |
| `tests/` | Automated pipeline and metric tests |
| `data/` | Preserved local legacy imagery |
| `Mourilyan_Harbour_Fresh_Test_Images/` | Local survey images |
| `runs/` | Local training history |
| `yolo11n.pt` | Local base checkpoint for training |

From the repository root:

```bash
.venv/bin/python -m flask --app "machine learning/flask-application/app.py" run --port 5001
.venv/bin/python -m unittest discover -s "machine learning/tests" -v
.venv/bin/python "machine learning/evaluations/overview/build.py"
```

For older component README commands, change into this folder first and use
`../.venv/bin/python` instead of `.venv/bin/python`. Open evaluation notebooks with
this folder as the working directory. Historic output strings retain the original
paths; regenerate measured evidence with the appropriate runner when needed.

Stop/restart previously running application, notebook and review servers using
the new paths; a process started before the move may still retain old paths.
