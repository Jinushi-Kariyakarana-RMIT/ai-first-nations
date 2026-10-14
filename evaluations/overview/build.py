"""Build and execute only the lightweight overview and legacy binary entry point."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluations.notebook_style import align_notebook
from pathlib import Path
import sys
import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'evaluations/overview'
NOTEBOOKS = ROOT / 'evaluations/notebooks'
setup = '''from pathlib import Path
import sys
from IPython.display import display, HTML
root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'evaluations/overview/evidence.py').is_file())
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from evaluations.overview.evidence import Evidence
evidence = Evidence(root)
def show(frame):
    display(frame.style.hide(axis='index').format(precision=4, na_rep='Not available')
            .set_properties(**{'text-align': 'left', 'white-space': 'normal'})
            .set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}]))
display(HTML('<style>.jp-RenderedHTMLCommon h1,.jp-RenderedHTMLCommon h2,.jp-RenderedHTMLCommon h3 {text-align:left}</style>'))'''

def md(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)

def save(name, cells):
    notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    align_notebook(notebook)
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv[0] = sys.executable
    NotebookClient(notebook, km=manager, timeout=120, resources={'metadata': {'path': str(ROOT)}}).execute()
    nbf.validate(notebook)
    nbf.write(notebook, NOTEBOOKS / f'{name}.ipynb')
    html, _ = HTMLExporter(exclude_input=True).from_notebook_node(notebook)
    # Notebook links remain relative to notebooks/ even in this preview.
    html = html.replace('<head>', '<head><base href="../notebooks/">', 1)
    (OUT / f'{name}.html').write_text(html)
    print(f'{name}: executed {sum(c.cell_type == "code" for c in cells)} cells, no errors')

save('00_evaluation_overview', [
md('''# Model evaluation overview

## tl;dr
The three model features are integrated into the application. Saved software checks pass, but independent field accuracy and full HD compliance are not established. This is the single entry point to the detailed evaluation evidence.

This notebook **reads saved evidence only**. It does not train models, change settings, run servers, access the private review database, or overwrite detailed evaluations. Rerunning it refreshes the presentation, not the underlying model measurements.'''),
md('''## Context & Methods
Results are grouped by inference configuration and dataset. Current-configuration tree metrics use the shared application counting path; separate HTTP checks verify integration. Historical direct-YOLO results use a different path and must not be treated as current application scores.

### Key assumptions
The saved deployment record identifies the candidate configuration as deployed. These are recorded snapshots, not a live service health check. Source hashes below identify precisely which files were read. Missing classifier accuracy is not zero accuracy.'''),
code(setup),
md('''## Data
The external benchmark contains original photographs and derived stress cases. Variants are **not additional independent photographs**. NEON crown annotations do not provide mangrove presence or category labels.

[Preparation and source lineage](../../tree-counting-model/evaluation_pack/README.md) · [Alternative independent-label workflow](../new_benchmark/README.md)'''),
code('show(evidence.benchmark())'),
md('''## Results
### Binary, category and end-to-end accuracy
The table reflects the saved classifier audit, not live annotation progress. New review labels require a separately frozen and scored evaluation before these statuses can change.'''),
code('show(evidence.classifier_audit())'),
md('''### Tree counting through the application counting path
Precision/recall are fractions; MAE and bias are trees per image. Relative absolute error is a fraction of total reference crowns. The original test has site overlap and visible annotation markers. NEON is a reused external benchmark, not an untouched mangrove holdout. Confidence 0.10 was selected on validation data.

[Full tree evaluation, experiments and rubric audit](YOLO_Tree_Counter_Evaluation.ipynb) · [Deployment record](../tree_counter_hd/DEPLOYMENT.md)'''),
code("tree = evidence.tree_results()\nshow(tree[['Configuration', 'Population', 'images', 'precision', 'recall', 'f1', 'map50']])\nshow(tree[['Configuration', 'Population', 'Confidence threshold', 'expected', 'predicted', 'mae', 'bias', 'relative_absolute_error']])"),
md('''### Software controls
These saved checks verify application behaviour. They do not measure correctness against independently labelled images. Counts belong to different test suites and are not combined into an accuracy score.'''),
code('show(evidence.controls())'),
md('''### Historical evidence — separate from current scores
The old binary notebook reports one test size while its saved dataset totals imply another. The original images cannot be recovered, so this inconsistency remains unresolved and the historical score is excluded from current accuracy claims. Its Windows training paths remain preserved in the archive; the supported current binary notebook uses repository-relative paths.

[Original binary training record](../archive/evaluate_binary_detector_historical.ipynb) · [Historical direct-YOLO results](evaluate_tree_counter_model.ipynb) · [Recovery audit](../classifier_hd/DATA_RECOVERY.md)'''),
code('show(evidence.historical_binary())'),
md('''## Takeaways
Use the application features with counts treated as estimates. Completing more software checks does not establish field accuracy. The next accuracy evaluation requires reviewed labels, independent survey separation, and verified category mapping.

### Where everything lives
| Scope | Notebook | Shared code / evidence outside the notebook |
|---|---|---|
| Binary detector | [Detailed binary evaluation](Binary_Mangrove_Evaluation.ipynb) | `classifier_hd/evaluate.py`, `metrics.py`, saved audits and predictions |
| Category classifier | [Detailed category evaluation](Mangrove_Species_Evaluation.ipynb) | Same shared classifier evaluator and metrics |
| Tree counter | [Detailed tree evaluation](YOLO_Tree_Counter_Evaluation.ipynb) | `tree_counter_hd/evaluation.py`, challenges, HTTP checks and saved metrics |
| Dataset / tiler | [Tree dataset](evaluate_tree_counter_dataset.ipynb), [image tiler](evaluate_image_tiler.ipynb) | Source manifests and software fixtures |
| Combined pipeline | [Readiness check](evaluate_combined_pipeline.ipynb) | No independently scored end-to-end dataset |
| Legacy category entry | [Readiness check](evaluate_multiclass_classifier.ipynb) | See current detailed category notebook above |
| Historical evidence | Binary archive and direct-YOLO notebook above | Preserved old outputs, not rerun by this overview |

Detailed results remain in their component notebooks. This overview collects the principal results and links to the details. Raw predictions, preparation scripts, labelling UI/database and automated tests remain outside notebooks; they are not duplicated or converted into accuracy evidence.

### Evidence provenance
Hashes identify saved inputs for this execution; they do not certify model accuracy or freshness.'''),
code('show(evidence.provenance())')])

save('evaluate_binary_detector', [
md('''# Binary detector evaluation — current entry point

The original training notebook is preserved [in the historical archive](../archive/evaluate_binary_detector_historical.ipynb). It contains unavailable Windows source paths and inconsistent saved dataset counts; do not rerun it as current evidence.

Open [the current detailed binary evaluation](Binary_Mangrove_Evaluation.ipynb) or [the project overview](00_evaluation_overview.ipynb). This compatibility entry reads saved evidence without training or changing model files.'''),
code(setup), md('## Current evidence'), code('show(evidence.classifier_audit().iloc[:1])'),
md('## Historical count discrepancy — unresolved'), code('show(evidence.historical_binary())')])
