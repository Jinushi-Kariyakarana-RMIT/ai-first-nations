"""Build two evidence notebooks; no training and no fabricated labelled results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluations.notebook_style import align_notebook
from evaluations.layout_notice import apply_layout_notice
from pathlib import Path
import json, shutil
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent

criteria=[
('Research questions and evaluation design',5,'Implemented','Scope, independent evaluation unit, targets and protocol are documented.'),
('Dataset audit and annotation quality',10,'Partial','527-source inventory and duplicate checks; independent classifier annotations missing.'),
('Leakage prevention and splits',15,'Blocked','Original training membership unavailable; source/site-separated labelled holdout not verified.'),
('Reproducibility and model setup',10,'Partial','Exact weights, settings, versions and source hashes recorded; historical species class mapping remains unverified.'),
('Classification metrics and calibration',20,'Blocked','Tested metric implementation exists, but no independent labelled accuracy, calibration or curves can be calculated.'),
('Robustness, confounding and failure analysis',15,'Partial','Real-weight brightness/blur/channel/edge diagnostics; no labelled site/condition performance.'),
('Controlled improvement experiments',10,'Blocked','Two comparisons specified per model; no labelled validation data for fair selection or final testing.'),
('Flask integration and operational tests',10,'Implemented with limits','Actual HTTP parity, gate/failure tests, warm timing and process memory; limited hardware/repetitions.'),
('Notebook clarity and conclusions',5,'Implemented','Ten-section executed notebooks, source-backed tables, figures and explicit incomplete HD status.')]
import pandas as pd
pd.DataFrame(criteria,columns=['criterion','marks','status','evidence_or_gap']).to_csv(OUT/'specification_audit.csv',index=False)
legacy=dict(binary=dict(recorded_accuracy=.9896,recorded_macro_f1=.9889,recorded_test_tiles=5286,
                        saved_tile_totals=[19862,21770],test_tiles_implied_by_saved_totals=6246,
                        status='Historical outputs only; inconsistent counts, tile-level split and old absolute Windows paths. Not rerun as current accuracy.'),
            species=dict(status='Previous notebook checked checkpoint presence/hash only; no classification metrics.'),
            shared_transform_repair='Training subset and validation subset now use separate dataset copies. Existing checkpoint training history cannot be repaired retrospectively.',
            gating_repair='Category inference is skipped when the binary model is unavailable. Saved weights and normal positive-path predictions unchanged.')
(OUT/'previous_evaluation_audit.json').write_text(json.dumps(legacy,indent=2)+'\n')
for stem in ['Binary_Detection','Species_Classification']:
 source=Path('/Users/joseph/Downloads')/f'AI_First_Nations_{stem}_HD_Evaluation_Specification.pdf'
 destination=ROOT/'evaluations/specifications'/source.name
 if source.exists():shutil.copy2(source,destination)

for mode,title,filename in [('binary','Binary Mangrove Evaluation','Binary_Mangrove_Evaluation.ipynb'),('species','Mangrove Category (Species) Evaluation','Mangrove_Species_Evaluation.ipynb')]:
 cells=[]
 def md(text):cells.append(nbf.v4.new_markdown_cell(text.strip()))
 def code(text):cells.append(nbf.v4.new_code_cell(text.strip()))
 md(f'''# {title}

**Decision: technical verification completed; independent predictive evaluation blocked. Full HD compliance is not achieved.**

This notebook follows the supplied proposed rubric, which is not an official RMIT marking scheme. It evaluates the exact saved EfficientNet-B0 checkpoint and application path. Missing reference labels are never replaced with model predictions or synthetic accuracy scores. The other classifier has its own companion notebook.

The application remains available. Technical availability and scientifically verified field accuracy are separate conclusions.
''')
 code(f'''
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from bs4 import BeautifulSoup
from IPython.display import display as _display, HTML, Markdown, Image as DisplayImage
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'flask-application').exists())
OUT=ROOT/'evaluations/classifier_hd'
MODE={mode!r}
def read(name): return json.loads((OUT/name).read_text())
def display(*objects):
    for obj in objects:
        if isinstance(obj,pd.DataFrame):
            soup=BeautifulSoup(obj.to_html(index=False,justify='left'),'html.parser')
            for element in soup.select('th,td'):element['style']='text-align: left;'
            _display(HTML(str(soup)))
        else:_display(obj)
audit=read('data_audit.json');environment=read('environment.json');application=read('application_summary.json')
inventory=pd.read_csv(OUT/'source_inventory.csv');predictions=pd.DataFrame(read('unlabelled_predictions.json'))
checks=pd.read_csv(OUT/'application_checks.csv');checks=checks[checks.model.isin([MODE,'both'])]
plt.rcParams.update({{'figure.figsize':(10,4),'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titlelocation':'left','figure.dpi':110}})
display(pd.DataFrame([{{'Ground-truth images':audit['labelled_'+MODE+'_rows'], 'Application checks passed':int((checks.status=='PASS').sum()),'Application checks total':len(checks),'HD compliance':'Not met — missing independent labelled evaluation'}}]))
''')
 md('''## 1. Research design

Questions: Does the saved model work on independent source images? Do errors vary by site and conditions? Are confidence scores calibrated? Does tile averaging give reliable source-image decisions? Does Flask reproduce the exact offline probabilities?

The predictive evaluation unit is an **independent source image**, with separate tile diagnostics. Split sources/sites before tiling and freeze the final test set. Image siblings and adjacent frames must not cross partitions. Proposed rubric targets are not agreed production acceptance limits.

**Reproduce:** use the project `.venv` and run `.venv/bin/python evaluations/classifier_hd/run_notebooks.py --refresh`. This reruns saved-model diagnostics and actual HTTP checks once, then executes both notebooks. Without `--refresh`, saved evidence is validated against code/weight hashes before notebook execution. Temporary test fixtures are deleted automatically; the user's running app is untouched. Test server port 5053 must be free.
''')
 code('''protocol=read('experiment_protocol.json')
display(pd.DataFrame([protocol['targets_'+MODE]]))
''')
 md('''## 2. Dataset audit

The inventory contains 527 fresh Mourilyan source photographs. Folder names and filenames identify candidate route/date groups; they do not establish independent training exclusion. Both classifier label columns are blank. NEON tree-crown boxes are not mangrove-presence or category annotations and cannot be reused as classifier ground truth.

The label collection sheet is `label_manifest.csv`. It preserves source ID/path, site/survey group, image hash, intended split, reviewer, annotation provenance and training-exclusion confirmation. Reviewers must provide labels; the evaluation never guesses them.
''')
 code('''display(pd.DataFrame([{'Source images':len(inventory),'Binary labels':audit['labelled_binary_rows'],'Category labels':audit['labelled_species_rows'],'Exact duplicate excess':audit['exact_duplicate_excess'],'Near-duplicate candidate pairs':audit['near_duplicate_candidate_pairs']}]))
counts=inventory.groupby('survey_id').size().sort_values()
fig,ax=plt.subplots(figsize=(10,3.5));ax.barh([str(x).rsplit('-',1)[-1].replace('Route','Route ') for x in counts.index],counts.values,color='#246b63')
ax.set(xlabel='Original images (not independent test samples)',title='Available source images by folder group');ax.set_xlim(0,counts.max()*1.2)
for i,value in enumerate(counts.values):ax.text(value+2,i,str(value),va='center')
fig.tight_layout();plt.show()
display(inventory[['source_image_id','survey_id','width','height']].head(5))
''')
 md('''**Interpretation:** source coverage is documented, but folder counts do not measure label balance or model accuracy. Per-class support is unavailable. Review the 13 perceptual-similarity candidates in `near_duplicate_candidates.csv`; similarity is not proof of duplication.
''')
 code('''chosen=inventory.iloc[[0,len(inventory)//2,len(inventory)-1]]
fig,axes=plt.subplots(1,3,figsize=(12,3.5))
for ax,(_,row) in zip(axes,chosen.iterrows()):
    with Image.open(ROOT/row.image_path) as image:
        image.thumbnail((600,450));ax.imshow(image)
    ax.set_title(str(row.survey_id).rsplit('-',1)[-1].replace('Route','Route '),fontsize=11);ax.axis('off')
fig.suptitle('Source previews — unlabelled, not scored test examples',x=.01,ha='left');fig.tight_layout();plt.show()
''')
 md('''## 3. Split integrity

**Not met.** No original training-source registry is available to prove training exclusion. No independent labelled validation/test split is established. Exact SHA-256 and 64-bit difference hashes are recorded for the fresh inventory; near-duplicate candidates use Hamming distance ≤6. Geographical overlap and temporally adjacent frames still require survey metadata and human review.

The legacy binary notebook tiled first and randomly split tiles. Its saved tile totals (41,632) imply 6,246 test tiles under its split code, whereas the saved report lists 5,286. Those outputs remain historical, not a reproducible independent baseline. The classifier readiness notebook contains no scored test set.
''')
 code('''legacy=read('previous_evaluation_audit.json')
display(pd.DataFrame([{'Item':key,'Evidence':str(value)} for key,value in legacy[MODE].items()]))
display(pd.read_csv(OUT/'near_duplicate_candidates.csv').head(5))
''')
 md('''## 4. Model reproducibility

Both models use EfficientNet-B0 with a 128-unit hidden classifier head. Inference converts to RGB, tiles at 512×512 with 64-pixel overlap, resizes each tile to 224×224 and uses ImageNet normalization. Tile softmax probabilities are averaged at source-image level. Batch size is 64; seed is 42. Exact checkpoint hashes, code hashes and package versions are saved.

Binary class order agrees with the historical code. **Species class order is the serving fallback `orange, red, yellow`, not a verified training-label mapping.** A state dictionary alone cannot prove semantic label order. These names are dataset categories, not confirmed biological species.
''')
 code('''model=environment[MODE]
display(pd.DataFrame([{'Item':'Model SHA-256','Value':model['sha256']},{'Item':'Class order','Value':str(model['labels'])},{'Item':'Device','Value':environment['device']},{'Item':'Head','Value':model['architecture']},{'Item':'Transforms','Value':environment['transforms']}]))
display(pd.DataFrame([{'Package':k,'Version':v} for k,v in environment['versions'].items()]))
''')
 md('''## 5. Core predictive metrics and calibration

**Blocked: independent labels are missing.** Accuracy, confusion matrices, class precision/recall/F1, balanced accuracy, ROC/PR curves and calibration cannot be computed defensibly from these unlabelled images.

`metrics.py` implements all core classification metrics. Undefined metrics are JSON `null`, not zero. Binary Mangrove is positive. PR area uses average precision (step integral). Binary Brier uses Mangrove probability; multiclass Brier sums squared errors across the three classes (range 0–2). ECE uses ten equal-width bins of top-class confidence. A class with no positives/negatives has undefined ROC/PR area. These definitions and known-answer tests are software evidence, not model scores.

No ROC curve, reliability diagram or confusion matrix is fabricated. After independent labels exist, report source-level metrics separately from tile metrics, include support and confidence intervals resampling entire independent sources/sites, and plot correct versus incorrect confidence.
''')
 code('''metric_names=['Independent test images','Overall accuracy','Macro F1','Balanced accuracy','Per-class precision / recall / F1','ROC-AUC / average precision','Brier / ECE / log loss','Full-image accuracy','Held-out-site performance']
display(pd.DataFrame({'Metric':metric_names,'Current result':['Not measured — required labels and independence evidence missing']*len(metric_names)}))
validation=read('verification.json');display(pd.DataFrame([validation]))
''')
 md('''## 6. Tile-to-image analysis

For each technical fixture, collect all tile probabilities and coordinates, calculate their arithmetic mean independently and compare it with the deployed source-level function. Record lineage in `tile_predictions.json`. The input suite includes three source photos with reference, darker and blurred variants, small/edge-sized images, grayscale, RGBA and one full-resolution source.

These are 14 technical cases derived from only three source images, not 14 independent accuracy samples. Species is run directly on every diagnostic input to inspect software behaviour; only eligible binary-positive images are classified through the real application.
''')
 code('''frame=predictions[predictions.model==MODE].copy()
display(frame[['id','variant','tiles','label','confidence','tile_vote_agreement']].round(4))
display(pd.read_csv(OUT/'tiler_coverage.csv'))
fig,ax=plt.subplots(figsize=(10,4));ax.barh(frame.id,frame.tile_vote_agreement,color='#246b63');ax.set(xlim=(0,1),xlabel='Fraction of tile labels agreeing with aggregated label',title='Tile agreement is consistency, not accuracy');fig.tight_layout();plt.show()
''')
 md('''**Interpretation:** agreement with source aggregation verifies implementation consistency. A high agreement fraction can still be consistently wrong. Zero uncovered pixels demonstrates tiler coverage, not reliable detection of small mangrove patches within mixed scenes.
''')
 md('''## 7. Robustness and failure analysis

Measure probability/label sensitivity to darker and blurred versions paired with their reference source. Brightness is multiplied by 0.55; blur uses a 3-pixel Gaussian radius on 1024×768 derivatives. These are artificial conditions, not labelled field subgroups. No accuracy, false-positive rate, minority-class recall or cross-site performance is inferred from them.

Real water/mud/other-vegetation negatives and reviewed Orange/Red/Yellow examples remain required. Unsupported and mixed-category scenes cannot be validated from a single model-assigned image label.
''')
 code('''paired=frame[frame.id.str.startswith('source_')].copy();paired['source_group']=paired.id.str.extract(r'(source_\\d+)')
fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True)
for ax,(source,group) in zip(axes,paired.groupby('source_group')):
    ax.bar(group.variant,group.confidence,color=['#246b63','#b37635','#7a8791'])
    ax.set(ylim=(0,1),title=source.replace('_',' '),xlabel='Image condition')
axes[0].set_ylabel('Top-class model confidence');fig.tight_layout();plt.show()
display(paired[['source_group','variant','label','confidence']].round(4))
''')
 md('''**Interpretation:** confidence changes and label flips describe sensitivity only. Without ground truth they cannot establish whether a transformation improves or damages accuracy. Model confidence is not a calibrated probability of correctness.
''')
 md('''## 8. Controlled improvements

The two predictive comparisons below are registered as a plan, **not completed experiments**. Both need independent labelled validation data; the final test set must remain untouched. No threshold, calibration temperature, aggregation or checkpoint was selected using these unlabelled cases.

Two verified engineering repairs were completed separately: training and validation now have independent transform objects, and missing binary detection no longer permits category inference. These changes cannot repair the provenance or training of the inherited checkpoint and do not establish an accuracy improvement.
''')
 code('''display(pd.DataFrame({'Planned comparison':protocol[MODE+'_experiments'],'Status':['Blocked — labelled validation/test set required']*2}))
display(pd.DataFrame(read('repairs.json'))[['file','issue']])
''')
 md('''## 9. Flask end-to-end verification

Real localhost HTTP requests follow upload → image validation → shared tiler → saved classifiers → source aggregation → template context → rendered results. Compare labels and probabilities with independent direct calls at absolute tolerance 1e-6. Tree counting stays enabled for normal HTTP requests, so measured latency/RSS describes the complete application, not classifier-only performance.

Controlled branch tests force upstream decisions to exercise rejection, confidence boundaries, unavailable models and exceptions; these are explicitly identified separately from real unmodified-model predictions. Missing checkpoints are simulated using isolated paths without moving user weights. Temporary fixtures, uploads and browser profiles are deleted on exit.
''')
 code('''display(checks[['test','status','kind','expected','observed']])
timing=pd.read_csv(OUT/'latency_memory.csv')
display(timing.round(3))
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
grouped=timing.groupby('id').agg(seconds=('seconds','median'),memory=('peak_process_rss_mb','max'),n=('seconds','size')).sort_values('seconds')
labels=[f'{name} (n={int(row.n)})' for name,row in grouped.iterrows()]
axes[0].barh(labels,grouped.seconds,color='#246b63');axes[0].set(xlabel='Median warm HTTP seconds',title='Full application response time')
axes[1].barh(labels,grouped.memory,color='#7a8791');axes[1].set(xlabel='Maximum sampled process RSS (MiB)',title='All models and allocator cache included')
fig.tight_layout();plt.show()
if (OUT/'flask_result.png').exists():display(DisplayImage(filename=str(OUT/'flask_result.png'),width=1000))
''')
 md('''**Interpretation:** these checks verify the tested code paths, not prediction accuracy. There are three warm repetitions for the reference and one for each other case. Cold model loading is measured once separately. RSS samples at 10 ms may miss short peaks; no speed/memory service-level target is agreed. Historical species mapping remains a separate semantic blocker even when numeric Flask parity passes.
''')
 md('''## 10. Conclusions and readiness

**Functionality: available, with the recorded software checks. Full HD evaluation: not yet met.** The old binary notebook had a potentially leaking tile split and inconsistent saved counts; the old category notebook was only a readiness check. This notebook adds current-weight reproducibility, source inventory, duplicate screening, real prediction parity, probability sensitivity and measured application behaviour.

It does **not** add independent accuracy evidence: reviewed labels, original training membership and verified category mapping are still missing. Complete the source/site-separated dataset first, then execute the planned comparisons, calibration, full-image/site metrics and uncertainty analysis. Do not describe the historical 98.96% binary score or an inherited approximately 97% category claim as fresh-survey accuracy.

All metric targets below remain unassessed, not failed zero scores. A rigorous evaluation may report poor model accuracy and still meet an academic standard, but missing non-negotiable evidence cannot be replaced with more software checks.
''')
 code('''display(pd.read_csv(OUT/'specification_audit.csv'))
keys=['Source-level precision','Source-level recall','Macro F1','Balanced accuracy','Calibration','Held-out site','Flask numeric parity']
display(pd.DataFrame({'Criterion':keys,'Historical baseline':['Not independently verified']*6+['No equivalent classifier-specific evidence'], 'Current':['Blocked — independent labels required']*6+[f"{int((checks.status=='PASS').sum())}/{len(checks)} applicable checks passed"], 'Target':['See supplied rubric']*6+['All critical checks pass']}))
''')
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'AI First Nations (.venv)','language':'python'}})
 align_notebook(nb); apply_layout_notice(nb)
 nbf.validate(nb);nbf.write(nb,ROOT/'evaluations/notebooks'/filename)
 print(filename)
