"""Reproducible tree-counter evidence using the unchanged application functions."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

os.environ['YOLO_AUTOINSTALL'] = 'false'

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'tree-counting-model'))
from tree_counter import DEFAULT_MODEL_PATH, _nms, _to_global_boxes, count_trees
from image_tiler import tile_image
from evaluate_prepared_pack import RecordingModel, match_counts

PACK = PROJECT / 'tree-counting-model/evaluation_pack'
OUT = PROJECT / 'evaluations/tree_counter_hd'
BASELINE = dict(confidence=.15, iou=.45, tile_size=640, overlap=64)


def save_json(name, data):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_boxes(path, width, height):
    boxes = []
    for line in Path(path).read_text().splitlines():
        cls, x, y, w, h = map(float, line.split())
        assert cls == 0 and w > 0 and h > 0
        box = [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]
        assert box[0] >= -1e-5 and box[1] >= -1e-5 and box[2] <= width+1e-5 and box[3] <= height+1e-5
        boxes.append(box)
    return boxes


def samples():
    groups = {}
    for split in ['train', 'valid', 'test']:
        rows = []
        for path in sorted((PROJECT / 'tree-counting-model/dataset1' / split / 'images').glob('*')):
            if not path.is_file():
                continue
            w, h = Image.open(path).size
            label = path.parent.parent / 'labels' / (path.stem + '.txt')
            rows.append(dict(id=path.stem, image=str(path.relative_to(PROJECT)), width=w, height=h,
                             site=path.stem.split('_')[0], split=split,
                             boxes=read_boxes(label,w,h), label=str(label.relative_to(PROJECT))))
        groups[split] = rows
    manifest = pd.read_csv(PACK / 'manifest.csv')
    groups['external'] = []
    for row in manifest[manifest.variant == 'original'].to_dict('records'):
        groups['external'].append(dict(id=row['parent_id'], image=str((PACK/row['image']).relative_to(PROJECT)),
                                      width=row['width'], height=row['height'], site=row['site'], split='external',
                                      boxes=read_boxes(PACK/row['label'],row['width'],row['height']),
                                      label=str((PACK/row['label']).relative_to(PROJECT))))
    return groups


def setup():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.use_deterministic_algorithms(True, warn_only=True)
    OUT.mkdir(exist_ok=True)
    history = PROJECT / 'runs/detect/tree_counter_baseline_30'
    for name in ['args.yaml','results.csv']:
        if (history/name).exists():
            (OUT/'training_history').mkdir(exist_ok=True)
            shutil.copy2(history/name, OUT/'training_history'/name)
    packages = {name: version(name) for name in ['torch','torchvision','ultralytics','numpy','Pillow','pandas',
                                               'matplotlib','pycocotools','Flask','psutil','nbformat','nbclient','nbconvert',
                                               'pi-heif','requests','beautifulsoup4','ipykernel',
                                               'opencv-python','PyYAML','ipython','jupyter-client']}
    environment = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
                       platform=platform.platform(), cpu=platform.processor(), machine=platform.machine(),
                       hardware=subprocess.check_output(['system_profiler','SPHardwareDataType'],text=True).split('Serial Number')[0],
                       model_sha256=digest(DEFAULT_MODEL_PATH), seed=42, device='cpu', packages=packages,
                       application_settings=BASELINE, ap_confidence_floor=.001,
                       evaluation_code_sha256={p.name:digest(p) for p in OUT.glob('*.py')},
                       ap_max_detections_per_image=300, source_manifest_sha256=digest(PACK/'manifest.csv'),
                       spec_sha256=digest(PROJECT/'evaluations/specifications/AI_First_Nations_YOLO_Tree_Counter_HD_Evaluation_Specification.pdf'))
    save_json('environment.json',environment)
    (OUT/'requirements-lock.txt').write_text('\n'.join(f'{k}=={v}' for k,v in packages.items())+'\n')
    groups = samples()
    rows = []
    for group, data in groups.items():
        for s in data:
            rows.append({k:v for k,v in s.items() if k != 'boxes'} | dict(crowns=len(s['boxes']),
                         image_sha256=digest(PROJECT/s['image']), label_sha256=digest(PROJECT/s['label'])))
    pd.DataFrame(rows).to_csv(OUT/'split_manifest.csv',index=False)
    prefixes={k:sorted({s['site'] for s in data}) for k,data in groups.items()}
    overlaps = dict(train_valid=sorted(set(prefixes['train']) & set(prefixes['valid'])),
                    train_test=sorted(set(prefixes['train']) & set(prefixes['test'])))
    audit = dict(site_prefixes=prefixes, original_site_overlap=overlaps,
                 original_split_claim='Within-source historical split; NOT site-independent.',
                 external_claim='Separate published NEON source. Previously inspected benchmark; NOT a new blind final test.',
                 survey_totals='Not defensible: adjacent photographs may overlap; site sums are image totals, not a tree inventory.',
                 previous_pack_audit=json.loads((PACK/'quality_audit.json').read_text()))
    save_json('split_audit.json',audit)
    return groups, environment


def predict(model, sample, config):
    recorder = RecordingModel(model)
    path = PROJECT/sample['image']
    result = count_trees(path, model=recorder, **config)
    _, positions, _ = tile_image(path, tile_size=config['tile_size'], overlap=config['overlap'],pad=True)
    raw = []
    tiles = []
    for tile_index, (prediction, position) in enumerate(zip(recorder.results,positions)):
        boxes = _to_global_boxes(prediction, position)
        raw.extend(boxes)
        left,top,right,bottom=position
        # Per-tile truth: crown centre lies inside valid tile. Tiles overlap;
        # these counts are diagnostics and must never be summed as survey totals.
        truth=[b for b in sample['boxes'] if left <= (b[0]+b[2])/2 < right and top <= (b[1]+b[3])/2 < bottom]
        tiles.append(dict(image_id=sample['id'], tile=tile_index, position=position,
                          expected_by_centre=len(truth), predicted=len(boxes)))
    boxes = _nms(raw, iou_threshold=config['iou'])
    assert len(boxes)==result['estimated_trees']
    return dict(id=sample['id'], boxes=boxes, raw_boxes=raw, tiles=tiles,
                raw_count=len(raw), predicted=len(boxes), expected=len(sample['boxes']),
                duplicate_count=len(raw)-len(boxes), site=sample['site'], image=sample['image'])


def operational_metrics(data, predictions):
    rows=[]
    for s,p in zip(data,predictions):
        assert s['id']==p['id']
        tp,fp,fn=match_counts(p['boxes'],s['boxes'])
        rows.append(dict(id=s['id'],site=s['site'],image=s['image'],expected=len(s['boxes']),predicted=len(p['boxes']),
                         error=len(p['boxes'])-len(s['boxes']),tp=tp,fp=fp,fn=fn,
                         raw_count=p['raw_count'],duplicate_count=p['duplicate_count']))
    f=pd.DataFrame(rows); e=f.error.to_numpy();tp=int(f.tp.sum());fp=int(f.fp.sum());fn=int(f.fn.sum())
    total=int(f.expected.sum())
    metrics=dict(images=len(rows),expected=total,predicted=int(f.predicted.sum()),tp=tp,fp=fp,fn=fn,
                 precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                 f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                 mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.mean(e**2))),bias=float(e.mean()),
                 relative_absolute_error=float(np.abs(e).sum()/total) if total else None,
                 zero_tree_images=int((f.expected==0).sum()),false_positives_on_empty=int(f.loc[f.expected==0,'predicted'].sum()),
                 duplicate_count=int(f.duplicate_count.sum()))
    return metrics,rows


def coco_metrics(data, predictions):
    """Official COCO evaluator, 101 recall points, 10 IoUs, maxDets=300."""
    dataset=dict(images=[],annotations=[],categories=[dict(id=1,name='tree')],info={})
    detections=[]; annotation_id=1
    for image_id,(s,p) in enumerate(zip(data,predictions),1):
        dataset['images'].append(dict(id=image_id,width=s['width'],height=s['height']))
        for x1,y1,x2,y2 in s['boxes']:
            dataset['annotations'].append(dict(id=annotation_id,image_id=image_id,category_id=1,
                                               bbox=[x1,y1,x2-x1,y2-y1],area=(x2-x1)*(y2-y1),iscrowd=0))
            annotation_id+=1
        for x1,y1,x2,y2,score in p['boxes']:
            detections.append(dict(image_id=image_id,category_id=1,bbox=[x1,y1,x2-x1,y2-y1],score=score))
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO();gt.dataset=dataset;gt.createIndex()
        if detections:
            dt=gt.loadRes(detections)
        else:
            dt=COCO();dt.dataset=dict(images=dataset['images'],annotations=[],categories=dataset['categories']);dt.createIndex()
        ev=COCOeval(gt,dt,'bbox');ev.params.maxDets=[1,10,300];ev.evaluate();ev.accumulate()
    precision=ev.eval['precision'][:,:,0,0,-1]
    valid=precision[precision>=0];p50=precision[0];p50=p50[p50>=0]
    return dict(map50=float(p50.mean()) if len(p50) else None,
                map50_95=float(valid.mean()) if len(valid) else None,
                recall_axis=ev.params.recThrs.tolist(),
                precision_curve_iou50=[float(x) if x>=0 else None for x in precision[0]],
                ious=ev.params.iouThrs.tolist(),max_detections=300,confidence_floor=.001)


def validation_experiments(model, valid):
    # Register candidates and decision rule before inspecting the new runs.
    configs=[BASELINE | dict(confidence=round(t,2)) for t in [.1,.15,.2,.3,.4,.5,.6,.7,.8,.9]]
    configs += [BASELINE | dict(iou=iou) for iou in [.3,.6,.7]]
    save_json('experiment_protocol.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),
              candidates=configs,selection='Minimum validation count MAE; tie prefers baseline, then closest settings.',
              validation_ids=[s['id'] for s in valid],
              prohibition='Do not select settings using original-test, external-test, stress, or fresh-survey outputs.'))
    rows=[]
    for config in configs:
        predictions=[predict(model,s,config) for s in valid]
        metric,_=operational_metrics(valid,predictions)
        rows.append(config | metric)
    ranked=sorted(rows,key=lambda r:(r['mae'],abs(r['confidence']-.15)+abs(r['iou']-.45)))
    selected={k:ranked[0][k] for k in BASELINE}
    pd.DataFrame(rows).to_csv(OUT/'validation_experiments.csv',index=False)
    save_json('locked_settings.json',dict(locked_utc=datetime.now(timezone.utc).isoformat(),
              baseline=BASELINE,candidate=selected,validation_mae=ranked[0]['mae'],
              model_sha256=digest(DEFAULT_MODEL_PATH),
              selected_using='18 original validation images only; original site-overlap caveat applies.',
              deployment='Candidate is evaluation-only. Production Flask settings remain unchanged.'))
    return rows,selected


def evaluate_dataset(model, data, name, config):
    operating=[];low=[]
    for i,s in enumerate(data,1):
        operating.append(predict(model,s,config))
        low.append(predict(model,s,config | dict(confidence=.001)))
        if i%50==0:print(f'{name}: {i}/{len(data)}',flush=True)
    metrics,rows=operational_metrics(data,operating)
    ap=coco_metrics(data,low)
    metrics.update(map50=ap['map50'],map50_95=ap['map50_95'])
    save_json(f'{name}/metrics.json',metrics)
    save_json(f'{name}/predictions.json',operating)
    save_json(f'{name}/ap_predictions.json',low)
    save_json(f'{name}/precision_recall.json',ap)
    pd.DataFrame(rows).to_csv(OUT/name/'per_image.csv',index=False)
    pd.DataFrame([t for p in operating for t in p['tiles']]).to_csv(OUT/name/'per_tile.csv',index=False)
    site=pd.DataFrame(rows).groupby('site').agg(images=('id','size'),image_crown_total=('expected','sum'),
                                             predicted_image_total=('predicted','sum'),bias=('error','mean'))
    site.to_csv(OUT/name/'site_image_totals_not_survey_counts.csv')
    return metrics,operating


def tiler_checks():
    rows=[]
    for size in [(200,180),(640,640),(641,641),(1200,800),(4032,3024)]:
        for overlap in [0,64,128]:
            tiles,pos,_=tile_image(Image.new('RGB',size),tile_size=640,overlap=overlap)
            covered=np.zeros((size[1],size[0]),dtype=bool)
            for l,t,r,b in pos:covered[t:b,l:r]=True
            rows.append(dict(size=str(size),overlap=overlap,tiles=len(tiles),
                             uncovered_pixels=int((~covered).sum()),fixed_640=all(t.size==(640,640) for t in tiles)))
    assert all(r['uncovered_pixels']==0 and r['fixed_640'] for r in rows)
    pd.DataFrame(rows).to_csv(OUT/'tiler_coverage.csv',index=False)
    return rows


def run():
    from ultralytics import YOLO
    groups,env=setup()
    model=YOLO(str(DEFAULT_MODEL_PATH));model.to('cpu')
    print('Validation-only experiments',flush=True)
    experiments,selected=validation_experiments(model,groups['valid'])
    print('Locked settings:',selected,flush=True)
    results={}
    for name,config in [('baseline',BASELINE),('candidate',selected)]:
        for split in ['test','external']:
            key=name+'_'+split
            print('Evaluating',key,flush=True)
            results[key],_=evaluate_dataset(model,groups[split],key,config)
    tiler_checks()
    save_json('results_summary.json',results)
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    run()
