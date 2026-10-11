"""Evaluate frozen weights using the application's tiled inference and NMS."""
import argparse
import csv
import hashlib
import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import ultralytics
from PIL import Image

from tree_counter import (DEFAULT_MODEL_PATH, _box_iou, _load_yolo, _nms,
                          _save_overlay, _to_global_boxes, count_trees)
from image_tiler import tile_image

ROOT = Path(__file__).resolve().parent / 'evaluation_pack'


class RecordingModel:
    """Record the exact YOLO results consumed by count_trees, without rerunning."""
    def __init__(self, model):
        self.model = model
        self.results = None

    def predict(self, **kwargs):
        self.results = self.model.predict(**kwargs)
        return self.results


def match_counts(predictions, ground_truth, threshold=0.5):
    used, true_positives = set(), 0
    gt = np.asarray(ground_truth, dtype=float).reshape(-1, 4)
    for pred in sorted(predictions, key=lambda b: b[4], reverse=True):
        if not len(gt):
            break
        overlaps = _box_iou(np.asarray(pred[:4]), gt)
        candidates = [i for i in range(len(gt)) if i not in used and overlaps[i] >= threshold]
        if candidates:
            index = max(candidates, key=lambda i: overlaps[i])
            used.add(index)
            true_positives += 1
    return true_positives, len(predictions)-true_positives, len(gt)-true_positives


def summarize(rows):
    errors = np.asarray([r['predicted']-r['expected'] for r in rows], dtype=float)
    tp, fp, fn = [sum(r[key] for r in rows) for key in ('tp', 'fp', 'fn')]
    expected = sum(r['expected'] for r in rows)
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/(tp+fn) if tp+fn else None
    # Resample sites, keeping correlated images together. This is an exploratory
    # interval across these observed sites, not a precision guarantee for mangroves.
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['site']].append(abs(r['predicted']-r['expected']))
    groups = list(grouped.values())
    rng = np.random.default_rng(42)
    means = []
    for _ in range(2000):
        drawn = [groups[i] for i in rng.integers(0, len(groups), size=len(groups))]
        means.append(sum(sum(g) for g in drawn)/sum(len(g) for g in drawn))
    return dict(images=len(rows), sites=len(groups), expected_trees=expected,
                predicted_trees=sum(r['predicted'] for r in rows),
                mae=float(np.abs(errors).mean()), rmse=float(np.sqrt((errors**2).mean())),
                bias=float(errors.mean()), wape=float(np.abs(errors).sum()/expected) if expected else None,
                precision_iou50=precision, recall_iou50=recall,
                f1_iou50=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                tp=tp, fp=fp, fn=fn,
                mae_site_bootstrap_95_interval=[float(v) for v in np.quantile(means,[.025,.975])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-only', action='store_true')
    args = parser.parse_args()
    with (ROOT / 'manifest.csv').open() as handle:
        manifest = list(csv.DictReader(handle))
    if args.original_only:
        manifest = [r for r in manifest if r['variant'] == 'original']
    # Match the app's default CPU inference rather than changing device/thresholds.
    model = RecordingModel(_load_yolo()(str(DEFAULT_MODEL_PATH)))
    rows, boxes_by_image = [], {}
    for index, row in enumerate(manifest, 1):
        path = ROOT / row['image']
        result = count_trees(path, model=model, tile_size=640, overlap=64, confidence=.15, iou=.45)
        _, positions, _ = tile_image(path, tile_size=640, overlap=64, pad=True)
        all_boxes = []
        for prediction, position in zip(model.results, positions):
            all_boxes.extend(_to_global_boxes(prediction, position))
        predictions = _nms(all_boxes, iou_threshold=.45)
        assert len(predictions) == result['estimated_trees']
        width, height = int(row['width']), int(row['height'])
        truth = []
        for line in (ROOT / row['label']).read_text().splitlines():
            _, cx, cy, w, h = map(float, line.split())
            truth.append([(cx-w/2)*width, (cy-h/2)*height, (cx+w/2)*width, (cy+h/2)*height])
        assert len(truth) == int(row['expected_count'])
        tp, fp, fn = match_counts(predictions, truth)
        rows.append(dict(id=row['id'], parent_id=row['parent_id'], site=row['site'], variant=row['variant'],
                         image=row['image'], expected=len(truth), predicted=len(predictions),
                         error=len(predictions)-len(truth), absolute_error=abs(len(predictions)-len(truth)),
                         tp=tp, fp=fp, fn=fn, tiles=result['tiles_processed']))
        boxes_by_image[row['id']] = predictions
        if index % 50 == 0:
            print(f'Evaluated {index}/{len(manifest)} images', flush=True)
    output = ROOT / ('results_original_only' if args.original_only else 'results')
    output.mkdir(exist_ok=True)
    with (output / 'per_image.csv').open('w', newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output / 'predicted_boxes.json').write_text(json.dumps(boxes_by_image))
    variants = sorted({r['variant'] for r in rows})
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                   model_sha256=hashlib.sha256(DEFAULT_MODEL_PATH.read_bytes()).hexdigest(),
                   manifest_sha256=hashlib.sha256((ROOT/'manifest.csv').read_bytes()).hexdigest(),
                   python=platform.python_version(), torch=torch.__version__, ultralytics=ultralytics.__version__,
                   confidence=.15, nms_iou=.45, matching_iou=.5, tile_size=640, overlap=64,
                   by_variant={v:summarize([r for r in rows if r['variant']==v]) for v in variants},
                   by_site_original={s:summarize([r for r in rows if r['variant']=='original' and r['site']==s])
                                     for s in sorted({r['site'] for r in rows if r['variant']=='original'})},
                   limitations=['Variants are paired robustness tests, not additional independent samples.',
                                'NEON forest imagery is not a Mourilyan mangrove validation set.',
                                'All scores use published image labels, not field-confirmed stem counts.',
                                'The source has no required representation of tree-free scenes.',
                                'Reported precision and recall are at one fixed threshold, not mAP.'])
    (output / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Worst original errors are transparent review examples, not a representative sample.
    worst=sorted([r for r in rows if r['variant']=='original'],key=lambda r:r['absolute_error'],reverse=True)[:12]
    for row in worst:
        path = output / 'worst_cases' / (row['parent_id']+'.jpg')
        _save_overlay(ROOT/row['image'], boxes_by_image[row['id']], path)
    lines=['# Tree counter evaluation results', '',
           'Frozen current model; same tiled counting path and thresholds as the application.', '',
           '| Test condition | Images | Count MAE | Bias | Precision @ IoU .5 | Recall @ IoU .5 |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for variant, metric in summary['by_variant'].items():
        p, r = metric['precision_iou50'], metric['recall_iou50']
        p_text = f'{p:.1%}' if p is not None else 'undefined'
        r_text = f'{r:.1%}' if r is not None else 'undefined'
        lines.append(f"| {variant} | {metric['images']} | {metric['mae']:.2f} | {metric['bias']:.2f} | {p_text} | {r_text} |")
    lines += ['', 'MAE is average absolute count error per image. Negative bias means undercounting.',
              'Precision and recall compare individual boxes with the source labels; a matching total alone is insufficient.',
              '', '## Interpretation', '',
              'Use the original-image result as the external benchmark. Interpret each mock condition relative to its own original parents.',
              'Do not pool the variants into a larger independent sample or use these test images to retrain while retaining the test claim.',
              'The site-cluster bootstrap interval in summary.json is exploratory and cannot establish Mourilyan performance.',
              '', 'See per_image.csv, predicted_boxes.json and worst_cases for inspectable evidence.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary['by_variant'],indent=2))


if __name__ == '__main__':
    main()
