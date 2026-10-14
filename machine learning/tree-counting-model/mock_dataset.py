"""Deterministic schematic fixtures, not evidence of real-world accuracy."""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent / 'mock_dataset'


def generate():
    rows = []
    hashes = set()
    previews = []
    scenarios = [('empty', 0), ('single', 1), ('sparse', 8),
                 ('medium', 24), ('dense', 60), ('tile_boundaries', 12)]
    for split, repetitions in [('train', 3), ('valid', 1), ('test', 1)]:
        for repeat in range(repetitions):
            for scenario, count in scenarios:
                name = f'{split}_{repeat}_{scenario}'
                rng = random.Random(name)
                width, height = (1280, 960) if scenario == 'tile_boundaries' else (640, 640)
                image = Image.new('RGB', (width, height),
                                  (rng.randint(110, 145), rng.randint(100, 125), 80))
                draw = ImageDraw.Draw(image)
                # Background marks are non-tree distractors, including empty scenes.
                for _ in range(150):
                    x, y = rng.randrange(width), rng.randrange(height)
                    draw.line((x, y, x + 5, y + 2), fill=(135, 125, 95), width=2)
                cells = [(x, y) for y in range(8) for x in range(8)]
                rng.shuffle(cells)
                boxes = []
                for index in range(count):
                    gx, gy = cells[index]
                    radius = rng.randint(17, 29)
                    cx, cy = gx * 80 + 40 + rng.randint(-6, 6), gy * 80 + 40 + rng.randint(-6, 6)
                    if scenario == 'tile_boundaries':
                        cx = [576, 640, 1152][index % 3] + rng.randint(-5, 5)
                        cy = 100 + (index // 3) * 210
                    box = (cx - radius, cy - radius, cx + radius, cy + radius)
                    draw.ellipse(box, fill=(rng.randint(30, 65), rng.randint(85, 135), 40))
                    for _ in range(35):
                        dx, dy = rng.randint(-radius + 5, radius - 5), rng.randint(-radius + 5, radius - 5)
                        if dx * dx + dy * dy < (radius - 5) ** 2:
                            draw.ellipse((cx + dx - 3, cy + dy - 3, cx + dx + 3, cy + dy + 3),
                                         fill=(65, rng.randint(115, 165), 55))
                    boxes.append(box)
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                assert digest not in hashes, 'Duplicate image across splits'
                hashes.add(digest)
                image_path = ROOT / split / 'images' / f'{name}.png'
                label_path = ROOT / split / 'labels' / f'{name}.txt'
                image_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(image_path)
                labels = []
                for x1, y1, x2, y2 in boxes:
                    assert 0 <= x1 < x2 < width and 0 <= y1 < y2 < height
                    labels.append(f'0 {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}')
                label_path.write_text('\n'.join(labels) + ('\n' if labels else ''))
                rows.append(dict(split=split, scenario=scenario, image=str(image_path.relative_to(ROOT)),
                                 labels=str(label_path.relative_to(ROOT)), expected_count=count, sha256=digest))
                if split == 'test':
                    annotated = image.copy()
                    overlay = ImageDraw.Draw(annotated)
                    for box in boxes:
                        overlay.rectangle(box, outline='yellow', width=2)
                    annotated.thumbnail((320, 240))
                    previews.append((scenario, count, annotated))
    with (ROOT / 'counts.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / 'data.yaml').write_text(f'path: {ROOT}\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n  0: tree\n')
    sheet = Image.new('RGB', (960, 560), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, (scenario, count, preview) in enumerate(previews):
        x, y = (index % 3) * 320, (index // 3) * 280
        draw.text((x + 8, y + 8), f'{scenario}: {count} synthetic trees', fill='black')
        sheet.paste(preview, (x, y + 30))
    sheet.save(ROOT / 'label_preview.jpg')
    print(f'Validated {len(rows)} unique images and matching labels at {ROOT}')


def evaluate():
    from tree_counter import count_trees, _load_yolo, DEFAULT_MODEL_PATH
    model = _load_yolo()(str(DEFAULT_MODEL_PATH))
    with (ROOT / 'counts.csv').open() as handle:
        samples = [row for row in csv.DictReader(handle) if row['split'] == 'test']
    results = []
    for row in samples:
        image_path = ROOT / row['image']
        prediction = count_trees(image_path, model=model,
                                 overlay_path=ROOT / 'predictions' / image_path.name)
        expected = int(row['expected_count'])
        predicted = prediction['estimated_trees']
        results.append(dict(image=row['image'], expected=expected, predicted=predicted,
                            error=predicted-expected, absolute_error=abs(predicted-expected)))
    report = dict(purpose='Synthetic pipeline smoke test only; not real-world accuracy.',
                  mean_absolute_error=sum(r['absolute_error'] for r in results)/len(results),
                  results=results)
    (ROOT / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    generate()
    if args.evaluate:
        evaluate()
