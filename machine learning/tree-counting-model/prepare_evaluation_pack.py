"""Prepare published labels and deterministic stress cases; never train a model."""
import csv
import hashlib
import io
import json
import time
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'evaluation_pack'
REVISION = '96b29d566ca7ea9f604de3e1235c503ab101cb8a'
BASE = f'https://raw.githubusercontent.com/weecology/NeonTreeEvaluation/{REVISION}/'
VARIANTS = ['original', 'flip_horizontal', 'rotate_90', 'darker', 'blur', 'low_resolution']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch(path, destination, git_sha=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        data = destination.read_bytes()
    else:
        for attempt in range(4):
            try:
                response = requests.get(BASE + path, timeout=60)
                response.raise_for_status()
                data = response.content
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)
        destination.write_bytes(data)
    if git_sha:
        actual = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        if actual != git_sha:
            raise ValueError(f'Source checksum mismatch: {path}')
    return data


def site_of(stem):
    parts = stem.split('_')
    return parts[1] if parts[0].isdigit() and len(parts[0]) == 4 else parts[0]


def labels_from_xml(data, width, height):
    root = ET.fromstring(data)
    assert int(root.findtext('size/width')) == width
    assert int(root.findtext('size/height')) == height
    boxes, corrections = [], []
    for index, obj in enumerate(root.findall('object')):
        assert obj.findtext('name').lower() == 'tree'
        values = [float(obj.findtext('bndbox/' + key)) for key in ('xmin', 'ymin', 'xmax', 'ymax')]
        clipped = [max(0, min(width, values[0])), max(0, min(height, values[1])),
                   max(0, min(width, values[2])), max(0, min(height, values[3]))]
        assert clipped[0] < clipped[2] and clipped[1] < clipped[3], (index, values)
        if values != clipped:
            corrections.append(dict(object=index, original=values, clipped=clipped))
        boxes.append(clipped)
    assert len({tuple(b) for b in boxes}) == len(boxes), 'Duplicate source boxes'
    return boxes, corrections


def transform(image, boxes, variant):
    width, height = image.size
    if variant == 'flip_horizontal':
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), [[width-x2, y1, width-x1, y2] for x1,y1,x2,y2 in boxes]
    if variant == 'rotate_90':
        return image.transpose(Image.Transpose.ROTATE_90), [[y1, width-x2, y2, width-x1] for x1,y1,x2,y2 in boxes]
    if variant == 'darker':
        image = ImageEnhance.Brightness(image).enhance(0.65)
    elif variant == 'blur':
        image = image.filter(ImageFilter.GaussianBlur(1.2))
    elif variant == 'low_resolution':
        image = image.resize((width//2, height//2), Image.Resampling.BOX).resize((width, height), Image.Resampling.BILINEAR)
    elif variant != 'original':
        raise ValueError(variant)
    return image, boxes


def write_labels(path, boxes, size):
    width, height = size
    lines = []
    for x1, y1, x2, y2 in boxes:
        assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
        lines.append(f'0 {(x1+x2)/(2*width):.9f} {(y1+y2)/(2*height):.9f} {(x2-x1)/width:.9f} {(y2-y1)/height:.9f}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + ('\n' if lines else ''))


def prepare_one(pair):
    image_entry, xml_entry = pair
    stem = Path(image_entry['path']).stem
    raw = fetch(image_entry['path'], ROOT / 'source/rgb' / (stem + '.tif'), image_entry['sha'])
    xml = fetch(xml_entry['path'], ROOT / 'source/annotations' / (stem + '.xml'), xml_entry['sha'])
    image = Image.open(io.BytesIO(raw)).convert('RGB')
    boxes, corrections = labels_from_xml(xml, *image.size)
    rows = []
    for variant in VARIANTS:
        im, transformed = transform(image, boxes, variant)
        group = 'real_test' if variant == 'original' else 'mock_stress/' + variant
        image_path = ROOT / group / 'images' / (stem + '.png')
        label_path = ROOT / group / 'labels' / (stem + '.txt')
        image_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(image_path)
        write_labels(label_path, transformed, im.size)
        rows.append(dict(id=stem + '__' + variant, parent_id=stem, site=site_of(stem),
                         variant=variant, image=str(image_path.relative_to(ROOT)),
                         label=str(label_path.relative_to(ROOT)), width=im.width, height=im.height,
                         expected_count=len(boxes), label_origin='NEON published human annotation',
                         source_image_sha256=sha(raw), pixel_sha256=sha(im.tobytes()),
                         label_sha256=sha(label_path.read_bytes()), source_url=BASE + image_entry['path']))
    return rows, dict(image=stem, clipped_boxes=corrections)


def fingerprint(image):
    a = np.asarray(image.convert('L').resize((9, 8)), dtype=np.int16)
    return np.packbits(a[:, 1:] > a[:, :-1]).tobytes().hex()


def write_review_files(rows):
    """Keep clean evaluation photos separate from numbered reference previews."""
    fields = ['filename', 'variant', 'expected_count', 'site', 'image', 'label', 'parent_id']
    with (ROOT / 'counts.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: Path(row['image']).name if key == 'filename' else row[key] for key in fields})
    preview_dir = ROOT / 'labelled_previews'
    preview_dir.mkdir(exist_ok=True)
    for row in rows:
        if row['variant'] != 'original':
            continue
        im = Image.open(ROOT / row['image']).convert('RGB')
        canvas = Image.new('RGB', (im.width * 2, im.height * 2 + 30), 'white')
        canvas.paste(im.resize((im.width*2, im.height*2)), (0, 30))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"{row['parent_id']} | Reference count: {row['expected_count']}", fill='black')
        for index, line in enumerate((ROOT / row['label']).read_text().splitlines(), 1):
            _, cx, cy, w, h = map(float, line.split())
            box=((cx-w/2)*im.width*2, (cy-h/2)*im.height*2+30,
                 (cx+w/2)*im.width*2, (cy+h/2)*im.height*2+30)
            draw.rectangle(box, outline='yellow', width=2)
            draw.text((box[0]+1,box[1]+1),str(index),fill='yellow',stroke_width=1,stroke_fill='black')
        canvas.save(preview_dir / (row['parent_id'] + '.jpg'), quality=92)


def verify_pairs(rows):
    for row in rows:
        im = Image.open(ROOT / row['image']).convert('RGB')
        assert im.size == (int(row['width']), int(row['height']))
        assert sha(im.tobytes()) == row['pixel_sha256']
        data = (ROOT / row['label']).read_bytes()
        assert sha(data) == row['label_sha256']
        lines = data.decode().splitlines()
        assert len(lines) == int(row['expected_count'])
        for line in lines:
            cls, x, y, w, h = map(float, line.split())
            assert cls == 0 and w > 0 and h > 0
            assert x-w/2 >= -1e-8 and y-h/2 >= -1e-8
            assert x+w/2 <= 1+1e-8 and y+h/2 <= 1+1e-8
    return dict(pairs_checked=len(rows), status='passed',
                checks=['image decodability', 'pixel checksums', 'label checksums',
                        'label count equals expected count', 'normalized bounding box bounds'])


def main():
    ROOT.mkdir(exist_ok=True)
    tree_path = ROOT / 'source/repository_tree.json'
    if not tree_path.exists():
        response = requests.get(f'https://api.github.com/repos/weecology/NeonTreeEvaluation/git/trees/{REVISION}?recursive=1', timeout=60)
        response.raise_for_status()
        tree_path.parent.mkdir(parents=True, exist_ok=True)
        tree_path.write_text(json.dumps(response.json(), indent=2))
    tree = json.loads(tree_path.read_text())
    assert tree['sha'] == REVISION and not tree.get('truncated')
    annotations = {Path(x['path']).stem: x for x in tree['tree'] if x['path'].startswith('annotations/') and x['path'].endswith('.xml')}
    pairs = [(x, annotations[Path(x['path']).stem]) for x in tree['tree']
             if x['path'].startswith('evaluation/RGB/') and Path(x['path']).stem in annotations]
    for name in ['LICENSE', 'README.md']:
        entry = next(x for x in tree['tree'] if x['path'] == name)
        fetch(name, ROOT / 'source' / name, entry['sha'])
    rows, corrections = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, (new_rows, audit) in enumerate(pool.map(prepare_one, pairs), 1):
            rows.extend(new_rows)
            if audit['clipped_boxes']:
                corrections.append(audit)
            if index % 20 == 0:
                print(f'Prepared {index}/{len(pairs)} real images and their variants', flush=True)
    original = [r for r in rows if r['variant'] == 'original']
    assert len({r['pixel_sha256'] for r in original}) == len(original)
    existing = []
    for p in sorted((HERE / 'dataset1').glob('*/images/*')):
        if p.is_file():
            im = Image.open(p).convert('RGB')
            existing.append(dict(path=str(p.relative_to(HERE)), pixel_sha256=sha(im.tobytes()), dhash=fingerprint(im)))
    old_hashes = {r['pixel_sha256'] for r in existing}
    assert not old_hashes.intersection(r['pixel_sha256'] for r in original), 'Existing dataset leakage'
    nearest = []
    for row in original:
        f = int(fingerprint(Image.open(ROOT / row['image'])), 16)
        distance, old = min(((f ^ int(e['dhash'], 16)).bit_count(), e['path']) for e in existing)
        nearest.append(dict(image=row['image'], nearest_existing=old, dhash_distance=distance))
    with (ROOT / 'manifest.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_review_files(rows)
    for variant in VARIANTS:
        group = 'real_test' if variant == 'original' else 'mock_stress/' + variant
        (ROOT / group / 'data.yaml').write_text(f'path: {ROOT / group}\ntrain: null\nval: images\ntest: images\nnames:\n  0: tree\n')
    audit = dict(source_revision=REVISION, real_images=len(original), mock_variants=len(rows)-len(original),
                 unique_source_images=len(original), source_sites=dict(Counter(r['site'] for r in original)),
                 source_tree_annotations=sum(r['expected_count'] for r in original),
                 count_min=min(r['expected_count'] for r in original), count_max=max(r['expected_count'] for r in original),
                 empty_images=sum(r['expected_count']==0 for r in original),
                 exact_duplicate_originals=0, exact_matches_existing_dataset=0,
                 existing_images_checked=len(existing), nearest_existing_dhash=nearest,
                 possible_near_duplicates=[r for r in nearest if r['dhash_distance'] <= 8],
                 pair_validation=verify_pairs(rows),
                 source_box_clipping=corrections,
                 caveat='Hash checks do not prove geographic independence or exclude unknown pretraining exposure.')
    (ROOT / 'quality_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    # Representative label preview, selected by site before any model predictions.
    selected = list({r['site']: r for r in reversed(original)}.values())[:20]
    sheet = Image.new('RGB', (1200, 5*270), 'white')
    d = ImageDraw.Draw(sheet)
    for i, row in enumerate(selected):
        im = Image.open(ROOT / row['image']); draw = ImageDraw.Draw(im)
        for line in (ROOT / row['label']).read_text().splitlines():
            _, cx, cy, w, h = map(float, line.split())
            draw.rectangle(((cx-w/2)*im.width, (cy-h/2)*im.height, (cx+w/2)*im.width, (cy+h/2)*im.height), outline='yellow', width=1)
        im.thumbnail((290, 240)); x, y = (i%4)*300, (i//4)*270
        sheet.paste(im,(x,y+25));d.text((x+3,y+5),f"{row['parent_id']}: {row['expected_count']}",fill='black')
    sheet.save(ROOT / 'label_preview.jpg')
    # Inventory fresh images, keeping unknown labels distinct from zero trees.
    fresh_root = HERE.parent / 'Mourilyan_Harbour_Fresh_Test_Images'
    inventory = []
    for p in sorted(fresh_root.rglob('*')):
        if p.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}:
            continue
        with Image.open(p) as im:
            inventory.append(dict(image=str(p.relative_to(HERE.parent)), flight=p.parent.name,
                                  width=im.width, height=im.height, label_status='unlabelled', expected_count=''))
    if inventory:
        with (ROOT / 'mourilyan_inventory.csv').open('w', newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(inventory[0]));writer.writeheader();writer.writerows(inventory)
    print(json.dumps({k:v for k,v in audit.items() if k not in ['nearest_existing_dhash','source_box_clipping']},indent=2))


if __name__ == '__main__':
    main()
