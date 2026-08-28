import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_MODEL_PATH = Path(__file__).parent / "best_tree_counter.pt"
DEFAULT_DATASET_YAML = Path(__file__).parent / "dataset1" / "data.yaml"


def _load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "The tree counter needs ultralytics. Install it with: "
            "pip install ultralytics"
        ) from exc
    return YOLO


def _tile_image(image_path, tile_size=640, overlap=64):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    step = tile_size - overlap

    tiles = []
    positions = []

    y_starts = list(range(0, max(height - tile_size + 1, 1), step))
    x_starts = list(range(0, max(width - tile_size + 1, 1), step))

    if y_starts[-1] != max(height - tile_size, 0):
        y_starts.append(max(height - tile_size, 0))
    if x_starts[-1] != max(width - tile_size, 0):
        x_starts.append(max(width - tile_size, 0))

    for top in y_starts:
        for left in x_starts:
            right = min(left + tile_size, width)
            bottom = min(top + tile_size, height)
            tile = image.crop((left, top, right, bottom))

            if tile.size != (tile_size, tile_size):
                padded = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
                padded.paste(tile, (0, 0))
                tile = padded

            tiles.append(tile)
            positions.append((left, top, right, bottom))

    return tiles, positions, image.size


def _to_global_boxes(result, position):
    left, top, right, bottom = position
    valid_width = right - left
    valid_height = bottom - top
    global_boxes = []

    if result.boxes is None:
        return global_boxes

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box

        # Ignore detections that fall entirely in padded tile area.
        if x1 >= valid_width or y1 >= valid_height:
            continue

        x1 = max(0.0, min(float(x1), float(valid_width))) + left
        y1 = max(0.0, min(float(y1), float(valid_height))) + top
        x2 = max(0.0, min(float(x2), float(valid_width))) + left
        y2 = max(0.0, min(float(y2), float(valid_height))) + top

        if x2 > x1 and y2 > y1:
            global_boxes.append([x1, y1, x2, y2, float(score)])

    return global_boxes


def _box_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    boxes_area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0, boxes[:, 3] - boxes[:, 1]
    )
    union = box_area + boxes_area - intersection
    return intersection / np.maximum(union, 1e-9)


def _nms(boxes, iou_threshold=0.45):
    if not boxes:
        return []

    boxes_array = np.array(boxes, dtype=np.float32)
    order = boxes_array[:, 4].argsort()[::-1]
    keep = []

    while order.size > 0:
        current = order[0]
        keep.append(current)

        if order.size == 1:
            break

        ious = _box_iou(boxes_array[current], boxes_array[order[1:]])
        order = order[1:][ious <= iou_threshold]

    return boxes_array[keep].tolist()


def train_tree_counter(
    data_yaml=DEFAULT_DATASET_YAML,
    base_model="yolo11n.pt",
    epochs=100,
    image_size=640,
    output_name="tree_counter",
):
    YOLO = _load_yolo()
    model = YOLO(base_model)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        name=output_name,
    )
    return results


def count_trees(
    image_path,
    model_path=DEFAULT_MODEL_PATH,
    tile_size=640,
    overlap=64,
    confidence=0.25,
    iou=0.45,
):
    YOLO = _load_yolo()
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Tree counter model not found: {model_path}. "
            "Train a model first or pass --model /path/to/best.pt."
        )

    tiles, positions, image_size = _tile_image(image_path, tile_size, overlap)
    model = YOLO(str(model_path))
    all_boxes = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        tile_paths = []

        for index, tile in enumerate(tiles):
            tile_path = tmpdir_path / f"tile_{index:05d}.jpg"
            tile.save(tile_path, quality=95)
            tile_paths.append(tile_path)

        results = model.predict(
            source=[str(path) for path in tile_paths],
            imgsz=tile_size,
            conf=confidence,
            iou=iou,
            verbose=False,
        )

        for result, position in zip(results, positions):
            all_boxes.extend(_to_global_boxes(result, position))

    final_boxes = _nms(all_boxes, iou_threshold=iou)

    return {
        "image": str(image_path),
        "image_width": image_size[0],
        "image_height": image_size[1],
        "tile_size": tile_size,
        "overlap": overlap,
        "tiles_processed": len(tiles),
        "raw_detections": len(all_boxes),
        "tree_count": len(final_boxes),
        "average_confidence": (
            float(np.mean([box[4] for box in final_boxes])) if final_boxes else 0.0
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Standalone tree counter.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data", default=str(DEFAULT_DATASET_YAML))
    train_parser.add_argument("--base-model", default="yolo11n.pt")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--image-size", type=int, default=640)
    train_parser.add_argument("--name", default="tree_counter")

    count_parser = subparsers.add_parser("count")
    count_parser.add_argument("image")
    count_parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    count_parser.add_argument("--tile-size", type=int, default=640)
    count_parser.add_argument("--overlap", type=int, default=64)
    count_parser.add_argument("--confidence", type=float, default=0.25)
    count_parser.add_argument("--iou", type=float, default=0.45)

    args = parser.parse_args()

    if args.command == "train":
        train_tree_counter(
            data_yaml=args.data,
            base_model=args.base_model,
            epochs=args.epochs,
            image_size=args.image_size,
            output_name=args.name,
        )
    elif args.command == "count":
        result = count_trees(
            image_path=args.image,
            model_path=args.model,
            tile_size=args.tile_size,
            overlap=args.overlap,
            confidence=args.confidence,
            iou=args.iou,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
