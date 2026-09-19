"""Standalone tree-counting training and inference.

The public result contains count-level diagnostics. Bounding boxes remain an
internal implementation detail used to remove duplicates and draw overlays.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
FLASK_APP_DIR = PROJECT_ROOT / "flask-application"
if str(FLASK_APP_DIR) not in sys.path:
    sys.path.insert(0, str(FLASK_APP_DIR))

from image_tiler import tile_image  # noqa: E402


DEFAULT_MODEL_PATH = MODULE_DIR / "best_tree_counter.pt"
DEFAULT_DATASET_YAML = MODULE_DIR / "dataset1" / "data.yaml"


def _load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "The tree counter requires ultralytics. Install the project requirements first."
        ) from exc
    return YOLO


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
        if x1 >= valid_width or y1 >= valid_height:
            continue

        clipped = [
            max(0.0, min(float(x1), float(valid_width))) + left,
            max(0.0, min(float(y1), float(valid_height))) + top,
            max(0.0, min(float(x2), float(valid_width))) + left,
            max(0.0, min(float(y2), float(valid_height))) + top,
            float(score),
        ]
        if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
            global_boxes.append(clipped)

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

    boxes_array = np.asarray(boxes, dtype=np.float32)
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


def _save_overlay(image_path, boxes, overlay_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(image.size) / 500))
    for x1, y1, x2, y2, _score in boxes:
        draw.rectangle((x1, y1, x2, y2), outline=(22, 112, 74), width=line_width)

    output = Path(overlay_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=92)
    return str(output)


def train_tree_counter(
    data_yaml=DEFAULT_DATASET_YAML,
    base_model="yolo11n.pt",
    epochs=30,
    image_size=640,
    output_name="tree_counter",
    device=None,
):
    YOLO = _load_yolo()
    model = YOLO(base_model)
    return model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        name=output_name,
        device=device,
        seed=42,
        deterministic=True,
    )


def count_trees(
    image_path,
    model_path=DEFAULT_MODEL_PATH,
    tile_size=640,
    overlap=64,
    confidence=0.15,
    iou=0.45,
    overlay_path=None,
    model=None,
):
    """Estimate visible trees in an image and return structured diagnostics."""
    model_path = Path(model_path)
    if model is None:
        if not model_path.exists():
            raise FileNotFoundError(f"Tree counter model not found: {model_path}")
        model = _load_yolo()(str(model_path))

    tiles, positions, image_size = tile_image(
        image=image_path,
        tile_size=tile_size,
        overlap=overlap,
        pad=True,
    )
    # Keep PIL sources in RGB so preprocessing matches file-path evaluation.
    sources = tiles
    results = model.predict(
        source=sources,
        imgsz=tile_size,
        conf=confidence,
        iou=iou,
        verbose=False,
    )

    all_boxes = []
    for result, position in zip(results, positions):
        all_boxes.extend(_to_global_boxes(result, position))
    final_boxes = _nms(all_boxes, iou_threshold=iou)

    saved_overlay = None
    if overlay_path is not None:
        saved_overlay = _save_overlay(image_path, final_boxes, overlay_path)

    return {
        "status": "available",
        "estimated_trees": len(final_boxes),
        "detected_crowns": len(final_boxes),
        "tiles_processed": len(tiles),
        "raw_detections": len(all_boxes),
        "average_confidence": (
            float(np.mean([box[4] for box in final_boxes])) if final_boxes else 0.0
        ),
        "image_width": image_size[0],
        "image_height": image_size[1],
        "tile_size": tile_size,
        "overlap": overlap,
        "overlay_path": saved_overlay,
    }


def main():
    parser = argparse.ArgumentParser(description="Standalone visible-tree counter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data", default=str(DEFAULT_DATASET_YAML))
    train_parser.add_argument("--base-model", default="yolo11n.pt")
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--image-size", type=int, default=640)
    train_parser.add_argument("--name", default="tree_counter")
    train_parser.add_argument("--device", default=None)

    count_parser = subparsers.add_parser("count")
    count_parser.add_argument("image")
    count_parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    count_parser.add_argument("--tile-size", type=int, default=640)
    count_parser.add_argument("--overlap", type=int, default=64)
    count_parser.add_argument("--confidence", type=float, default=0.15)
    count_parser.add_argument("--iou", type=float, default=0.45)
    count_parser.add_argument("--overlay")

    args = parser.parse_args()
    if args.command == "train":
        train_tree_counter(
            data_yaml=args.data,
            base_model=args.base_model,
            epochs=args.epochs,
            image_size=args.image_size,
            output_name=args.name,
            device=args.device,
        )
    else:
        result = count_trees(
            image_path=args.image,
            model_path=args.model,
            tile_size=args.tile_size,
            overlap=args.overlap,
            confidence=args.confidence,
            iou=args.iou,
            overlay_path=args.overlay,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
