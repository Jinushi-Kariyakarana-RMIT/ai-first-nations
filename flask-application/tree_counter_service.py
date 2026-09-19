"""Flask-facing adapter for the standalone tree counter."""

import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
TREE_COUNTER_DIR = PROJECT_ROOT / "tree-counting-model"
TREE_COUNTER_MODEL_PATH = TREE_COUNTER_DIR / "best_tree_counter.pt"

if str(TREE_COUNTER_DIR) not in sys.path:
    sys.path.insert(0, str(TREE_COUNTER_DIR))

from tree_counter import count_trees  # noqa: E402


def load_tree_counter_model():
    if not TREE_COUNTER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Tree counter weights not found: {TREE_COUNTER_MODEL_PATH}"
        )

    from ultralytics import YOLO

    return YOLO(str(TREE_COUNTER_MODEL_PATH))


def predict_tree_count(image_path, model, overlay_path):
    if model is None:
        return {
            "status": "unavailable",
            "message": "Tree counting is not available in this environment.",
        }

    return count_trees(
        image_path=image_path,
        model=model,
        tile_size=640,
        overlap=64,
        confidence=0.15,
        iou=0.45,
        overlay_path=overlay_path,
    )
