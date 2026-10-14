import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "tree-counting-model" / "tree_counter.py"
SPEC = importlib.util.spec_from_file_location("tree_counter", MODULE_PATH)
tree_counter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tree_counter)


class FakeBoxes:
    def __init__(self, boxes, confidence):
        self.xyxy = torch.tensor(boxes, dtype=torch.float32)
        self.conf = torch.tensor(confidence, dtype=torch.float32)


class FakeResult:
    def __init__(self, boxes, confidence):
        self.boxes = FakeBoxes(boxes, confidence)


class FakeModel:
    def predict(self, source, **_kwargs):
        self.source_count = len(source)
        return [
            FakeResult([[200, 100, 260, 160]], [0.8]),
            FakeResult([[40, 100, 100, 160]], [0.9]),
        ]


class TreeCounterTests(unittest.TestCase):
    def test_nms_keeps_highest_confidence_duplicate(self):
        boxes = [
            [10, 10, 30, 30, 0.7],
            [11, 11, 31, 31, 0.9],
            [100, 100, 120, 120, 0.8],
        ]
        kept = tree_counter._nms(boxes, iou_threshold=0.45)
        self.assertEqual(len(kept), 2)
        self.assertAlmostEqual(max(box[4] for box in kept), 0.9, places=5)

    def test_count_uses_shared_tiles_and_removes_cross_tile_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "input.jpg"
            overlay_path = Path(directory) / "overlay.jpg"
            Image.new("RGB", (800, 640), "white").save(image_path)

            model = FakeModel()
            result = tree_counter.count_trees(
                image_path=image_path,
                model=model,
                tile_size=640,
                overlap=64,
                overlay_path=overlay_path,
            )

            self.assertEqual(model.source_count, 2)
            self.assertEqual(result["tiles_processed"], 2)
            self.assertEqual(result["raw_detections"], 2)
            self.assertEqual(result["estimated_trees"], 1)
            self.assertTrue(overlay_path.exists())


if __name__ == "__main__":
    unittest.main()
