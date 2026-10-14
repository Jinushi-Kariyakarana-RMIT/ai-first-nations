"""Check label geometry and metric behavior, independently of trained weights."""
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tree-counting-model'))
from prepare_evaluation_pack import transform
from evaluate_prepared_pack import match_counts


class EvaluationPackTests(unittest.TestCase):
    def test_transformed_box_follows_pixels_on_non_square_image(self):
        pixels = np.zeros((60, 100, 3), dtype=np.uint8)
        pixels[10:30, 20:50] = 255
        for variant in ['flip_horizontal', 'rotate_90']:
            with self.subTest(variant=variant):
                image, boxes = transform(Image.fromarray(pixels), [[20, 10, 50, 30]], variant)
                y, x = np.where(np.asarray(image)[:, :, 0] > 0)
                self.assertEqual(boxes[0], [x.min(), y.min(), x.max()+1, y.max()+1])

    def test_multiple_predictions_cannot_match_same_tree(self):
        self.assertEqual(match_counts([[0,0,10,10,.9], [0,0,10,10,.8]], [[0,0,10,10]]), (1,1,0))

    def test_unmatched_prediction_and_missed_tree_are_both_counted(self):
        self.assertEqual(match_counts([[20,20,30,30,.9]], [[0,0,10,10]]), (0,1,1))

    def test_empty_predictions_and_labels(self):
        self.assertEqual(match_counts([], [[0,0,10,10]]), (0,0,1))
        self.assertEqual(match_counts([[0,0,10,10,.9]], []), (0,1,0))
        self.assertEqual(match_counts([], []), (0,0,0))
