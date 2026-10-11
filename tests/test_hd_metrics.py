"""Known-answer checks for official AP and zero-count conventions."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluations/tree_counter_hd'))
from evaluation import coco_metrics, operational_metrics


class HDMetricTests(unittest.TestCase):
    def sample(self,boxes):
        return dict(id='fixture',image='fixture.png',site='fixture',width=100,height=100,boxes=boxes)

    def prediction(self,boxes):
        return dict(id='fixture',boxes=boxes,raw_count=len(boxes),duplicate_count=0)

    def test_perfect_detection_has_perfect_ap(self):
        metric=coco_metrics([self.sample([[10,10,40,40]])],[self.prediction([[10,10,40,40,.9]])])
        self.assertAlmostEqual(metric['map50'],1)
        self.assertAlmostEqual(metric['map50_95'],1)

    def test_nonoverlapping_detection_has_zero_ap(self):
        metric=coco_metrics([self.sample([[10,10,40,40]])],[self.prediction([[60,60,90,90,.9]])])
        self.assertEqual(metric['map50'],0)

    def test_empty_truth_relative_error_is_undefined_not_zero(self):
        metric,_=operational_metrics([self.sample([])],[self.prediction([[10,10,40,40,.9]])])
        self.assertIsNone(metric['relative_absolute_error'])
        self.assertEqual(metric['mae'],1)
        self.assertEqual(metric['false_positives_on_empty'],1)
