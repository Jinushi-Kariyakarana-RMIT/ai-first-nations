import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluations/classifier_hd'))
from metrics import score

class ClassifierMetricTests(unittest.TestCase):
    def test_perfect_binary(self):
        result=score([0,1],[[1,0],[0,1]],['Non-Mangrove','Mangrove'])
        self.assertEqual(result['confusion_matrix'],[[1,0],[0,1]])
        for key in ['accuracy','macro_f1','balanced_accuracy']:self.assertEqual(result[key],1)
        for key in ['brier','ece']:self.assertEqual(result[key],0)
    def test_confident_wrong(self):
        result=score([0,1],[[0,1],[1,0]],['no','yes'])
        self.assertEqual(result['accuracy'],0);self.assertEqual(result['brier'],1);self.assertEqual(result['ece'],1)
    def test_missing_class_is_not_zero(self):
        r=score([0],[[1,0,0]],['orange','red','yellow'])
        self.assertIsNone(r['balanced_accuracy']);self.assertIsNone(r['macro_f1'])
        self.assertIsNone(r['classes'][1]['recall']);self.assertIsNone(r['classes'][1]['roc_auc'])
    def test_multiclass_brier_convention(self):
        r=score([0,1,2],[[0,1,0],[0,0,1],[1,0,0]],['o','r','y'])
        self.assertEqual(r['brier'],2)
    def test_invalid_probabilities_rejected(self):
        with self.assertRaises(ValueError):score([0],[[.9,.9]],['no','yes'])
    def test_empty_labels_rejected(self):
        with self.assertRaises(ValueError):score([],[],['no','yes'])
