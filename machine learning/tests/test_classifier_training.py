import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'flask-application'))
from ml_model import configure_subset_transforms

class TransformDataset(torch.utils.data.Dataset):
    transform=None
    def __len__(self):return 4
    def __getitem__(self,index):return self.transform(index) if self.transform else index

class TrainingTransformTests(unittest.TestCase):
    def test_training_augmentation_does_not_change_validation(self):
        source=TransformDataset()
        train,valid=torch.utils.data.random_split(source,[2,2],generator=torch.Generator().manual_seed(42))
        configure_subset_transforms(train,valid,lambda value:('augmented',value),lambda value:('evaluation',value))
        self.assertEqual(train[0][0],'augmented')
        self.assertEqual(valid[0][0],'evaluation')
        self.assertIsNone(source.transform)
        self.assertIsNot(train.dataset,valid.dataset)
