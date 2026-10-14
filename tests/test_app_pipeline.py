"""Regression checks for upload isolation, startup and partial failures."""
import importlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'flask-application'))

class AppPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {'FLASK_DEBUG': 'false'}), \
             patch('ml_model.load_model', side_effect=FileNotFoundError('missing')), \
             patch('ml_model.train_model', side_effect=AssertionError('Must not train')), \
             patch('binary_detector.load_binary_model', return_value=None), \
             patch('tree_counter_service.load_tree_counter_model', return_value=None):
            cls.module = importlib.import_module('app')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = patch.dict(self.module.app.config, {'TESTING': True, 'UPLOAD_FOLDER': self.temp.name})
        self.config.start()
        self.client = self.module.app.test_client()

    def tearDown(self):
        self.config.stop()
        self.temp.cleanup()

    def upload(self, content=None):
        if content is None:
            stream = io.BytesIO()
            Image.new('RGB', (32,32), 'green').save(stream, format='PNG')
            content = stream.getvalue()
        return self.client.post('/analyse', data={'orthomosaic': (io.BytesIO(content), 'same.png')})

    def test_duplicate_names_preserve_previous_upload(self):
        first = self.upload(); second = self.upload()
        self.assertNotEqual(first.location, second.location)
        self.assertEqual(len(list(Path(self.temp.name).glob('*.png'))), 2)

    def test_corrupt_duplicate_does_not_delete_previous_image(self):
        self.upload()
        before = {p.name:p.read_bytes() for p in Path(self.temp.name).iterdir()}
        self.upload(b'not an image')
        self.assertEqual(before, {p.name:p.read_bytes() for p in Path(self.temp.name).iterdir()})

    def test_startup_never_falls_back_to_training(self):
        with patch('app.load_model', side_effect=FileNotFoundError('missing')), \
             patch('app.load_binary_model', return_value=None), \
             patch('app.load_tree_counter_model', return_value=None), \
             patch('ml_model.train_model') as training:
            self.module.initialize_models()
            training.assert_not_called()

    def test_partial_failure_is_not_reported_as_success(self):
        response = self.upload()
        binary={'prediction':'Non-Mangrove','confidence':.9,'n_tiles':1}
        with patch.object(self.module,'binary_model',object()), \
             patch.object(self.module,'predict_binary',return_value=binary):
            page = self.client.get(response.location).get_data(as_text=True)
        self.assertIn('Analysis completed with a warning', page)
        self.assertNotIn('Analysis completed successfully', page)

    def test_directory_is_not_analysed(self):
        self.assertEqual(self.client.get('/analyze/..').status_code,302)
