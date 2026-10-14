"""Read-only overview of saved evidence; never runs inference or training."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

class Evidence:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.sources = {}

    def read(self, relative):
        raw = (self.root / relative).read_bytes()
        self.sources[relative] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def tree_results(self):
        metrics = self.read('evaluations/tree_counter_hd/results_summary.json')
        lock = self.read('evaluations/tree_counter_hd/locked_settings.json')
        deployed = self.read('evaluations/tree_counter_hd/deployment_validation/flask_summary.json')
        if deployed['settings'] != lock['candidate'] or deployed['model_sha256'] != lock['model_sha256']:
            raise ValueError('Deployment evidence differs from candidate; review before presenting current results.')
        rows = []
        for key, values in metrics.items():
            version, population = key.split('_')
            rows.append({'Configuration': 'Deployed configuration' if version == 'candidate' else 'Previous app baseline',
                         'Population': 'Original test (site overlap / crown markers)' if population == 'test' else 'Reused external NEON',
                         'Confidence threshold': lock[version]['confidence'], **values})
        return pd.DataFrame(rows)

    def controls(self):
        rows = []
        for name, path in [('Classifier application', 'evaluations/classifier_hd/application_summary.json'),
                           ('Deployed tree application', 'evaluations/tree_counter_hd/deployment_validation/flask_summary.json')]:
            data = self.read(path)
            rows.append({'Scope': name, 'Checks': data['tests'], 'Passed': data['passed'],
                         'Failed': len(data['failed']), 'Meaning': 'Saved software checks; not predictive accuracy'})
        return pd.DataFrame(rows)

    def classifier_audit(self):
        audit = self.read('evaluations/classifier_hd/data_audit.json')
        return pd.DataFrame([
            {'Model': 'Binary mangrove detector', 'Reviewed labels in saved audit': audit['labelled_binary_rows'],
             'Independent accuracy': 'Not evaluated', 'Gap': audit['split_status']},
            {'Model': 'Mangrove category classifier', 'Reviewed labels in saved audit': audit['labelled_species_rows'],
             'Independent accuracy': 'Not evaluated', 'Gap': audit['species_mapping']},
            {'Model': 'Combined pipeline', 'Reviewed labels in saved audit': None,
             'Independent accuracy': 'Not evaluated', 'Gap': 'No scored, independently labelled end-to-end evaluation'}])

    def historical_binary(self):
        data = self.read('evaluations/classifier_hd/previous_evaluation_audit.json')['binary']
        return pd.DataFrame([{'Historical accuracy': data['recorded_accuracy'],
                              'Historical macro F1': data['recorded_macro_f1'],
                              'Recorded test tiles': data['recorded_test_tiles'],
                              'Test tiles implied by saved totals': data['test_tiles_implied_by_saved_totals'],
                              'Status': data['status']}])

    def benchmark(self):
        data = self.read('tree-counting-model/evaluation_pack/quality_audit.json')
        return pd.DataFrame([{'Original photographs': data['real_images'],
                              'Annotated crowns': data['source_tree_annotations'],
                              'Sites': len(data['source_sites']), 'Derived stress variants': data['mock_variants'],
                              'Source revision': data['source_revision']}])

    def provenance(self):
        return pd.DataFrame([{'Evidence file': name, 'SHA-256': value} for name, value in self.sources.items()])
