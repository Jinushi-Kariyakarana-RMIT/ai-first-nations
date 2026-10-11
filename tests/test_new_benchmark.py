import csv,hashlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluations/new_benchmark'))
import benchmark

class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.patches=[patch.object(benchmark,'ROOT',self.root),patch.object(benchmark,'HERE',self.root),patch.object(benchmark,'DB',self.root/'reviews.sqlite3')]
        for p in self.patches:p.start()
        Image.new('RGB',(16,16),'green').save(self.root/'source.png')
        with benchmark.database() as con:
            con.execute('INSERT INTO images(id,image_path,sha256,site_id,survey_id,split,provenance) VALUES(?,?,?,?,?,?,?)',('one','source.png',benchmark.sha(self.root/'source.png'),'site_a','survey_a','development','Test fixture only'))
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()
    def test_no_initial_pseudo_labels(self):
        with benchmark.database() as con:r=con.execute('SELECT * FROM images').fetchone()
        self.assertEqual(r['binary_label'],'');self.assertEqual(r['reviewed_utc'],'')
    def test_review_requires_consistent_labels_and_provenance(self):
        with self.assertRaises(ValueError):benchmark.update_review('one','Non-Mangrove','orange','Reviewer','Field record')
        with self.assertRaises(ValueError):benchmark.update_review('one','Mangrove','Unknown','','Field record')
        benchmark.update_review('one','Mangrove','Unknown','Reviewer','Field record')
        with benchmark.database() as con:
            self.assertEqual(con.execute('SELECT count(*) FROM history').fetchone()[0],1)
    def test_changed_pixels_reject_review(self):
        Image.new('RGB',(16,16),'blue').save(self.root/'source.png')
        with self.assertRaises(ValueError):benchmark.update_review('one','Mangrove','Unknown','Reviewer','Field record')
    def test_frozen_labels_are_immutable(self):
        with benchmark.database() as con:con.execute('UPDATE images SET frozen=1')
        with self.assertRaises(ValueError):benchmark.update_review('one','Mangrove','Unknown','Reviewer','Field record')
    def test_cross_split_site_rejected(self):
        Image.new('RGB',(16,16),'blue').save(self.root/'new.png')
        rows=[dict(id='new',image_path='new.png',site_id='site_a',survey_id='survey_b',split='test',provenance='Field survey',training_exclusion_evidence='New collection')]
        file=self.root/'import.csv'
        with file.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        with self.assertRaises(ValueError):benchmark.import_sources(file)
        with benchmark.database() as con:self.assertEqual(con.execute('SELECT count(*) FROM images').fetchone()[0],1)
    def test_final_test_requires_reviewed_classes(self):
        with benchmark.database() as con:con.execute("UPDATE images SET split='test'")
        with self.assertRaises(ValueError):benchmark.freeze()
        self.assertFalse((self.root/'test_lock.json').exists())
    def test_paths_cannot_escape_workspace(self):
        with self.assertRaises(ValueError):benchmark.image_path({'image_path':'../outside.png'})
    def test_export_keeps_review_provenance(self):
        benchmark.update_review('one','Non-Mangrove','Not applicable','Reviewer','Field verification')
        benchmark.export()
        row=list(csv.DictReader((self.root/'reviewed_manifest.csv').open()))[0]
        self.assertEqual(row['reviewer'],'Reviewer');self.assertEqual(row['binary_label'],'Non-Mangrove')

    def test_reviewed_binary_scoring_uses_reference_labels(self):
        Image.new('RGB',(16,16),'blue').save(self.root/'second.png')
        with benchmark.database() as con:
            con.execute('INSERT INTO images(id,image_path,sha256,site_id,survey_id,split,provenance) VALUES(?,?,?,?,?,?,?)',('two','second.png',benchmark.sha(self.root/'second.png'),'site_b','survey_b','development','Synthetic unit test only'))
        benchmark.update_review('one','Mangrove','Unknown','Unit test','Synthetic known-answer fixture')
        benchmark.update_review('two','Non-Mangrove','Not applicable','Unit test','Synthetic known-answer fixture')
        def prediction(path,model):
            positive=Path(path).name=='source.png'
            return dict(prediction='Mangrove' if positive else 'Non-Mangrove',confidence=.9,probs=[.1,.9] if positive else [.9,.1],n_tiles=1)
        with patch('binary_detector.load_binary_model',return_value=object()),patch('binary_detector.predict_binary',side_effect=prediction):
            benchmark.score('development')
        result=json.loads((self.root/'results/development/metrics.json').read_text())
        self.assertEqual(result['binary']['confusion_matrix'],[[1,0],[0,1]])
        self.assertIn('Not scored',result['species']['status'])
        with self.assertRaises(ValueError):benchmark.score('development')
