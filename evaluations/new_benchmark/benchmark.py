"""Build a new reviewed benchmark when historical training data is unavailable.

Local-only review UI. Existing inspected imagery is development data. Unseen test
surveys are imported separately and frozen before evaluation. No pseudo-labels.
"""
from pathlib import Path
import argparse,csv,hashlib,json,os,secrets,sqlite3,sys
from datetime import datetime,timezone
from contextlib import contextmanager
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
DB=HERE/'reviews.sqlite3'
BINARY=['Non-Mangrove','Mangrove','Uncertain']
SPECIES=['orange','red','yellow','Mixed','Unknown','Not applicable']

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def image_path(row):
    path=(ROOT/row['image_path']).resolve()
    if not path.is_relative_to(ROOT.resolve()):raise ValueError('Images must be inside the project workspace.')
    return path
@contextmanager
def database():
    con=sqlite3.connect(DB);con.row_factory=sqlite3.Row
    con.execute('''CREATE TABLE IF NOT EXISTS images (
      id TEXT PRIMARY KEY, image_path TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE,
      site_id TEXT NOT NULL, survey_id TEXT NOT NULL, split TEXT NOT NULL,
      provenance TEXT NOT NULL, training_exclusion_evidence TEXT NOT NULL DEFAULT '',
      binary_label TEXT NOT NULL DEFAULT '', species_label TEXT NOT NULL DEFAULT '',
      reviewer TEXT NOT NULL DEFAULT '', annotation_basis TEXT NOT NULL DEFAULT '',
      reviewed_utc TEXT NOT NULL DEFAULT '', frozen INTEGER NOT NULL DEFAULT 0)''')
    con.execute('''CREATE TABLE IF NOT EXISTS history (id TEXT, changed_utc TEXT, previous_json TEXT, new_json TEXT)''')
    try:yield con;con.commit()
    finally:con.close()

def prepare():
    inventory=ROOT/'evaluations/classifier_hd/source_inventory.csv'
    with database() as con:
        for row in csv.DictReader(inventory.open()):
            con.execute('INSERT OR IGNORE INTO images(id,image_path,sha256,site_id,survey_id,split,provenance) VALUES(?,?,?,?,?,?,?)',
                        (row['source_image_id'],row['image_path'],row['sha256'],row['site_id'],row['survey_id'],'development','Existing inspected Mourilyan pool; not a blinded holdout.'))
        print('Review candidates:',con.execute('SELECT count(*) FROM images').fetchone()[0])

def validate_review(binary,species,reviewer,basis):
    if binary not in BINARY or species not in SPECIES:raise ValueError('Choose valid labels.')
    if not reviewer.strip() or not basis.strip():raise ValueError('Reviewer and annotation basis are required.')
    if binary=='Non-Mangrove' and species!='Not applicable':raise ValueError('Non-mangrove images need category Not applicable.')
    if binary=='Uncertain' and species!='Unknown':raise ValueError('Uncertain images need category Unknown.')
    if binary=='Mangrove' and species=='Not applicable':raise ValueError('Choose a category, Mixed, or Unknown for mangrove images.')

def update_review(identity,binary,species,reviewer,basis):
    validate_review(binary,species,reviewer,basis)
    with database() as con:
        row=con.execute('SELECT * FROM images WHERE id=?',(identity,)).fetchone()
        if row is None:raise ValueError('Unknown image.')
        if row['frozen']:raise ValueError('This record belongs to a frozen benchmark. It cannot be edited.')
        if sha(image_path(row))!=row['sha256']:raise ValueError('Image changed since import. Review stopped.')
        values=dict(binary_label=binary,species_label=species,reviewer=reviewer.strip(),annotation_basis=basis.strip(),reviewed_utc=now())
        con.execute('UPDATE images SET binary_label=?,species_label=?,reviewer=?,annotation_basis=?,reviewed_utc=? WHERE id=?',tuple(values.values())+(identity,))
        con.execute('INSERT INTO history VALUES(?,?,?,?)',(identity,now(),json.dumps(dict(row)),json.dumps(values)))

def import_sources(manifest):
    rows=list(csv.DictReader(Path(manifest).open()))
    with database() as con:
        if (HERE/'test_lock.json').exists():raise ValueError('Final test is already frozen; start a separately versioned benchmark.')
        existing=list(con.execute('SELECT * FROM images'))
        for row in rows:
            for key in ['id','image_path','site_id','survey_id','split','provenance','training_exclusion_evidence']:
                if not row.get(key,'').strip():raise ValueError(f'Missing {key}')
            if row['split'] not in ['validation','test']:raise ValueError('Imported split must be validation or test.')
            if any(x['site_id']==row['site_id'] and x['split']!=row['split'] for x in existing):raise ValueError('Site overlaps another split. Use a separate unseen site.')
            path=image_path(row)
            from PIL import Image
            with Image.open(path) as im:im.verify()
            digest=sha(path)
            if any(x['sha256']==digest for x in existing):raise ValueError('Duplicate image across benchmark sources.')
            identity=row['id']
            con.execute('INSERT INTO images(id,image_path,sha256,site_id,survey_id,split,provenance,training_exclusion_evidence) VALUES(?,?,?,?,?,?,?,?)',
                        (identity,row['image_path'],digest,row['site_id'],row['survey_id'],row['split'],row['provenance'],row['training_exclusion_evidence']))
            existing.append(dict(row,sha256=digest))
    print('Imported',len(rows),'new sources. Labels require independent review; provenance assertions still require audit.')

def export():
    with database() as con:rows=[dict(r) for r in con.execute('SELECT * FROM images ORDER BY split,site_id,survey_id,id')]
    path=HERE/'reviewed_manifest.csv'
    with path.open('w',newline='') as f:
        keys=list(rows[0]) if rows else ['id'];writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    print(path)

def freeze():
    lock=HERE/'test_lock.json'
    if lock.exists():raise ValueError('Final test already frozen.')
    with database() as con:
        rows=[dict(r) for r in con.execute("SELECT * FROM images WHERE split='test' ORDER BY id")]
        if not rows or any(not r['reviewed_utc'] for r in rows):raise ValueError('Every test source must have an independent review first.')
        if {r['binary_label'] for r in rows}.intersection({'Mangrove','Non-Mangrove'})!={'Mangrove','Non-Mangrove'}:raise ValueError('Test needs reviewed mangrove and non-mangrove sources.')
        approval=HERE/'holdout_review.json'
        if not approval.exists():raise ValueError('Complete holdout_review.json from its example after reviewing near-duplicates, geographic overlap and provenance.')
        review=json.loads(approval.read_text())
        for key in ['reviewer','training_exclusion_basis','overlap_review_basis']:
            if not str(review.get(key,'')).strip():raise ValueError(f'Missing holdout review: {key}')
        if review.get('approved') is not True:raise ValueError('Holdout reviewer approval is missing.')
        for r in rows:
            if sha(image_path(r))!=r['sha256']:raise ValueError('Source bytes changed.')
        model_files=['flask-application/bestBinary.pth','flask-application/best_mangrove_model.pth']
        payload=dict(created_utc=now(),rows=rows,review=review,models={p:sha(ROOT/p) for p in model_files},
                     settings={'tile_size':512,'overlap':64,'aggregation':'mean tile softmax','binary_threshold':.5},
                     purpose='Fixed baseline final test. Never select thresholds on these results.')
        # Exclusive create avoids replacing an earlier frozen test.
        with lock.open('x') as f:json.dump(payload,f,indent=2)
        con.execute("UPDATE images SET frozen=1 WHERE split='test'")
    print('Frozen',len(rows),'test sources. Do not tune on final-test results.')

def score(split):
    sys.path.insert(0,str(ROOT/'evaluations/classifier_hd'));sys.path.insert(0,str(ROOT/'flask-application'))
    import numpy as np
    from metrics import score as metrics
    from binary_detector import load_binary_model,predict_binary
    from ml_model import load_model,predicting_test,INFERENCE_TRANSFORM
    with database() as con:rows=[dict(r) for r in con.execute('SELECT * FROM images WHERE split=? ORDER BY id',(split,))]
    if not rows or any(not r['reviewed_utc'] for r in rows):raise ValueError('All selected sources require review before evaluation.')
    if split=='test':
        lock=json.loads((HERE/'test_lock.json').read_text())
        for file,digest in lock['models'].items():
            if sha(ROOT/file)!=digest:raise ValueError('Model differs from frozen test protocol.')
        locked={r['id']:r for r in lock['rows']}
        for row in rows:
            if row['id'] not in locked or any(row[k]!=locked[row['id']][k] for k in ['sha256','binary_label','species_label','reviewer','annotation_basis','image_path','site_id','survey_id']):raise ValueError('Frozen manifest differs from database.')
        if len(rows)!=len(locked):raise ValueError('Frozen test membership changed.')
    known=[r for r in rows if r['binary_label'] in BINARY[:2]]
    if set(r['binary_label'] for r in known)!=set(BINARY[:2]):raise ValueError('Both binary classes are required; uncertain examples do not receive guessed truth.')
    result_dir=HERE/'results'/split
    if result_dir.exists():raise ValueError('Results already exist. Preserve this run and version the benchmark explicitly.')
    predictions=[];binary=load_binary_model()
    for row in known:
        if sha(image_path(row))!=row['sha256']:raise ValueError('Source bytes changed.')
        prediction=predict_binary(str(image_path(row)),binary)
        predictions.append(dict(id=row['id'],site_id=row['site_id'],survey_id=row['survey_id'],truth=row['binary_label'],**prediction))
    binary_metrics=metrics([BINARY.index(r['truth']) for r in predictions],[r['probs'] for r in predictions],BINARY[:2])
    result=dict(split=split,n_total=len(rows),uncertain_excluded=len(rows)-len(known),binary=binary_metrics,species={'status':'Not scored: verified checkpoint category mapping required.'})
    mapping_file=HERE/'category_mapping.json'
    species_predictions=[]
    if mapping_file.exists():
        mapping=json.loads(mapping_file.read_text())
        if mapping.get('verified') is not True or not mapping.get('reviewer') or not mapping.get('evidence'):raise ValueError('Category mapping requires verification evidence.')
        if mapping.get('model_sha256')!=sha(ROOT/'flask-application/best_mangrove_model.pth'):raise ValueError('Category mapping belongs to a different checkpoint.')
        model,labels,device=load_model()
        if mapping.get('labels')!=labels:raise ValueError('Verified mapping differs from serving order.')
        eligible=[r for r in rows if r['binary_label']=='Mangrove' and r['species_label'] in labels]
        if set(r['species_label'] for r in eligible)!=set(labels):raise ValueError('All three independently labelled categories are required.')
        for row in eligible:
            if sha(image_path(row))!=row['sha256']:raise ValueError('Source bytes changed.')
            label,confidence,prob=predicting_test(str(image_path(row)),model,INFERENCE_TRANSFORM,labels,device)
            species_predictions.append(dict(id=row['id'],truth=row['species_label'],prediction=label,probabilities=prob.tolist()))
        result['species']=metrics([labels.index(r['truth']) for r in species_predictions],[r['probabilities'] for r in species_predictions],labels)
        result['species_scope']='Conditional category accuracy on ground-truth mangrove images, not end-to-end gated accuracy.'
    result_dir.mkdir(parents=True)
    for filename,payload in [('metrics.json',result),('binary_predictions.json',predictions),('species_predictions.json',species_predictions)]:
        (result_dir/filename).write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print('Scored reviewed sources:',result_dir)

def create_app():
    from flask import Flask,request,render_template,send_file,redirect,url_for,session,abort
    app=Flask(__name__,template_folder=str(HERE/'templates'));app.secret_key=secrets.token_hex(32)
    @app.route('/')
    def index():
        with database() as con:
            total=con.execute('SELECT count(*) FROM images').fetchone()[0]
            reviewed=con.execute("SELECT count(*) FROM images WHERE reviewed_utc!=''").fetchone()[0]
            identity=request.args.get('id')
            row=con.execute('SELECT * FROM images WHERE id=?',(identity,)).fetchone() if identity else con.execute("SELECT * FROM images WHERE reviewed_utc='' AND frozen=0 ORDER BY split,id LIMIT 1").fetchone()
        session.setdefault('csrf',secrets.token_urlsafe(32))
        return render_template('review.html',row=row,total=total,reviewed=reviewed,binary=BINARY,species=SPECIES,csrf=session['csrf'])
    @app.route('/image/<identity>')
    def image(identity):
        with database() as con:row=con.execute('SELECT * FROM images WHERE id=?',(identity,)).fetchone()
        if row is None:abort(404)
        return send_file(image_path(row))
    @app.route('/review',methods=['POST'])
    def review():
        if not secrets.compare_digest(request.form.get('csrf',''),session.get('csrf','invalid')):abort(403)
        try:update_review(request.form['id'],request.form['binary'],request.form['species'],request.form['reviewer'],request.form['basis'])
        except (ValueError,KeyError) as e:return str(e),400
        return redirect(url_for('index'))
    return app

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','serve','import','export','freeze','score']);parser.add_argument('--manifest');parser.add_argument('--split',choices=['development','validation','test'],default='development');args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='serve':create_app().run(host='127.0.0.1',port=5054,debug=False)
    elif args.command=='import':import_sources(args.manifest)
    elif args.command=='export':export()
    elif args.command=='freeze':freeze()
    else:score(args.split)
