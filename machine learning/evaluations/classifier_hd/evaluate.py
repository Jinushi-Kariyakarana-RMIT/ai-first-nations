"""Audit and exercise deployed classifiers. Unlabelled predictions are not accuracy.

Runs no training, performs no checkpoint downloads and never assigns ground truth
from model outputs. Test uploads and derived fixtures live in a temporary folder.
"""
from pathlib import Path
import contextlib, hashlib, importlib, io, json, logging, os, platform, random, subprocess, sys, tempfile, threading, time
from importlib.metadata import version
from unittest.mock import patch
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import psutil, requests, torch
from bs4 import BeautifulSoup
from flask import template_rendered
from werkzeug.serving import make_server

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'flask-application'))
os.environ['YOLO_AUTOINSTALL']='false'
os.environ['FLASK_SECRET_KEY']='isolated-classifier-evaluation'
SEED=42
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
def relative(path):return str(Path(path).relative_to(ROOT))

def data_audit():
    fresh=sorted((ROOT/'Mourilyan_Harbour_Fresh_Test_Images').rglob('*.JPG'))
    rows=[]
    for path in fresh:
        with Image.open(path) as im:
            width,height=im.size
            im.draft('RGB',(144,144));small=np.asarray(im.convert('L').resize((9,8)))
            dh=sum(int(b)<<i for i,b in enumerate((small[:,1:]>small[:,:-1]).flatten()))
        rows.append(dict(source_image_id=path.stem,image_path=relative(path),site_id='Mourilyan Harbour',
                         survey_id=path.parent.name,metadata_basis='Folder and filename, not independently verified survey metadata',
                         width=width,height=height,sha256=sha(path),dhash=f'{dh:016x}',
                         binary_label='',species_label='',split='unassigned',annotation_provenance='',reviewer='',
                         training_exclusion_verified=False))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'source_inventory.csv',index=False)
    template=OUT/'label_manifest.csv'
    if not template.exists():frame.to_csv(template,index=False)
    pairs=[]
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            distance=(int(a['dhash'],16)^int(b['dhash'],16)).bit_count()
            if distance<=6:pairs.append(dict(first=a['source_image_id'],second=b['source_image_id'],dhash_distance=distance,
                                           interpretation='Review candidate only; perceptual similarity is not proof of duplication.'))
    pd.DataFrame(pairs,columns=['first','second','dhash_distance','interpretation']).to_csv(OUT/'near_duplicate_candidates.csv',index=False)
    labels=pd.read_csv(template,keep_default_na=False)
    labelled_binary=int(labels.binary_label.isin(['Non-Mangrove','Mangrove']).sum())
    labelled_species=int(labels.species_label.isin(['orange','red','yellow']).sum())
    audit=dict(fresh_source_images=len(rows),labelled_binary_rows=labelled_binary,labelled_species_rows=labelled_species,
               missing_expected_datasets=[x for x in ['TrainingData','test'] if not (ROOT/x).exists()],
               exact_duplicate_excess=int(frame.sha256.duplicated().sum()),near_duplicate_candidate_pairs=len(pairs),
               historical_training_manifest_available=False,
               split_status='BLOCKED: no verified original training membership or independent labelled source/site holdout.',
               binary_mapping='0 Non-Mangrove, 1 Mangrove: consistent with legacy notebook code; checkpoint carries no class metadata.',
               species_mapping='0 orange, 1 red, 2 yellow: serving fallback only; original dataset/checkpoint mapping metadata missing.',
               not_eligible='NEON crown boxes and synthetic tree labels do not establish mangrove presence or Orange/Red/Yellow labels.')
    save('data_audit.json',audit)
    return fresh

def tile_probabilities(path,model,transform,device):
    from image_tiler import tile_image_for_model
    tiles,positions=tile_image_for_model(path,transform,512,64)
    with torch.inference_mode():
        p=np.concatenate([torch.softmax(model(tiles[i:i+64].to(device)),dim=1).cpu().numpy() for i in range(0,len(tiles),64)])
    return p,positions


def run():
    fresh=data_audit();print('Data inventory complete',flush=True)
    tests=[];latency=[];predictions=[];tiles_saved=[];proc=psutil.Process()
    def check(model,name,passed,expected,observed,kind='actual model / route'):
        tests.append(dict(model=model,test=name,status='PASS' if passed else 'FAIL',kind=kind,expected=str(expected),observed=str(observed)))
    started=time.perf_counter()
    with patch('ml_model.train_model',side_effect=AssertionError('No training permitted')),patch('torch.hub.download_url_to_file',side_effect=AssertionError('No downloads permitted')),patch('ultralytics.utils.downloads.safe_download',side_effect=AssertionError('No downloads permitted')):
        module=importlib.import_module('app')
    cold_start=time.perf_counter()-started
    import binary_detector as binary
    import ml_model as species
    from image_tiler import tile_image
    check('both','Load exact local checkpoints without training/download',module.binary_model is not None and module.model is not None,'Two classifier models loaded',bool(module.binary_model is not None and module.model is not None))
    check('both','Identical deployed transforms',repr(binary.INFERENCE_TRANSFORM)==repr(species.INFERENCE_TRANSFORM),'same resize/RGB normalization',repr(binary.INFERENCE_TRANSFORM))
    packages=['torch','torchvision','numpy','pandas','scikit-learn','Pillow','Flask','nbformat','nbclient','nbconvert','psutil','requests','beautifulsoup4','ultralytics']
    env=dict(seed=SEED,python=sys.version,platform=platform.platform(),machine=platform.machine(),device=str(module.gpu),
             cpu_count=os.cpu_count(),physical_ram_gib=round(psutil.virtual_memory().total/1024**3,2),
             versions={x:version(x) for x in packages},cold_process_model_loading_seconds=cold_start,
             binary=dict(architecture=str(module.binary_model.classifier),sha256=sha(binary.BINARY_MODEL_PATH),labels=binary.BINARY_LABELS),
             species=dict(architecture=str(module.model.classifier),sha256=sha(species.MODEL_PATH),labels=module.mangrove_type,label_mapping_verified=False),
             transforms=repr(binary.INFERENCE_TRANSFORM),tile_size=512,overlap=64,batch_size=64,aggregation='Arithmetic mean of per-tile softmax probabilities',
             class_gate='Binary predicts Mangrove with winning probability > 0.5; otherwise category classifier does not run.',
             probability_tolerance=1e-6,model_selection='No weights, thresholds or aggregation selected or changed using these unlabelled images.',
             source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'flask-application/app.py',ROOT/'flask-application/ml_model.py',ROOT/'flask-application/binary_detector.py',ROOT/'flask-application/image_tiler.py',OUT/'evaluate.py',OUT/'metrics.py']})
    save('environment.json',env)
    (OUT/'requirements-lock.txt').write_text('\n'.join(f'{x}=={version(x)}' for x in packages)+'\n')
    protocol=dict(seed=SEED,accuracy_gate='Reviewed labels, verified checkpoint mapping and training exclusion; frozen source/site-separated validation and test manifests required before scoring.',
                  binary_experiments=['Validation-only threshold selection for recall/precision trade-off','Validation-only temperature scaling minimizing log loss; independent test Brier/ECE comparison'],
                  species_experiments=['Validation-only temperature scaling minimizing log loss','Compare mean tile probabilities against mean tile logits on labelled validation sources'],
                  selection='Register grids, objective and tie rule before inference on a locked final test set; no selection is made in this run.',
                  status='NOT RUN: labelled independent validation/test sources unavailable. Sensitivity diagnostics below are not improvement experiments.',
                  targets_binary=dict(source_recall=.95,source_precision=.90,macro_f1=.90),
                  targets_species=dict(macro_f1=.90,per_class_recall=.85,balanced_accuracy=.90),
                  targets_status='Proposed in supplied rubrics, not pre-agreed production acceptance limits.')
    save('experiment_protocol.json',protocol)
    with tempfile.TemporaryDirectory(prefix='mangrove-classifier-eval-') as temporary:
        temp=Path(temporary);fixtures=[]
        chosen=[fresh[0],fresh[len(fresh)//2],fresh[-1]]
        for i,path in enumerate(chosen):
            with Image.open(path) as im:base=im.convert('RGB');base.thumbnail((1024,768))
            variants={'reference':base,'darker':ImageEnhance.Brightness(base).enhance(.55),'blur':base.filter(ImageFilter.GaussianBlur(3))}
            for variant,im in variants.items():
                target=temp/f'source_{i}_{variant}.png';im.save(target)
                fixtures.append(dict(id=f'source_{i}_{variant}',path=target,source=relative(path),variant=variant,derived=True))
        for name,size,mode in [('small',(180,120),'RGB'),('edge',(513,769),'RGB'),('grayscale',(512,512),'L'),('rgba',(512,512),'RGBA')]:
            target=temp/f'{name}.png'
            with Image.open(chosen[0]) as im:im.convert(mode).resize(size).save(target)
            fixtures.append(dict(id=name,path=target,source=relative(chosen[0]),variant=name,derived=True))
        fixtures.append(dict(id='full_resolution',path=chosen[1],source=relative(chosen[1]),variant='original',derived=False))
        coverage=[]
        for w,h in [(180,120),(512,512),(513,769),(1024,768),(4032,3024)]:
            _,positions,_=tile_image(Image.new('RGB',(w,h)),512,64)
            covered=np.zeros((h,w),dtype=bool)
            for x0,y0,x1,y1 in positions:covered[y0:y1,x0:x1]=True
            coverage.append(dict(width=w,height=h,tiles=len(positions),uncovered_pixels=int((~covered).sum())))
        pd.DataFrame(coverage).to_csv(OUT/'tiler_coverage.csv',index=False)
        check('both','Shared tiler covers small, exact, edge and full-size images',all(x['uncovered_pixels']==0 for x in coverage),'0 uncovered pixels',coverage)
        # Direct inference and independently collected tile probabilities.
        for fixture in fixtures:
            path=fixture['path'];print('Inference',fixture['id'],flush=True)
            for name,model,transform,device in [('binary',module.binary_model,binary.INFERENCE_TRANSFORM,binary.GPU),('species',module.model,species.INFERENCE_TRANSFORM,module.gpu)]:
                start=time.perf_counter();p,positions=tile_probabilities(path,model,transform,device);elapsed=time.perf_counter()-start
                if name=='binary':direct=binary.predict_binary(str(path),model);prob=np.array(direct['probs']);label=direct['prediction']
                else:
                    label,confidence,prob=species.predicting_test(str(path),model,transform,module.mangrove_type,device)
                delta=float(np.max(np.abs(prob-p.mean(0))))
                check(name,fixture['id']+' tile-to-source probability parity',delta<=1e-6,'maximum absolute difference <= 1e-6',delta)
                labels=binary.BINARY_LABELS if name=='binary' else module.mangrove_type
                row=dict(id=fixture['id'],model=name,source=fixture['source'],variant=fixture['variant'],derived=fixture['derived'],
                         label=label,confidence=float(prob.max()),tiles=len(p),tile_inference_seconds=elapsed,
                         probability=prob.tolist(),tile_vote_agreement=float((p.argmax(1)==prob.argmax()).mean()),
                         truth=None,accuracy=None,interpretation='Unlabelled diagnostic; category output is conditional on mangrove input and is not a species claim.')
                predictions.append(row)
                for j,(position,pr) in enumerate(zip(positions,p)):
                    tiles_saved.append(dict(image_id=fixture['id'],model=name,tile_id=j,source=fixture['source'],box=list(position),probabilities=pr.tolist(),label=labels[int(pr.argmax())]))
        save('tile_predictions.json',tiles_saved);save('unlabelled_predictions.json',predictions)
        pd.DataFrame([{k:v for k,v in r.items() if k!='probability'} for r in predictions]).to_csv(OUT/'unlabelled_predictions.csv',index=False)
        # Actual localhost HTTP requests through all deployed models; observe render context.
        upload=temp/'uploads';upload.mkdir();module.app.config['UPLOAD_FOLDER']=str(upload)
        capture={}
        def observed(sender,template,context,**extra):capture.clear();capture.update(context)
        template_rendered.connect(observed,module.app,weak=False)
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        server=make_server('127.0.0.1',5053,module.app);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        session=requests.Session();base='http://127.0.0.1:5053';screenshot_url=None
        def post(path,name=None):
            capture.clear()
            with Path(path).open('rb') as handle:return session.post(base+'/analyse',files={'orthomosaic':(name or Path(path).name,handle)},timeout=240)
        try:
            for fixture in [x for x in fixtures if x['id'] in ['source_0_reference','small','edge','grayscale','rgba','full_resolution']]:
                direct=species.predict_combined(str(fixture['path']),module.binary_model,module.model,module.mangrove_type,module.gpu)
                repeats=3 if fixture['id']=='source_0_reference' else 1
                for repeat in range(repeats):
                    stop=threading.Event();rss=[proc.memory_info().rss]
                    def monitor():
                        while not stop.wait(.01):rss.append(proc.memory_info().rss)
                    watcher=threading.Thread(target=monitor,daemon=True);watcher.start();start=time.perf_counter()
                    try:response=post(fixture['path'],f'acceptance_{fixture["id"]}_{repeat}.png' if fixture['path'].suffix.lower()=='.png' else f'acceptance_{fixture["id"]}_{repeat}.jpg')
                    finally:elapsed=time.perf_counter()-start;stop.set();watcher.join()
                    actual=capture.get('result',{});observed_binary=actual.get('binary') or {}
                    delta=float(np.max(np.abs(np.array(direct['binary']['probs'])-observed_binary.get('probs',[999,999]))))
                    binary_ok=response.status_code==200 and direct['binary']['prediction']==observed_binary.get('prediction') and delta<=1e-6
                    check('binary',fixture['id']+f' HTTP parity #{repeat+1}',binary_ok,'HTTP 200; label and probabilities agree',dict(http=response.status_code,max_probability_error=delta,label=observed_binary.get('prediction')))
                    a=actual.get('multi_class');d=direct.get('multi_class')
                    if d:
                        diff=max(abs(d['probabilities'][key]-(a or {}).get('probabilities',{}).get(key,999)) for key in module.mangrove_type)
                        good=bool(a) and d['predicted_class']==a['predicted_class'] and diff<=1e-6
                    else:good=a is None;diff=None
                    check('species',fixture['id']+f' HTTP gating/parity #{repeat+1}',good,'Category output agrees, or absent when binary rejects',dict(category=(a or {}).get('predicted_class'),max_probability_error=diff))
                    text=BeautifulSoup(response.text,'html.parser').get_text(' ',strip=True)
                    check('both',fixture['id']+f' rendered result #{repeat+1}',('Analysis Results' in text and (not a or a['predicted_class'].title() in text)),'Expected results displayed',response.status_code)
                    latency.append(dict(id=fixture['id'],repeat=repeat+1,seconds=elapsed,peak_process_rss_mb=max(rss)/1024**2,
                                        width=Image.open(fixture['path']).width,height=Image.open(fixture['path']).height,kind='warm HTTP; all three models enabled'))
                    if fixture['id']=='source_0_reference' and repeat==0:screenshot_url=response.url
            # Controlled gate tests force branch inputs, not ground-truth accuracy.
            path=fixtures[0]['path']
            for label,confidence,expect_call in [('Non-Mangrove',.99,False),('Mangrove',.5,False),('Mangrove',.51,True),('Uncertain',0.,False)]:
                result=dict(prediction=label,confidence=confidence,probs=[1-confidence,confidence],n_tiles=1)
                with patch('binary_detector.predict_binary',return_value=result),patch('ml_model.predicting_test',wraps=species.predicting_test) as spy:
                    r=post(path);called=spy.call_count>0
                check('species',f'Upstream gate {label} {confidence}',r.status_code==200 and called==expect_call,expect_call,called,'controlled branch; real species weights')
            with patch.object(module,'binary_model',None),patch.object(module,'predict_mangrove',side_effect=AssertionError('Must not bypass binary gate')) as spy:
                r=post(path);actual=capture.get('result',{})
                check('species','Missing binary model prevents category inference',r.status_code==200 and actual.get('multi_class') is None and spy.call_count==0,'No classification without upstream detector',dict(http=r.status_code,called=spy.call_count),'controlled unavailable model')
            with patch.object(module,'model',None):
                r=post(path);actual=capture.get('result',{})
                check('species','Missing category model leaves binary result usable',r.status_code==200 and actual.get('binary') is not None and actual.get('multi_class') is None,'Binary only',r.status_code,'controlled unavailable model')
            for target,func in [(binary,'load_binary_model'),(species,'load_model')]:
                key='BINARY_MODEL_PATH' if target is binary else 'MODEL_PATH'
                with patch.object(target,key,temp/'missing.pth'):
                    try:getattr(target,func)();raised=False
                    except FileNotFoundError:raised=True
                check('binary' if target is binary else 'species','Missing checkpoint has explicit failure',raised,'FileNotFoundError',raised,'controlled missing path; real loader')
            for name,data,expected in [('corrupt.jpg',b'invalid','not a valid image'),('bad.txt',b'hello','Invalid file type')]:
                r=session.post(base+'/analyse',files={'orthomosaic':(name,data)},timeout=20)
                check('both',name+' rejected',r.status_code==200 and expected in r.text,expected,r.status_code)
            r=session.post(base+'/analyse',data={},timeout=20)
            check('both','Missing upload rejected',r.status_code==200 and 'No file part' in r.text,'No file part',r.status_code)
            with patch.object(module,'predict_combined',side_effect=RuntimeError('Injected classifier failure')):
                r=post(path);actual=capture.get('result',{})
                check('both','Inference exception leaves safe unavailable states',r.status_code==200 and actual.get('binary') is None and actual.get('multi_class') is None,'HTTP 200; no invented classifier result',r.status_code,'injected failure')
            chrome=Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
            if chrome.exists() and screenshot_url:
                subprocess.run([str(chrome),'--headless','--disable-gpu','--hide-scrollbars','--user-data-dir='+str(temp/'browser'),'--window-size=1440,1400','--screenshot='+str(OUT/'flask_result.png'),screenshot_url],capture_output=True,timeout=40,check=True)
        finally:
            server.shutdown();thread.join(timeout=5);template_rendered.disconnect(observed,module.app)
    pd.DataFrame(tests).to_csv(OUT/'application_checks.csv',index=False)
    pd.DataFrame(latency).to_csv(OUT/'latency_memory.csv',index=False)
    save('application_summary.json',dict(tests=len(tests),passed=sum(x['status']=='PASS' for x in tests),failed=[x for x in tests if x['status']=='FAIL'],
                                      temporary_files='TemporaryDirectory fixtures, uploads and browser profile removed after run.',
                                      memory='Process RSS sampled every 10 ms, all models and cache included; brief peaks may be missed.',
                                      timing='One cold process model-load measurement; HTTP cases warm after offline calls; 3 repetitions only for 1024x768 reference; other sizes n=1.',
                                      accuracy='Not measured: fixtures have no independent classifier labels. Passing technical checks does not satisfy data/accuracy rubric requirements.'))
    print(json.dumps(dict(checks=len(tests),passed=sum(x['status']=='PASS' for x in tests))),flush=True)
    if any(x['status']=='FAIL' for x in tests):raise RuntimeError('Application verification failed; inspect application_checks.csv')

if __name__=='__main__':run()
