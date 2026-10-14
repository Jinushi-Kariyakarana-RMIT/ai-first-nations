"""Real HTTP upload tests, process-RSS sampling and notebook/app parity."""
import argparse
import contextlib
import importlib
import io
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import psutil
import requests
from bs4 import BeautifulSoup
from PIL import Image
from werkzeug.serving import make_server

from evaluation import OUT, PROJECT, BASELINE, samples, save_json, digest
from tree_counter import DEFAULT_MODEL_PATH,count_trees


def run(serve=False, current=False):
    evidence = OUT
    output = OUT / "deployment_validation" if current else OUT
    output.mkdir(exist_ok=True)
    sys.path.insert(0,str(PROJECT/'flask-application'))
    os.environ['FLASK_DEBUG']='false'
    os.environ['FLASK_SECRET_KEY']='local-evaluation-isolated-session'
    start=time.perf_counter()
    # Fail if startup unexpectedly tries training or downloading checkpoint data.
    with patch('ml_model.train_model',side_effect=AssertionError('Unexpected training')), \
         patch('torch.hub.download_url_to_file',side_effect=AssertionError('Unexpected download')), \
         patch('ultralytics.utils.downloads.safe_download',side_effect=AssertionError('Unexpected download')):
        module=importlib.import_module('app')
    startup=time.perf_counter()-start
    assert module.tree_counter_model is not None and module.model is not None and module.binary_model is not None
    import tree_counter_service
    settings = dict(tree_counter_service.TREE_COUNTER_SETTINGS) if current else dict(BASELINE)
    # Historical notebook tests explicitly reproduce the baseline in this isolated process.
    tree_counter_service.TREE_COUNTER_SETTINGS.update(settings)
    upload=output/'app_uploads';upload.mkdir(exist_ok=True)
    module.app.config['UPLOAD_FOLDER']=str(upload)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    server=make_server('127.0.0.1',5052,module.app)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base='http://127.0.0.1:5052'
    session=requests.Session();process=psutil.Process();tests=[];timings=[]
    def record(name,expected,observed,passed):
        tests.append(dict(test=name,expected=expected,observed=str(observed),status='PASS' if passed else 'FAIL'))
    record('Checkpoint startup without training/download','All three local models available',
           'All models loaded under download/training guards',True)
    # Representative small photo, familiar image, composite, and one fresh full-resolution drone image.
    groups=samples();challenge=json.loads((evidence/'challenge_manifest.json').read_text())
    fresh=sorted((PROJECT/'Mourilyan_Harbour_Fresh_Test_Images').rglob('*.JPG'))
    inputs=[('neon',PROJECT/groups['external'][0]['image']),('original',PROJECT/groups['test'][0]['image']),
            ('composite',PROJECT/challenge['composites'][1]['image'])]
    if fresh:inputs.append(('fresh_drone',fresh[len(fresh)//2]))
    screenshot_url=None
    for kind,path in inputs:
        repeats=3 if kind!='fresh_drone' else 1
        expected_overlay=output/f'{kind}_direct_overlay.jpg'
        direct=count_trees(path,model=module.tree_counter_model,overlay_path=expected_overlay,**settings)
        for repeat in range(repeats):
            filename=f'hd_{kind}_{repeat}{path.suffix.lower()}'
            stop=threading.Event();rss=[]
            def sample_memory():
                while not stop.wait(.01):rss.append(process.memory_info().rss)
            rss_start=process.memory_info().rss;monitor=threading.Thread(target=sample_memory,daemon=True);monitor.start()
            began=time.perf_counter()
            try:
                with path.open('rb') as handle:
                    response=session.post(base+'/analyse',files={'orthomosaic':(filename,handle)},timeout=180)
            finally:
                elapsed=time.perf_counter()-began;stop.set();monitor.join()
            soup=BeautifulSoup(response.text,'html.parser');element=soup.select_one('.tree-count-value')
            count=int(element.get_text(strip=True)) if element else None
            stored_name=response.url.rsplit('/',1)[-1]
            overlay_path=upload/(Path(stored_name).stem+'-tree-overlay.jpg')
            dimensions_match=overlay_path.exists() and Image.open(overlay_path).size==Image.open(path).size
            identical_overlay=overlay_path.exists() and digest(overlay_path)==digest(expected_overlay)
            passed=response.status_code==200 and count==direct['estimated_trees'] and dimensions_match and identical_overlay
            record(f'{kind} upload #{repeat+1}: count and overlay parity',
                   f"HTTP 200; count={direct['estimated_trees']}; exact rendered-overlay checksum",
                   dict(http=response.status_code,count=count,dimensions_match=dimensions_match,overlay_identical=identical_overlay),passed)
            timings.append(dict(kind=kind,repeat=repeat+1,width=Image.open(path).width,height=Image.open(path).height,
                                seconds=elapsed,process_peak_rss_mb=max(rss+[rss_start,process.memory_info().rss])/1024**2,
                                rss_growth_mb=(max(rss+[rss_start])-rss_start)/1024**2,
                                predicted=count,ground_truth_available=kind!='fresh_drone'))
            if kind=='original' and repeat==0:screenshot_url=response.url
    for filename,body,expected in [('bad.txt',b'hello','Invalid file type'),('corrupt.jpg',b'not-an-image','not a valid image')]:
        response=session.post(base+'/analyse',files={'orthomosaic':(filename,body)},timeout=30)
        record(filename+' rejection',expected,response.status_code,response.status_code==200 and expected in response.text)
    response=session.post(base+'/analyse',files={'orthomosaic':('oversized.jpg',b'0'*(50*1024*1024+1))},timeout=30)
    record('Actual 50 MB upload limit','Graceful size message',response.status_code,
           response.status_code==200 and 'Image is too large' in response.text)
    response=session.post(base+'/analyse',data={},timeout=30)
    record('Missing file','Graceful missing-file message',response.status_code,response.status_code==200 and 'No file part' in response.text)
    # Simulate unavailable path without moving/removing the user's actual weights.
    import tree_counter_service
    missing=OUT/'does_not_exist.pt'
    with patch.object(tree_counter_service,'TREE_COUNTER_MODEL_PATH',missing):
        try:tree_counter_service.load_tree_counter_model();missing_raised=False
        except FileNotFoundError:missing_raised=True
    record('Missing tree weights loader','Explicit FileNotFoundError',missing_raised,missing_raised)
    saved=module.tree_counter_model;module.tree_counter_model=None
    try:
        path=inputs[0][1]
        with path.open('rb') as handle:
            response=session.post(base+'/analyse',files={'orthomosaic':('missing_model.png',handle)},timeout=90)
        record('Unavailable tree model in web route','HTTP 200 with unavailable message',response.status_code,
               response.status_code==200 and 'Tree counting is not available' in response.text)
    finally:module.tree_counter_model=saved
    with patch.object(module,'predict_tree_count',side_effect=RuntimeError('Injected test failure')):
        with inputs[0][1].open('rb') as handle:
            response=session.post(base+'/analyse',files={'orthomosaic':('inference_failure.png',handle)},timeout=90)
        record('Tree inference exception','Graceful error display',response.status_code,
               response.status_code==200 and 'Tree counting could not be completed' in response.text)
    pd.DataFrame(tests).to_csv(output/'flask_acceptance.csv',index=False)
    pd.DataFrame(timings).to_csv(output/'flask_latency_memory.csv',index=False)
    summary = dict(configuration='deployed' if current else 'historical baseline',settings=settings,startup_seconds=startup,tests=len(tests),passed=sum(t['status']=='PASS' for t in tests),
              failed=[t for t in tests if t['status']=='FAIL'],screenshot_url=screenshot_url,model_sha256=digest(DEFAULT_MODEL_PATH),
              memory_method='Process RSS sampled every 10 ms; includes all model weights and allocator cache. May miss sub-10-ms peaks; not GPU memory.',
              timing_method='Local HTTP upload through rendered result, all three models enabled; direct parity inference warms the model first. Startup measured separately.',
              performance_target='Not agreed; measurements reported without pass/fail speed claim.',
              fresh_accuracy='No accuracy metric: fresh image has no reviewed reference labels.')
    (output/'flask_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    chrome=Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
    if chrome.exists() and screenshot_url:
        with (output/'browser_capture.log').open('w') as log:
            subprocess.run([str(chrome),'--headless','--disable-gpu','--no-first-run','--hide-scrollbars',
                            '--user-data-dir=/tmp/ai-first-nations-hd-browser','--window-size=1440,1200',
                            '--virtual-time-budget=3000','--screenshot='+str(output/'flask_result.png'),screenshot_url],
                           stdout=log,stderr=log,timeout=45,check=True)
    print(json.dumps(dict(tests=len(tests),passed=sum(t['status']=='PASS' for t in tests),screenshot_url=screenshot_url),indent=2),flush=True)
    if serve:
        print('Isolated evaluation server remains on http://127.0.0.1:5052',flush=True)
        while True:time.sleep(1)
    server.shutdown()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--serve',action='store_true');parser.add_argument('--current',action='store_true',help='Validate deployed settings and save separate evidence');args=parser.parse_args();run(args.serve,args.current)
