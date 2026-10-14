"""Run verification and both notebooks with the project interpreter."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--refresh',action='store_true');args=parser.parse_args()
if args.refresh:subprocess.run([sys.executable,str(OUT/'evaluate.py')],cwd=ROOT,check=True)
env=json.loads((OUT/'environment.json').read_text())
for name,expected in env['source_hashes'].items():
 assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,f'Stale evidence: {name}; rerun --refresh'
for key,name in [('binary','bestBinary.pth'),('species','best_mangrove_model.pth')]:
 assert hashlib.sha256((ROOT/'flask-application'/name).read_bytes()).hexdigest()==env[key]['sha256'],'Checkpoint changed; rerun --refresh'
test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,text=True,capture_output=True)
(OUT/'automated_tests.txt').write_text(test.stdout+test.stderr)
assert test.returncode==0,'Automated verification failed'
(OUT/'verification.json').write_text(json.dumps({'automated_test_exit_code':test.returncode,'passed_tests':(test.stdout+test.stderr).count(' ... ok'),'meaning':'Known-answer metric and software tests, not measured model accuracy.'},indent=2)+'\n')
subprocess.run([sys.executable,str(OUT/'build_notebooks.py')],cwd=ROOT,check=True)
records=[]
for name in ['Binary_Mangrove_Evaluation','Mangrove_Species_Evaluation']:
 path=ROOT/'evaluations/notebooks'/f'{name}.ipynb';nb=nbformat.read(path,as_version=4)
 manager=KernelManager(kernel_name='python3');manager.kernel_spec.argv[0]=sys.executable
 NotebookClient(nb,km=manager,timeout=300,resources={'metadata':{'path':str(ROOT)}}).execute()
 nbformat.validate(nb);nbformat.write(nb,path)
 html,_=HTMLExporter().from_notebook_node(nb);(OUT/f'{name}.html').write_text(html)
 codes=[c for c in nb.cells if c.cell_type=='code']
 records.append(dict(notebook=name,code_cells=len(codes),executed=sum(c.execution_count is not None for c in codes),errors=sum(o.output_type=='error' for c in codes for o in c.outputs),figures=sum('image/png' in o.get('data',{}) for c in codes for o in c.outputs)))
(OUT/'notebook_execution.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))
