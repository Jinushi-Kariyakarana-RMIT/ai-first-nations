"""Execute using this Python environment, avoiding a different global Jupyter."""
import json
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evaluations/tree_counter_hd'
path=ROOT/'evaluations/notebooks/YOLO_Tree_Counter_Evaluation.ipynb'
notebook=nbformat.read(path,as_version=4)
manager=KernelManager(kernel_name='python3')
manager.kernel_spec.argv[0]=sys.executable
client=NotebookClient(notebook,km=manager,timeout=1200,resources={'metadata':{'path':str(ROOT)}})
client.execute()
nbformat.validate(notebook)
nbformat.write(notebook,path)
html,_=HTMLExporter().from_notebook_node(notebook)
(OUT/'YOLO_Tree_Counter_Evaluation.html').write_text(html)
codes=[c for c in notebook.cells if c.cell_type=='code']
status=dict(kernel_python=sys.executable,code_cells=len(codes),
            executed_cells=sum(c.execution_count is not None for c in codes),
            error_outputs=sum(o.output_type=='error' for c in codes for o in c.outputs),
            plot_outputs=sum('image/png' in o.get('data',{}) for c in codes for o in c.outputs))
(OUT/'notebook_execution.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status,indent=2))
