"""Presentation-only notebook formatting; never executes model or training cells."""
import re
import nbformat
from bs4 import BeautifulSoup

CSS = '<style id="notebook-left-alignment">h1,h2,h3,h4,h5,h6,th {text-align:left !important;} .dataframe th {text-align:left !important;}</style>'
TAG = 'notebook-left-alignment'


def align_notebook(notebook):
    """Preserve calculations and output values while aligning rendered headings."""
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            # Inline styles also work in viewers which sanitize stylesheet blocks.
            cell.source = re.sub(
                r'^(#{1,6})\s+(.+?)\s*#*$',
                lambda m: f'<h{len(m[1])} style="text-align: left;">{m[2]}</h{len(m[1])}>',
                cell.source, flags=re.MULTILINE)
        for output in cell.get('outputs', []):
            html = output.get('data', {}).get('text/html')
            if html and TAG not in html:
                soup = BeautifulSoup(html, 'html.parser')
                for element in soup.select('h1,h2,h3,h4,h5,h6,th'):
                    style = re.sub(r'text-align\s*:[^;]+;?', '', element.get('style', ''))
                    element['style'] = style + ';text-align: left !important;'
                    element.attrs.pop('align', None)
                output['data']['text/html'] = str(soup)
    if not any(TAG in c.metadata.get('tags', []) for c in notebook.cells):
        cell = nbformat.v4.new_code_cell(
            'from IPython.display import HTML, display\ndisplay(HTML(' + repr(CSS) + '))',
            metadata={'tags': [TAG], 'jupyter': {'source_hidden': True}},
            outputs=[nbformat.v4.new_output('display_data', data={'text/html': CSS, 'text/plain': '<Notebook heading styles>'})])
        notebook.cells.insert(0, cell)
    return notebook
