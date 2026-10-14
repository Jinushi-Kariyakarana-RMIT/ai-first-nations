"""Keep historical evidence distinguishable from current dataset availability."""
import nbformat

TAG = 'current-data-layout'
TEXT = '''> **Current dataset status (14 October 2026):** Mourilyan photos are outside the repository for manual application trials. References to the 527-photo pool below are historical evidence, not an active internal test or annotation dataset. Numerical results are preserved. See [current data availability](../CURRENT_DATA_LAYOUT.md).'''

def apply_layout_notice(notebook):
    if not any(TAG in cell.metadata.get('tags', []) for cell in notebook.cells):
        notebook.cells.insert(0, nbformat.v4.new_markdown_cell(TEXT, metadata={'tags': [TAG]}))
