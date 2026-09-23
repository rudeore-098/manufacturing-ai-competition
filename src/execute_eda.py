"""Execute selected EDA notebooks with the current Python and validate outputs.

Usage: python src/execute_eda.py 2 3 4 5
Rendered charts for visual review are saved under the ignored .venv directory.
"""
import base64
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / '.venv/mplconfig')
os.environ['IPYTHONDIR'] = str(ROOT / '.venv/ipython')
os.environ['JUPYTER_RUNTIME_DIR'] = str(ROOT / '.venv/jupyter_runtime')
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

for topic in map(int, sys.argv[1:]):
    path = ROOT / f'notebooks/{topic:02d}_topic_{topic:02d}_eda.ipynb'
    nb = nbformat.read(path, as_version=4)
    km = KernelManager(kernel_name='python3')
    km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    print(f'Executing {path.name}', flush=True)
    client = NotebookClient(nb, km=km, timeout=180, resources={'metadata': {'path': str(ROOT / 'notebooks')}})
    try:
        client.execute()
    finally:
        if km.has_kernel:
            km.shutdown_kernel(now=True)
    nbformat.validate(nb)
    nbformat.write(nb, path)
    review = ROOT / f'.venv/eda_checks/topic{topic:02d}'
    review.mkdir(parents=True, exist_ok=True)
    charts = 0
    for cell in nb.cells:
        for out in cell.get('outputs', []):
            if out.output_type == 'error':
                raise RuntimeError(out.evalue)
            if out.output_type == 'stream':
                print(out.text[:2200], flush=True)
            if 'image/png' in out.get('data', {}):
                charts += 1
                (review / f'chart_{charts:02d}.png').write_bytes(base64.b64decode(out.data['image/png']))
    print(f'PASS: {sum(c.cell_type == "code" for c in nb.cells)} code cells, {charts} charts', flush=True)
