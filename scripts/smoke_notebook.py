"""Execute the standalone notebook without source modules or dataset downloads."""
import json
import os
import argparse
from pathlib import Path
import tempfile

import nbformat
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pending', action='store_true', help='Execute safe default mode and save the main pending report.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / 'Homework_5_DualVision.ipynb', as_version=4)
    nbformat.validate(notebook)
    temporary_root = root / '.downloads'
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dualvision-notebook-', dir=temporary_root) as temporary:
        sandbox = Path(temporary)
        # A private kernelspec avoids modifying the user's Jupyter installation.
        kernel_dir = sandbox / 'kernels' / 'dualvision-test'
        kernel_dir.mkdir(parents=True)
        os.environ['JUPYTER_CONFIG_DIR'] = str(sandbox / 'jupyter-config')
        os.environ['JUPYTER_DATA_DIR'] = str(sandbox / 'jupyter-data')
        os.environ['JUPYTER_RUNTIME_DIR'] = str(sandbox / 'jupyter-runtime')
        os.environ['IPYTHONDIR'] = str(sandbox / 'ipython')
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpecManager
        (kernel_dir / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}',
                     '--HistoryManager.hist_file=:memory:'],
            'display_name': 'DualVision isolated smoke', 'language': 'python',
            'env': {'DUALVISION_SMOKE': '0' if args.pending else '1', 'DUALVISION_ROOT': str(sandbox),
                    'MPLBACKEND': 'Agg', 'PYTHONPATH': '', 'IPYTHONDIR': str(sandbox / 'ipython'),
                    'JUPYTER_RUNTIME_DIR': str(sandbox / 'jupyter-runtime'),
                    'JUPYTER_CONFIG_DIR': str(sandbox / 'jupyter-config'),
                    'JUPYTER_DATA_DIR': str(sandbox / 'jupyter-data'),
                    'MPLCONFIGDIR': str(sandbox / 'matplotlib')},
        }), encoding='utf-8')
        manager = KernelSpecManager(kernel_dirs=[str(sandbox / 'kernels')])
        kernel = KernelManager(kernel_name='dualvision-test', kernel_spec_manager=manager,
                               connection_file=str(sandbox / 'connection.json'))
        client = NotebookClient(notebook, timeout=300, km=kernel,
                                resources={'metadata': {'path': str(sandbox)}})
        try:
            client.execute()
        finally:
            # A supplied KernelManager is caller-owned; explicitly shut it down
            # before deleting the temporary Windows SQLite history directory.
            if kernel.has_kernel:
                kernel.shutdown_kernel(now=False)
            if client.kc is not None:
                client.kc.stop_channels()
        if not args.pending:
            for name in ['mnist', 'cifar10']:
                result = sandbox / 'results' / 'smoke' / 'notebook-smoke' / name / 'evaluation.json'
                assert result.exists(), f'{name} evaluation was not executed.'
                assert json.loads(result.read_text())['source'] == 'SYNTHETIC_SMOKE_ONLY'
        destination = root / 'results' / 'smoke'
        destination.mkdir(parents=True, exist_ok=True)
        target = root / 'Homework_5_DualVision.ipynb' if args.pending else destination / 'Homework_5_DualVision.smoke.ipynb'
        nbformat.write(notebook, target)
    print('PASS: complete notebook executed from an empty directory; ' +
          ('default report saved with pending results.' if args.pending else 'both synthetic experiments finished.'))


if __name__ == '__main__':
    main()
