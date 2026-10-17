"""Check the executed notebook, embedded assets and recorded acceptance evidence."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import nbformat

from src.utilities.runtime import write_json


def main():
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / 'Homework_5_DualVision.ipynb', as_version=4)
    nbformat.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == 'code']
    assert all(isinstance(cell.execution_count, int) for cell in code)
    assert not any(output.output_type == 'error' for cell in code for output in cell.outputs)
    displayed_pngs = {hashlib.sha256(base64.b64decode(output['data']['image/png'])).hexdigest()
                     for cell in code for output in cell.outputs
                     if 'image/png' in output.get('data', {})}
    selected = {'mnist': 'accuracy-mnist-997-20261017',
                'cifar10': 'accuracy-max-20261017'}
    expected_pngs = {hashlib.sha256(path.read_bytes()).hexdigest()
                    for name in ['mnist', 'cifar10']
                    for path in (root / 'results' / selected[name] / name).glob('*.png')}
    assert expected_pngs <= displayed_pngs, 'Notebook plots must match the saved measured figures exactly.'
    assert not any('from src.' in cell.source or 'import src' in cell.source for cell in code)
    assignments = {}
    for cell in code:
        for node in ast.parse(cell.source).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments[target.id] = node.value.value
    assert assignments['ALLOW_TRAINING'] is False and assignments['DOWNLOAD_DATA'] is False
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(assignments['SUBMISSION_ARCHIVE_B64']))) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        assert set(archive.namelist()) == set(manifest) | {'manifest.json'}
        for name, digest in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    cpu = json.loads((root / 'results/cpu_notebook_verification.json').read_text())
    assert cpu['device'] == 'cpu' and not cpu['cuda_visible']
    assert cpu['embedded_assets_only'] and cpu['no_dataset_directory'] and cpu['no_retraining']
    assert cpu['real_gradcam_cpu_verified']
    assert all(cpu['widget_callbacks'][name]['passed'] for name in ['mnist', 'cifar10'])
    audit = json.loads((root / 'results/real_experiment_audit.json').read_text())
    assert all(audit[name]['test_count'] == 10000 for name in ['mnist', 'cifar10'])
    cuda = json.loads((root / 'results/cuda_notebook_verification.json').read_text())
    assert cuda['device'] == 'cuda' and cuda['cuda_visible']
    assert cuda['automatic_device_selection'] and cpu['automatic_device_selection']
    assert cuda['embedded_assets_only'] and cuda['no_dataset_directory'] and cuda['no_retraining']
    assert cuda['real_gradcam_cuda_verified']
    assert all(cuda['widget_callbacks'][name]['passed'] for name in ['mnist', 'cifar10'])
    tests = ET.parse(root / 'results/portability_tests.xml').getroot()
    assert all(int(suite.get('failures', 0)) == int(suite.get('errors', 0)) == 0
               for suite in tests.iter('testsuite'))
    tracked = subprocess.check_output(['git', 'ls-files'], cwd=root, text=True).splitlines()
    assert not any(path.endswith(('.pt', '.pth')) or
                   (path.startswith('data/') and path != 'data/.gitkeep') for path in tracked)
    evidence = {'technical_submission_ready': True, 'executed_code_cells': len(code),
                'automatic_cpu_and_cuda_notebooks_verified': True,
                'embedded_assets_verified': len(manifest), 'notebook_error_outputs': 0,
                'displayed_report_figures_match_saved_pngs': len(expected_pngs),
                'cpu_without_datasets_or_retraining': True, 'both_upload_callbacks_verified': True,
                'real_gradcam_cpu_verified': True, 'test_metrics_and_figures_audited': True,
                'no_datasets_or_standalone_weights_tracked_by_git': True,
                'student_name': 'Zanyar Dawd',
                'personal_details_pending': any(marker in notebook.cells[0].source for marker in
                    ['[Your name]', '[Your ID]', '[Course name]', '[Instructor name]', '[Date]'])}
    write_json(root / 'results/submission_verification.json', evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
