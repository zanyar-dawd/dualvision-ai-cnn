"""Execute the final notebook on CPU or CUDA with only embedded assets, no retraining."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                        help='Expected device; notebook stays in auto mode. CPU hides CUDA.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / 'Homework_5_DualVision.ipynb', as_version=4)
    original_count = len(notebook.cells)
    notebook.cells.append(nbformat.v4.new_code_cell('''
from datetime import datetime, timezone
assert DEVICE.type == EXPECTED_DEVICE
assert torch.cuda.is_available() == (EXPECTED_DEVICE == "cuda")
assert DEVICE_PREFERENCE == "auto"
assert not (PROJECT_ROOT / "data").exists(), "Default notebook must not download datasets."
assert not ALLOW_TRAINING and not DOWNLOAD_DATA
verification = {"device": str(DEVICE), "cuda_visible": torch.cuda.is_available(),
                "no_dataset_directory": True, "no_retraining": True, "widget_callbacks": {}}
for name in ["mnist", "cifar10"]:
    experiment = experiments[name]
    assert experiment is not None and experiment["metadata"]["training_status"] == "complete"
    assert next(experiment["model"].parameters()).device.type == EXPECTED_DEVICE
    image_path = experiment["output_dir"] / "examples" / "test_index_0.png"
    payload = image_path.read_bytes()
    dashboard.children[1].value = name
    dashboard.children[3].value = ({"name": image_path.name, "type": "image/png", "size": len(payload),
        "content": memoryview(payload), "last_modified": datetime.now(timezone.utc)},)
    dashboard.children[4].click()
    assert dashboard.last_error is None and dashboard.last_prediction is not None
    expected = predict_file(image_path, experiment["checkpoint"], name)
    assert expected["device"] == EXPECTED_DEVICE
    np.testing.assert_allclose(dashboard.last_prediction["probabilities"], expected["probabilities"], atol=1e-7)
    verification["widget_callbacks"][name] = {"passed": True, "prediction": expected["top_classes"][0]}
    if name == "cifar10":
        heatmap, target, probabilities = grad_cam(experiment["model"], expected["tensor"].unsqueeze(0))
        assert heatmap.shape == (32, 32) and np.isfinite(heatmap).all()
        assert 0 <= heatmap.min() <= heatmap.max() <= 1
        verification["real_gradcam_" + EXPECTED_DEVICE + "_verified"] = True
write_json(PROJECT_ROOT / "device_verification.json", verification)
print("PASS: both real models, upload callbacks, saved reports and automatic device selection on", DEVICE)
'''))
    notebook.cells[-1].source = f'EXPECTED_DEVICE = {args.device!r}\n' + notebook.cells[-1].source
    with tempfile.TemporaryDirectory(prefix=f'dualvision-{args.device}-', dir=root / '.downloads') as temporary:
        sandbox = Path(temporary)
        kernel_dir = sandbox / 'kernels' / 'dualvision-cpu'
        kernel_dir.mkdir(parents=True)
        (kernel_dir / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}',
                     '--HistoryManager.hist_file=:memory:'],
            'display_name': 'DualVision CPU verification', 'language': 'python',
            'env': {'CUDA_VISIBLE_DEVICES': '-1' if args.device == 'cpu' else '0', 'DUALVISION_DEVICE': 'auto',
                    'DUALVISION_ROOT': str(sandbox), 'DUALVISION_SMOKE': '0', 'PYTHONPATH': '',
                    'MPLBACKEND': 'Agg', 'IPYTHONDIR': str(sandbox / 'ipython'),
                    'JUPYTER_RUNTIME_DIR': str(sandbox / 'jupyter-runtime'),
                    'MPLCONFIGDIR': str(sandbox / 'matplotlib')},
        }), encoding='utf-8')
        kernel = KernelManager(kernel_name='dualvision-cpu', connection_file=str(sandbox / 'connection.json'),
                               kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(sandbox / 'kernels')]))
        client = NotebookClient(notebook, timeout=300, km=kernel,
                                resources={'metadata': {'path': str(sandbox)}})
        try:
            client.execute()
            evidence = json.loads((sandbox / 'device_verification.json').read_text())
            evidence.update(executed_code_cells=sum(c.cell_type == 'code' for c in notebook.cells[:original_count]),
                            source_modules_available=False, embedded_assets_only=True,
                            automatic_device_selection=True,
                            note='CPU test hides CUDA; CUDA test uses the real GPU. Installed wheel includes CUDA libraries.')
            (root / f'results/{args.device}_notebook_verification.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
            notebook.cells = notebook.cells[:original_count]
            # Widget uploads are only verification inputs, not persistent notebook state.
            nbformat.validate(notebook)
            if args.device == 'cpu':
                nbformat.write(notebook, root / 'Homework_5_DualVision.ipynb')
        finally:
            if kernel.has_kernel:
                # Graceful protocol shutdown closes the actual kernel behind Windows' venv launcher.
                kernel.shutdown_kernel(now=False)
            if client.kc is not None:
                client.kc.stop_channels()
    print(f'PASS: final notebook executed on {args.device} from an empty folder with embedded weights, no data and no retraining.')


if __name__ == '__main__':
    main()
