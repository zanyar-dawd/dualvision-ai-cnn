"""Exercise both real trained models through the upload button callbacks on CPU."""
import contextlib
from datetime import datetime, timezone
import io
from pathlib import Path
import numpy as np
from torchvision.datasets import MNIST, CIFAR10

from src.workflow import run_experiment
from src.utilities.inference import prediction_dashboard, predict_file
from src.utilities.runtime import write_json


def main():
    experiments = {name: run_experiment(name, device_preference='cpu') for name in ['mnist', 'cifar10']}
    assert all(experiments.values()), 'Both real models must be trained.'
    panel = prediction_dashboard(experiments, 'cpu')
    assert panel is not None
    choice, upload, button = panel.children[1], panel.children[3], panel.children[4]
    evidence = {}
    for name, constructor in [('mnist', MNIST), ('cifar10', CIFAR10)]:
        data = constructor('data', train=False, download=False)
        original, target = data[0]  # fixed index, chosen before inspecting prediction
        folder = Path('results/baseline') / name / 'examples'
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'test_index_0.png'
        original.save(path)
        payload = path.read_bytes()
        choice.value = name
        upload.value = ({'name': path.name, 'type': 'image/png', 'size': len(payload),
                         'content': memoryview(payload), 'last_modified': datetime.now(timezone.utc)},)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            button.click()
        assert panel.last_error is None, panel.last_error
        result = panel.last_prediction
        assert result is not None, buffer.getvalue()
        fallback = predict_file(path, experiments[name]['checkpoint'], name, device='cpu')
        np.testing.assert_allclose(result['probabilities'], fallback['probabilities'], rtol=1e-6, atol=1e-7)
        assert np.isclose(result['probabilities'].sum(), 1)
        result['processed_image'].save(folder / 'processed_test_index_0.png')
        evidence[name] = {'device': 'cpu', 'source': 'official_test_index_0',
                          'true_class': experiments[name]['metadata']['classes'][target],
                          'top_classes': result['top_classes'], 'widget_callback_passed': True,
                          'file_fallback_matches': True, 'probabilities_sum': float(result['probabilities'].sum())}
        write_json(folder.parent / 'prediction_demonstration.json', evidence[name])
        print(f'{name}: CPU upload callback and file fallback PASS; actual top prediction {result["top_classes"][0]}.')
    write_json(Path('results/inference_verification.json'), evidence)
    panel.close()


if __name__ == '__main__':
    main()
