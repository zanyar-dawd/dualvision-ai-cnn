"""Real saved-weight portability; no downloads, training, or benchmark reevaluation."""
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.cnn import make_model
from src.training.engine import load_checkpoint
from src.utilities import runtime
from src.utilities.inference import predict_file
from src.visualization.gradcam import grad_cam


def test_broken_cuda_falls_back_and_explicit_request_fails(monkeypatch):
    original = runtime.verify_device_operation
    calls = []

    def probe(device):
        calls.append(device.type)
        if device.type == 'cuda':
            raise RuntimeError('simulated CUDA driver failure')
        original(device)

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(runtime, 'verify_device_operation', probe)
    with pytest.warns(RuntimeWarning, match='using CPU'):
        device, evidence = runtime.select_device()
    assert device.type == 'cpu' and calls == ['cuda', 'cpu']
    assert 'simulated' in evidence['fallback_reason'] and not evidence['cuda_verified']
    with pytest.raises(RuntimeError, match='Explicit CUDA verification failed'):
        runtime.select_device('cuda')
    calls.clear()
    device, evidence = runtime.select_device('cpu')
    assert device.type == 'cpu' and calls == ['cpu']  # Override never probes CUDA.
    for name in ['mnist', 'cifar10']:
        with pytest.warns(RuntimeWarning, match='using CPU'):
            prediction = predict_file(Path('results/baseline') / name / 'examples/test_index_0.png',
                                      Path('checkpoints/baseline') / name / 'best.pt', name)
        assert prediction['device'] == 'cpu'
        assert np.isfinite(prediction['probabilities']).all()


@pytest.mark.parametrize('device_name', ['cpu', 'cuda'])
@pytest.mark.parametrize('name', ['mnist', 'cifar10'])
def test_real_checkpoint_location_inference_and_gradcam(name, device_name, monkeypatch):
    if device_name == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA not available on this machine.')
    device, record = runtime.select_device(device_name)
    assert record['tensor_operation_verified']
    checkpoint = Path('checkpoints/baseline') / name / 'best.pt'
    model = make_model(name)
    locations = []
    original = torch.load

    def capture(*args, **kwargs):
        locations.append(kwargs['map_location'])
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, 'load', capture)
    load_checkpoint(checkpoint, model, device)
    assert locations == [device]
    assert next(model.parameters()).device.type == device_name
    path = Path('results/baseline') / name / 'examples/test_index_0.png'
    actual = predict_file(path, checkpoint, name, device=device)
    cpu = predict_file(path, checkpoint, name, device='cpu')
    assert actual['device'] == device_name
    np.testing.assert_allclose(actual['probabilities'], cpu['probabilities'], atol=1e-5, rtol=1e-4)
    assert actual['top_classes'][0][0] == cpu['top_classes'][0][0]
    if name == 'cifar10':
        # Caller may supply a CPU tensor even when the model is on CUDA.
        heatmap, _, _ = grad_cam(model, actual['tensor'].unsqueeze(0))
        assert heatmap.shape == (32, 32) and np.isfinite(heatmap).all()


def test_auto_uses_verified_cuda_when_available():
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available on this machine.')
    device, evidence = runtime.select_device()
    assert device.type == 'cuda' and evidence['cuda_verified']
    assert evidence['gpu'] and evidence['fallback_reason'] is None
