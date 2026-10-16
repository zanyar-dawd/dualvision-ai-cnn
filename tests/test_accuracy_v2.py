import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

from experiments.models_v2 import make_candidate
from experiments.data_v2 import CandidateData, candidate_transform, prepare_candidate_data
from experiments.train_v2 import fit_candidate, mixed_cross_entropy, scheduled_lr, run_candidate
from src.utilities.runtime import seed_everything, IMPLEMENTATION_VERSION
from src.utilities.data import image_transform
from src.visualization.gradcam import grad_cam
from src.training.engine import load_checkpoint

CONFIGS = json.loads(Path('configs/accuracy_v2.json').read_text())


@pytest.mark.parametrize('dataset,shape', [('mnist', (1, 28, 28)), ('cifar10', (3, 32, 32))])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_candidate_forward_optimizer_and_determinism(dataset, shape, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable.')
    seed_everything(42)
    model = make_candidate(dataset).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(2, *shape, device=device)
    before = next(model.parameters()).detach().clone()
    logits = model(inputs)
    assert logits.shape == (2, 10)
    loss = nn.functional.cross_entropy(logits, torch.tensor([1, 2], device=device))
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, next(model.parameters()))


def test_mixup_loss_and_warmup_cosine():
    logits = torch.tensor([[2., -1.], [0., 1.]], requires_grad=True)
    labels = torch.tensor([0, 1])
    other = labels.flip(0)
    expected = 0.25 * nn.functional.cross_entropy(logits, labels) + 0.75 * nn.functional.cross_entropy(logits, other)
    torch.testing.assert_close(mixed_cross_entropy(logits, labels, other, 0.25), expected)
    for config in CONFIGS.values():
        rates = [scheduled_lr(config, epoch) for epoch in range(1, config['epochs'] + 1)]
        assert rates[0] < rates[config['warmup_epochs']-1] == config['lr']
        assert rates[-1] < rates[-2] and min(rates) >= 0


def test_candidate_augmentations_and_no_test_data_access():
    seed_everything(42)
    for name, shape, mean, std in [('mnist', (1, 28, 28), [0.5], [0.25]),
                                  ('cifar10', (3, 32, 32), [0.5]*3, [0.25]*3)]:
        pixels = np.random.default_rng(42).integers(0, 255, (shape[1], shape[2], 3), dtype=np.uint8)
        image = Image.fromarray(pixels).convert('L' if name == 'mnist' else 'RGB')
        assert candidate_transform(name, mean, std)(image).shape == shape
        clean = image_transform(name, mean, std)
        torch.testing.assert_close(clean(image), clean(image))
    with patch('experiments.data_v2.datasets.MNIST', side_effect=AssertionError('Official data must not open in smoke mode')):
        bundle = prepare_candidate_data('mnist', Path('unused'), {**CONFIGS['mnist'], 'batch_size': 4}, smoke=True)
    assert not hasattr(bundle, 'test')
    assert bundle.metadata['source'] == 'SYNTHETIC_SMOKE_ONLY'


def test_wide_gradcam_identity_and_range():
    seed_everything(42)
    heatmap, target, probabilities = grad_cam(make_candidate('cifar10'), torch.randn(1, 3, 32, 32))
    assert heatmap.shape == (32, 32) and np.isfinite(heatmap).all()
    assert 0 <= heatmap.min() <= heatmap.max() <= 1
    assert probabilities.sum() == pytest.approx(1)


def test_candidate_complete_state_resume_and_immutable_run(tmp_path, monkeypatch):
    import experiments.train_v2 as training
    config = {**CONFIGS['mnist'], 'epochs': 3, 'warmup_epochs': 1, 'minimum_epochs': 1,
              'batch_size': 4, 'mixup_alpha': 0.2, 'mixup_probability': 1.0}
    seed_everything(42)
    images, labels = torch.randn(12, 1, 2, 2), torch.arange(12) % 10

    def bundle():
        data = TensorDataset(images, labels)
        loaders = [DataLoader(data, batch_size=4, shuffle=i == 0,
                             generator=torch.Generator().manual_seed(42+i)) for i in range(3)]
        return CandidateData(*loaders, [], [], [], {})

    def model():
        return nn.Sequential(nn.Flatten(), nn.Linear(4, 20), nn.ReLU(), nn.Dropout(0.2), nn.Linear(20, 10))

    meta = {'implementation': IMPLEMENTATION_VERSION, 'dataset': 'mnist', 'smoke': True}
    seed_everything(42)
    full = model()
    reference = fit_candidate(full, bundle(), torch.device('cpu'), config, tmp_path/'full', meta)
    seed_everything(42)
    interrupted = model()
    original = training.train_candidate_epoch
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('simulated process interruption')
        return original(*args, **kwargs)

    monkeypatch.setattr(training, 'train_candidate_epoch', fail_second)
    with pytest.raises(RuntimeError, match='simulated'):
        fit_candidate(interrupted, bundle(), torch.device('cpu'), config, tmp_path/'resume', meta)
    monkeypatch.setattr(training, 'train_candidate_epoch', original)
    with pytest.raises(FileExistsError):
        fit_candidate(interrupted, bundle(), torch.device('cpu'), config, tmp_path/'resume', meta)
    seed_everything(999)
    recovered = model()
    result = fit_candidate(recovered, bundle(), torch.device('cpu'), config, tmp_path/'resume', meta, resume=True)
    assert result['best_epoch'] == reference['best_epoch']
    for row, expected in zip(result['history'], reference['history']):
        for key in ['train_loss', 'validation_loss', 'clean_train_accuracy', 'lr']:
            assert row[key] == expected[key]
    loaded = model()
    load_checkpoint(tmp_path/'resume/best.pt', loaded, 'cpu')
    for key, weights in full.state_dict().items():
        torch.testing.assert_close(weights, recovered.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(weights, loaded.state_dict()[key], rtol=0, atol=0)
    assert not (tmp_path/'resume/evaluation.json').exists()


def test_baseline_cannot_be_overwritten():
    with pytest.raises(ValueError, match='never baseline'):
        run_candidate('mnist', CONFIGS['mnist'], experiment_id='baseline')
