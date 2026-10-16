import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from src.models.cnn import DigitNet, VisionNet, make_model
from src.utilities.runtime import seed_everything, select_device, IMPLEMENTATION_VERSION
from src.utilities.data import split_indices, channel_statistics, image_transform, prepare_data
from src.utilities.inference import prepare_user_image, predict_image
from src.training.engine import epoch_pass, save_checkpoint, load_checkpoint, fit
from src.evaluation.metrics import classification_metrics, evaluate_once
from src.visualization.gradcam import grad_cam
from src.workflow import run_experiment, DEFAULT_CONFIGS


@pytest.fixture(autouse=True)
def seeded():
    seed_everything(42)


@pytest.mark.parametrize('name,shape', [('mnist', (4, 1, 28, 28)), ('cifar10', (4, 3, 32, 32))])
@pytest.mark.parametrize('device_name', ['cpu', 'cuda'])
def test_forward_loss_optimizer_and_device(name, shape, device_name):
    if device_name == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable; CPU path still tested.')
    device, evidence = select_device(device_name)
    assert evidence['tensor_operation_verified']
    model = make_model(name).to(device)
    images = torch.randn(shape, device=device)
    targets = torch.arange(4, device=device)
    before = next(model.parameters()).detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    logits = model(images)
    assert logits.shape == (4, 10) and logits.device.type == device_name
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    assert not torch.equal(before, next(model.parameters()))


def test_eval_weighting_and_mode():
    model = torch.nn.Identity()
    logits = torch.tensor([[5., 0.], [0., 5.], [0., 5.], [0., 5.], [5., 0.]])
    labels = torch.tensor([0, 1, 0, 1, 1])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    result = epoch_pass(model, loader, torch.device('cpu'))
    assert result['accuracy'] == pytest.approx(3 / 5)
    assert result['loss'] == pytest.approx(torch.nn.functional.cross_entropy(logits, labels).item())
    assert not model.training


def test_checkpoint_roundtrip_and_metadata(tmp_path):
    model = DigitNet().eval()
    images = torch.randn(2, 1, 28, 28)
    with torch.inference_mode():
        expected = model(images)
    path = tmp_path / 'best.pt'
    meta = {'implementation': IMPLEMENTATION_VERSION, 'dataset': 'mnist'}
    save_checkpoint(path, model, meta)
    loaded = DigitNet()
    assert load_checkpoint(path, loaded, 'cpu', {'dataset': 'mnist'}) == meta
    with torch.inference_mode():
        torch.testing.assert_close(loaded(images), expected)
    with pytest.raises(ValueError, match='metadata mismatch'):
        load_checkpoint(path, loaded, 'cpu', {'dataset': 'cifar10'})


def test_split_and_training_only_statistics():
    labels = np.repeat(np.arange(10), 20)
    train, validation = split_indices(labels)
    train2, validation2 = split_indices(labels)
    np.testing.assert_array_equal(train, train2)
    np.testing.assert_array_equal(validation, validation2)
    assert set(train).isdisjoint(validation) and len(train) == 180 and len(validation) == 20
    raw = np.arange(200 * 4, dtype=np.uint8).reshape(200, 2, 2)
    mean, std = channel_statistics(raw, train, True)
    changed = raw.copy()
    changed[validation] = 255
    assert channel_statistics(changed, train, True) == (mean, std)
    assert mean[0] == pytest.approx((raw[train] / 255.).mean())
    assert std[0] == pytest.approx((raw[train] / 255.).std())


def test_transforms_and_upload_polarity():
    image = Image.new('L', (60, 80), 'white')
    ImageDraw.Draw(image).line([(28, 15), (28, 65)], fill='black', width=5)
    x, processed = prepare_user_image(image, 'mnist', [0.5], [0.25])
    inverse = Image.fromarray(255 - np.asarray(image))
    other, _ = prepare_user_image(inverse, 'mnist', [0.5], [0.25])
    torch.testing.assert_close(x, other)
    assert x.shape == (1, 28, 28) and processed.size == (28, 28)
    transparent = Image.new('RGBA', (60, 80), (0, 0, 0, 0))
    ImageDraw.Draw(transparent).line([(28, 15), (28, 65)], fill='white', width=5)
    transparent_tensor, _ = prepare_user_image(transparent, 'mnist', [0.5], [0.25])
    torch.testing.assert_close(x, transparent_tensor)
    with pytest.raises(ValueError, match='No visible digit'):
        prepare_user_image(Image.new('L', (30, 30), 'white'), 'mnist', [0.5], [0.25])
    rgb = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    transform = image_transform('cifar10', [0.5] * 3, [0.25] * 3)
    torch.testing.assert_close(transform(rgb), transform(rgb))
    assert not any(isinstance(op, (transforms.RandomCrop, transforms.RandomHorizontalFlip)) for op in transform.transforms)
    assert any(isinstance(op, transforms.RandomCrop) for op in image_transform('cifar10', [0.5]*3, [0.25]*3, True).transforms)
    upload, _ = prepare_user_image(rgb.convert('RGBA'), 'cifar10', [0.5] * 3, [0.25] * 3)
    assert upload.shape == (3, 32, 32)


def test_metrics_known_example():
    result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], ['a', 'b'])
    assert result['accuracy'] == 0.75
    assert result['confusion_matrix'] == [[1, 1], [0, 2]]
    assert result['macro_precision'] == pytest.approx((1 + 2/3) / 2)
    assert result['macro_recall'] == 0.75


def test_gradcam_class_specific_and_hook_cleanup():
    model = VisionNet().eval()
    image = torch.randn(1, 3, 32, 32)
    original_hooks = len(model.stage3._forward_hooks)
    heatmap, target, probabilities = grad_cam(model, image, 0)
    second, _, _ = grad_cam(model, image, 1)
    assert heatmap.shape == (32, 32) and target == 0
    assert np.isfinite(heatmap).all() and 0 <= heatmap.min() <= heatmap.max() <= 1
    assert probabilities.sum() == pytest.approx(1.0)
    assert not np.allclose(heatmap, second)
    assert len(model.stage3._forward_hooks) == original_hooks
    # Degenerate maps remain zero rather than being replaced with invented saliency.
    with torch.no_grad():
        model.head[-1].weight.zero_()
    zero, _, _ = grad_cam(model, image, 0)
    assert not zero.any()


def test_evaluation_cached_and_tampering_detected(tmp_path):
    model = DigitNet().eval()
    loader = DataLoader(TensorDataset(torch.randn(5, 1, 28, 28), torch.arange(5)), batch_size=2)
    checkpoint = tmp_path / 'best.pt'
    save_checkpoint(checkpoint, model, {'implementation': IMPLEMENTATION_VERSION})
    classes = [str(i) for i in range(10)]
    result = evaluate_once(model, loader, 'cpu', classes, checkpoint, tmp_path, 'test-fixture')
    def forbidden_forward(*args):
        raise AssertionError('Cached evaluation must not run the model again.')
    model.forward = forbidden_forward
    assert evaluate_once(model, loader, 'cpu', classes, checkpoint, tmp_path, 'test-fixture') == result
    result['accuracy'] = -1
    (tmp_path / 'evaluation.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='inconsistent'):
        evaluate_once(model, loader, 'cpu', classes, checkpoint, tmp_path, 'test-fixture')


def test_pending_does_not_download_or_train(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Pending mode must not access data.')
    monkeypatch.setattr('src.workflow.prepare_data', forbidden)
    assert run_experiment('mnist', root=tmp_path) is None
    with pytest.raises(ValueError, match='allow_training'):
        run_experiment('mnist', root=tmp_path, retrain=True)


def test_best_checkpoint_selection_early_stopping(tmp_path, monkeypatch):
    scores = iter([0.7, 0.8, 0.6, 0.5])
    def fake_epoch(model, loader, device, optimizer=None):
        if optimizer is not None:
            # Exercise scheduler ordering without spending epochs on synthetic learning.
            optimizer.zero_grad()
            sum(p.sum() for p in model.parameters()).backward()
            optimizer.step()
            return {'loss': 1., 'accuracy': 0.5, 'count': 4}
        return {'loss': 1., 'accuracy': next(scores), 'count': 4}
    monkeypatch.setattr('src.training.engine.epoch_pass', fake_epoch)
    model = DigitNet()
    bundle = prepare_data('mnist', tmp_path, 4, smoke=True)
    meta = {'implementation': IMPLEMENTATION_VERSION, 'dataset': 'mnist'}
    config = {**DEFAULT_CONFIGS['mnist'], 'epochs': 10, 'patience': 2}
    path = tmp_path / 'best.pt'
    result = fit(model, bundle, torch.device('cpu'), config, path, meta)
    assert result['best_epoch'] == 2 and len(result['history']) == 4
    assert result['stop_reason'] == 'early_stopping' and result['training_status'] == 'complete'


def test_configs_match_documented_defaults():
    assert json.loads(Path('configs/experiments.json').read_text()) == DEFAULT_CONFIGS


def test_automatic_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    device, record = select_device('auto')
    assert device.type == 'cpu' and record['tensor_operation_verified']
    with pytest.raises(RuntimeError, match='unavailable'):
        select_device('cuda')


def test_synthetic_weights_cannot_be_used_for_user_predictions():
    with pytest.raises(ValueError, match='real-data'):
        predict_image(DigitNet(), Image.new('L', (28, 28)),
                      {'smoke': True, 'training_status': 'complete'}, 'cpu')
