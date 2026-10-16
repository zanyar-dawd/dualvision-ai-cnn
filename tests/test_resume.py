"""A saved epoch boundary reproduces the uninterrupted CPU training trajectory."""
import pytest
import torch
import src.training.engine as engine
from src.models.cnn import DigitNet
from src.utilities.data import prepare_data
from src.utilities.runtime import IMPLEMENTATION_VERSION, seed_everything
from src.workflow import DEFAULT_CONFIGS


def test_epoch_checkpoint_restores_optimizer_scheduler_and_random_streams(tmp_path, monkeypatch):
    config = {**DEFAULT_CONFIGS['mnist'], 'epochs': 3, 'batch_size': 8}
    meta = {'implementation': IMPLEMENTATION_VERSION, 'dataset': 'mnist', 'smoke': True}
    seed_everything(42)
    reference = DigitNet()
    bundle = prepare_data('mnist', tmp_path, 8, smoke=True)
    uninterrupted = engine.fit(reference, bundle, torch.device('cpu'), config, tmp_path / 'full/best.pt', meta)
    seed_everything(42)
    interrupted = DigitNet()
    bundle = prepare_data('mnist', tmp_path, 8, smoke=True)
    original = engine.epoch_pass
    calls = 0

    def interrupt_second_epoch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('simulated external interruption')
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, 'epoch_pass', interrupt_second_epoch)
    path = tmp_path / 'resumed/best.pt'
    with pytest.raises(RuntimeError, match='simulated'):
        engine.fit(interrupted, bundle, torch.device('cpu'), config, path, meta)
    monkeypatch.setattr(engine, 'epoch_pass', original)
    seed_everything(999)  # resumption must restore the saved streams, not these new ones
    resumed = DigitNet()
    fresh_bundle = prepare_data('mnist', tmp_path, 8, smoke=True)
    result = engine.fit(resumed, fresh_bundle, torch.device('cpu'), config, path, meta, resume=True)
    assert result['best_epoch'] == uninterrupted['best_epoch']
    for actual, expected in zip(result['history'], uninterrupted['history']):
        for key in ['epoch', 'lr', 'train_loss', 'train_accuracy', 'validation_loss', 'validation_accuracy']:
            assert actual[key] == expected[key]
    for key, tensor in reference.state_dict().items():
        torch.testing.assert_close(tensor, resumed.state_dict()[key], rtol=0, atol=0)
