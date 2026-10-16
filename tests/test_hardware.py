import json
from unittest.mock import patch
import pytest
from src.utilities.hardware import TrainingGuard, HardwareStop, TrainingBudgetExceeded
import torch
from src.models.cnn import DigitNet
from src.utilities.data import prepare_data
from src.utilities.runtime import IMPLEMENTATION_VERSION
from src.training.engine import fit
from src.workflow import DEFAULT_CONFIGS


def test_telemetry_threshold_logs_before_stopping(tmp_path):
    guard = TrainingGuard(tmp_path / 'gpu.jsonl', 1)
    with patch('src.utilities.hardware.subprocess.check_output', return_value='85, 500, 4096, 99, 40\n'):
        with pytest.raises(HardwareStop, match='85 C'):
            guard.check()
    assert json.loads((tmp_path / 'gpu.jsonl').read_text())['temperature_c'] == 85


def test_budget_and_memory_stops(tmp_path):
    guard = TrainingGuard(tmp_path / 'gpu.jsonl', 1)
    with patch('src.utilities.hardware.subprocess.check_output', return_value='70, 4000, 4096, 99, 40\n'):
        with pytest.raises(HardwareStop, match='memory pressure'):
            guard.check()
    guard.deadline = 0
    with pytest.raises(TrainingBudgetExceeded):
        guard.check()


def test_hardware_stop_preserves_best_and_interrupted_state(tmp_path, monkeypatch):
    calls = 0
    def pass_then_stop(model, loader, device, optimizer=None):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise HardwareStop('simulated temperature stop')
        if optimizer is not None:
            optimizer.zero_grad()
            sum(p.sum() for p in model.parameters()).backward()
            optimizer.step()
        return {'loss': 1.0, 'accuracy': 0.5, 'count': 4}
    monkeypatch.setattr('src.training.engine.epoch_pass', pass_then_stop)
    path = tmp_path / 'best.pt'
    with pytest.raises(HardwareStop, match='temperature'):
        fit(DigitNet(), prepare_data('mnist', tmp_path, 4, smoke=True), torch.device('cpu'),
            {**DEFAULT_CONFIGS['mnist'], 'epochs': 3}, path,
            {'implementation': IMPLEMENTATION_VERSION, 'dataset': 'mnist'})
    best = torch.load(path, weights_only=True)
    interrupted = torch.load(tmp_path / 'interrupted.pt', weights_only=True)
    assert best['metadata']['best_epoch'] == 1
    assert interrupted['metadata']['interrupted_epoch'] == 2
    assert (tmp_path / 'optimizer_interrupted.pt').exists()
    assert json.loads((tmp_path / 'interruption.json').read_text())['training_status'] == 'interrupted'
