"""Audit saved real results against checkpoints, histories and telemetry without re-testing."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.evaluation.metrics import load_evaluation
from src.models.cnn import make_model
from src.training.engine import load_checkpoint
from src.utilities.runtime import parameter_count, write_json


def main():
    evidence = {}
    selected_experiments = {'mnist': 'accuracy-mnist-997-20261017',
                            'cifar10': 'accuracy-max-20261017'}
    for name in ['mnist', 'cifar10']:
        experiment_id = selected_experiments[name]
        folder = Path('results') / experiment_id / name
        checkpoint = Path('checkpoints') / experiment_id / name / 'best.pt'
        stored = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model = make_model(name, stored['metadata'].get('architecture'))
        meta = load_checkpoint(checkpoint, model, 'cpu', {'dataset': name, 'smoke': False, 'training_status': 'complete'})
        training_json = json.loads((folder / 'training.json').read_text())
        assert meta == training_json
        metrics = load_evaluation(folder / 'evaluation.json', checkpoint, meta['classes'])
        assert metrics['count'] == 10000 and meta['runtime']['device'] == 'cuda'
        assert parameter_count(model) == meta['parameters']
        history = meta['history']
        selected_epoch = max(history, key=lambda row: (row['validation_accuracy'], -row['validation_loss']))
        assert selected_epoch['epoch'] == meta['best_epoch']
        curves = pd.read_csv(folder / 'history.csv')
        for key in ['epoch', 'train_accuracy', 'validation_accuracy', 'train_loss', 'validation_loss']:
            np.testing.assert_allclose(curves[key], [row[key] for row in history], atol=1e-12)
        per_class = pd.read_csv(folder / 'per_class_metrics.csv', index_col=0)
        for label in meta['classes']:
            for key in ['precision', 'recall', 'f1-score', 'support']:
                assert np.isclose(per_class.loc[label, key], metrics['report'][label][key])
        telemetry = [json.loads(line) for line in (checkpoint.parent / 'gpu_telemetry.jsonl').read_text().splitlines()]
        peak_c = max(row['temperature_c'] for row in telemetry)
        peak_memory = max(row['memory_used_mib'] for row in telemetry)
        assert peak_c < 85
        if meta.get('telemetry'):
            assert peak_c == meta['telemetry']['peak_temperature_c']
            assert peak_memory == meta['telemetry']['peak_device_memory_used_mib']
        assert all(row['memory_used_mib'] < row['memory_total_mib'] * 0.95 for row in telemetry)
        figures = list(folder.glob('*.png'))
        assert len(figures) >= (5 if name == 'cifar10' else 4)
        for figure in figures:
            with Image.open(figure) as image:
                image.verify()
        evidence[name] = {'accuracy': metrics['accuracy'], 'test_loss': metrics['loss'],
                          'test_count': metrics['count'], 'selected_epoch': meta['best_epoch'],
                          'completed_epochs': len(history), 'training_seconds': meta['training_seconds'],
                          'peak_temperature_c': peak_c, 'peak_device_memory_mib': peak_memory,
                          'checkpoint_metadata_matches': True, 'selection_uses_validation_only': True,
                          'prediction_metrics_match': True, 'curve_csv_matches_history': True,
                          'class_report_csv_matches': True, 'pngs_decode': True}
    write_json(Path('results/real_experiment_audit.json'), evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
