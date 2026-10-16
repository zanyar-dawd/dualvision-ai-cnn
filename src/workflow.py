"""Explicit experiment orchestration shared with the self-contained notebook."""
from pathlib import Path

import pandas as pd
import torch
from src.models.cnn import make_model
from src.utilities.runtime import IMPLEMENTATION_VERSION, seed_everything, select_device, parameter_count, write_json, file_sha256
from src.utilities.data import prepare_data
from src.training.engine import fit, mini_batch_check, load_checkpoint
from src.evaluation.metrics import evaluate_once, load_evaluation
from src.visualization.plots import make_figures, observed_errors
from src.visualization.gradcam import grad_cam_figure

DEFAULT_CONFIGS = {
    'mnist': {'epochs': 15, 'batch_size': 128, 'lr': 0.001, 'weight_decay': 0.0001, 'patience': 5, 'max_minutes': 30},
    'cifar10': {'epochs': 50, 'batch_size': 64, 'lr': 0.001, 'weight_decay': 0.0005, 'patience': 12, 'max_minutes': 115},
}


def run_experiment(dataset: str, root=Path('.'), experiment_id='baseline', config=None,
                   allow_training=False, download=False, retrain=False, smoke=False,
                   device_preference='auto', seed=42, resume=False):
    if not experiment_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in experiment_id):
        raise ValueError('Use a simple experiment ID containing letters, digits, underscores or hyphens.')
    root = Path(root)
    config = dict(DEFAULT_CONFIGS[dataset] if config is None else config)
    if smoke:
        config.update(epochs=1, batch_size=8, patience=1, max_minutes=2)
        base = root / 'results' / 'smoke' / experiment_id
        checkpoint = base / dataset / 'best.pt'
    else:
        base = root / 'results' / experiment_id
        checkpoint = root / 'checkpoints' / experiment_id / dataset / 'best.pt'
    output = base / dataset
    if retrain and not allow_training:
        raise ValueError('Retraining requires allow_training=True.')
    if resume and (not allow_training or retrain):
        raise ValueError('Resuming requires allow_training=True and retrain=False.')
    if retrain and checkpoint.exists():
        raise FileExistsError('Preserve prior results: choose a new experiment_id for retraining.')
    if not checkpoint.exists() and not (allow_training or smoke):
        print(f'{dataset}: PENDING — no trained checkpoint. Enable training explicitly after approving downloads/runtime.')
        return None
    seed_everything(seed)
    device, runtime = select_device(device_preference)
    if checkpoint.exists() and not smoke and not allow_training:
        return load_saved_experiment(dataset, checkpoint, output, device, config, seed)
    bundle = prepare_data(dataset, root / 'data', config['batch_size'], seed,
                          download=download, smoke=smoke, pin_memory=device.type == 'cuda')
    model = make_model(dataset).to(device)
    expected = {'implementation': IMPLEMENTATION_VERSION, 'dataset': dataset, 'smoke': smoke,
                'config': config, 'seed': seed, 'classes': bundle.classes,
                'mean': bundle.mean, 'std': bundle.std, 'split_sha256': bundle.metadata['split_sha256']}
    if checkpoint.exists():
        metadata = load_checkpoint(checkpoint, model, device, expected)
        if metadata.get('training_status') != 'complete':
            if not resume:
                raise RuntimeError('Training was interrupted. Explicitly resume or use a new experiment ID; no test evaluation performed.')
            metadata = fit(model, bundle, device, config, checkpoint, metadata, resume=True)
        print(f'{dataset}: loaded selected checkpoint (epoch {metadata["best_epoch"]}).')
    else:
        mini_batch_check(model, bundle.train, device)
        # The check consumed RNG and a loader shuffle; reconstruct both before training.
        seed_everything(seed)
        bundle.train.generator.manual_seed(seed)
        model = make_model(dataset).to(device)
        metadata = {**expected, 'parameters': parameter_count(model), 'runtime': runtime,
                    'data_sizes': bundle.metadata['sizes'], 'num_workers': 0,
                    'precision': 'float32', 'augmentation': 'crop32-pad4+horizontal-flip' if dataset == 'cifar10' else 'none'}
        write_json(output / 'split.json', bundle.metadata)
        metadata = fit(model, bundle, device, config, checkpoint, metadata)
    write_json(output / 'training.json', metadata)
    evaluation = evaluate_once(model, bundle.test, device, bundle.classes, checkpoint, output,
                               bundle.metadata['source'])
    figures = make_figures(metadata, evaluation, bundle, output)
    if dataset == 'cifar10':
        image, _ = bundle.test.dataset[0]
        path = output / 'gradcam.png'
        grad_cam_figure(model, image.unsqueeze(0).to(device), bundle.mean, bundle.std, bundle.classes, path, smoke=smoke)
        figures.append(path)
    print(('SYNTHETIC SMOKE ONLY — ' if smoke else '') + f'{dataset}: accuracy={evaluation["accuracy"]:.2%}')
    print(observed_errors(evaluation, bundle.classes))
    return {'model': model, 'metadata': metadata, 'evaluation': evaluation, 'bundle': bundle,
            'figures': figures, 'output_dir': output, 'checkpoint': checkpoint}


def load_saved_experiment(dataset, checkpoint, output, device, config, seed=42):
    """Teacher mode: CPU-compatible weights and saved report, with no dataset access."""
    import json
    model = make_model(dataset)
    meta = load_checkpoint(checkpoint, model, device, {'dataset': dataset, 'smoke': False,
                            'config': config, 'seed': seed, 'training_status': 'complete'})
    evaluation = load_evaluation(output / 'evaluation.json', checkpoint, meta['classes'])
    names = ['learning_curves.png', 'confusion_matrix.png', 'correct_samples.png', 'mistakes.png']
    if dataset == 'cifar10':
        names.append('gradcam.png')
    figures = [output / name for name in names]
    for path in figures:
        if not path.is_file():
            raise FileNotFoundError(f'Missing saved report figure: {path}')
    manifest = output / 'artifact_manifest.json'
    if manifest.exists():
        hashes = json.loads(manifest.read_text(encoding='utf-8'))
        for name, digest in hashes.items():
            if file_sha256(output / name) != digest:
                raise ValueError(f'Saved artifact integrity check failed: {name}')
    print(f'{dataset}: loaded trained epoch {meta["best_epoch"]} on {device}; using saved test results, no training or dataset access.')
    return {'model': model, 'metadata': meta, 'evaluation': evaluation, 'bundle': None,
            'figures': figures, 'output_dir': output, 'checkpoint': checkpoint}


def comparison_table(experiments: dict) -> pd.DataFrame:
    rows = []
    for dataset, experiment in experiments.items():
        if experiment is None:
            rows.append({'dataset': dataset, 'status': 'PENDING — not trained'})
            continue
        meta, result = experiment['metadata'], experiment['evaluation']
        rows.append({'dataset': dataset, 'status': 'SYNTHETIC SMOKE ONLY' if meta['smoke'] else 'measured test results',
                     'parameters': meta['parameters'], 'test_accuracy': result['accuracy'],
                     'macro_precision': result['macro_precision'], 'macro_recall': result['macro_recall'],
                     'macro_f1': result['macro_f1'], 'training_seconds': meta['training_seconds']})
    return pd.DataFrame(rows)
