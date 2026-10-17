"""Bounded validation-only experiments. Final test evaluation is a separate action."""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import pandas as pd

from experiments.models_v2 import make_candidate
from experiments.data_v2 import prepare_candidate_data
from src.utilities.data import image_transform
from src.utilities.runtime import IMPLEMENTATION_VERSION, seed_everything, select_device, parameter_count, write_json, file_sha256
from src.utilities.hardware import TrainingGuard, TrainingBudgetExceeded
from src.training.engine import epoch_pass, save_checkpoint, load_checkpoint
from src.evaluation.metrics import evaluate_once
from src.visualization.plots import make_figures
from src.visualization.gradcam import grad_cam_figure
from src.visualization.plots import plt, save_fig
from src.utilities.inference import predict_image

CANDIDATE_VERSION = 'accuracy-v2.1'


def recipe_signature(config):
    return hashlib.sha256(json.dumps({'version': CANDIDATE_VERSION, 'config': config}, sort_keys=True).encode()).hexdigest()


def validation_target(config):
    """Return a frozen config target, with the environment variable kept for legacy runs."""
    value = config.get('target_validation_accuracy')
    if value is None:
        value = os.environ.get('DUALVISION_TARGET_VALIDATION', '')
    if value in (None, ''):
        return None
    target = float(value)
    if not 0.0 < target <= 1.0:
        raise ValueError('Target validation accuracy must be between 0 and 1.')
    return target


def candidate_optimizer(model, config):
    if config['optimizer'] == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    if config['optimizer'] == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'],
                               momentum=0.9, nesterov=True)
    raise ValueError('Unsupported optimizer.')


def scheduled_lr(config, epoch):
    warmup = config['warmup_epochs']
    if warmup and epoch <= warmup:
        return config['lr'] * epoch / warmup
    progress = (epoch - warmup - 1) / max(1, config['epochs'] - warmup)
    return config['lr'] * (1 + math.cos(math.pi * min(1, max(0, progress)))) / 2


def mixed_cross_entropy(logits, targets, other_targets, weight):
    return weight * F.cross_entropy(logits, targets) + (1 - weight) * F.cross_entropy(logits, other_targets)


def train_candidate_epoch(model, loader, optimizer, device, config, guard):
    model.train()
    total_loss, weighted_correct, count = 0.0, 0.0, 0
    for images, labels in loader:
        guard.check()
        images, labels = images.to(device), labels.to(device)
        other, weight = labels, 1.0
        if config['mixup_alpha'] > 0 and np.random.random() < config['mixup_probability']:
            weight = float(np.random.beta(config['mixup_alpha'], config['mixup_alpha']))
            order = torch.randperm(len(labels), device=device)
            images = weight * images + (1 - weight) * images[order]
            other = labels[order]
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = mixed_cross_entropy(logits, labels, other, weight)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite candidate training loss.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        prediction = logits.argmax(1)
        weighted_correct += weight * (prediction == labels).sum().item() + (1-weight) * (prediction == other).sum().item()
        total_loss += loss.item() * len(labels)
        count += len(labels)
    return {'loss': total_loss / count, 'accuracy': weighted_correct / count}


class CandidateGuard:
    def __init__(self, device, folder, minutes, prior=0.0):
        self.deadline = time.monotonic() + minutes * 60 - prior
        self.gpu = TrainingGuard(folder / 'gpu_telemetry.jsonl', minutes - prior / 60,
                                 elapsed_offset=prior) if device.type == 'cuda' else None

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TrainingBudgetExceeded('Candidate cumulative training budget exhausted.')
        if self.gpu:
            self.gpu.check()


def cpu_weights(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def rng_snapshot(loader, device=torch.device('cpu')):
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (state[0], state[1].tolist(), *state[2:]),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            'loader': loader.generator.get_state()}


def restore_rng(state, loader, device=torch.device('cpu')):
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda' and state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    loader.generator.set_state(state['loader'])


def atomic_torch_save(payload, path):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def fit_candidate(model, bundle, device, config, folder, metadata, resume=False):
    """The resume snapshot atomically commits both current and validation-best weights."""
    folder.mkdir(parents=True, exist_ok=True)
    if any((folder / name).exists() for name in ['best.pt', 'resume.pt', 'interrupted.pt']) and not resume:
        raise FileExistsError('Use a new experiment ID or explicitly resume; existing candidates are preserved.')
    model.to(device)
    optimizer = candidate_optimizer(model, config)
    history, best_state, best_key, best_epoch, bad_epochs = [], None, (-1.0, float('-inf')), None, 0
    prior = 0.0
    if resume:
        state = torch.load(folder / 'resume.pt', map_location='cpu', weights_only=True)
        if state['config'] != config or state['candidate_version'] != CANDIDATE_VERSION or state['dataset'] != metadata['dataset']:
            raise ValueError('Resume recipe/version/dataset mismatch.')
        for key in ['split_sha256', 'mean', 'std', 'architecture', 'recipe_sha256']:
            if state['metadata'].get(key) != metadata.get(key):
                raise ValueError(f'Resume data/recipe mismatch: {key}')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        history, best_state, best_epoch = state['history'], state['best_state'], state['best_epoch']
        best_key, bad_epochs, prior = tuple(state['best_key']), state['bad_epochs'], state['seconds']
        log = folder / 'gpu_telemetry.jsonl'
        if log.exists():
            prior = max(prior, max(json.loads(line)['elapsed_seconds'] for line in log.read_text().splitlines()))
        interruption = folder / 'interruption.json'
        if interruption.exists():
            prior = max(prior, json.loads(interruption.read_text())['seconds'])
        prior += 5.0  # budget allowance for any work after the last telemetry sample
        restore_rng(state['rng'], bundle.train, device)
        assert state['scheduler']['next_epoch'] == len(history) + 1
        target_accuracy = validation_target(config)
        if target_accuracy is not None and history and max(row['validation_accuracy'] for row in history) >= target_accuracy:
            model.load_state_dict(best_state)
            metadata = {**metadata, 'training_status': 'complete', 'test_status': 'not_evaluated',
                        'best_epoch': best_epoch, 'best_validation_accuracy': best_key[0],
                        'history': history, 'stop_reason': f'target_validation_accuracy_{target_accuracy:.4f}',
                        'training_seconds': prior, 'telemetry': None,
                        'training_accuracy_kind': 'mixed-label weighted agreement in train mode; clean_train is unmixed evaluation-mode accuracy',
                        'overfitting_claim': 'No guarantee of absent overfitting; inspect clean train/validation curves and final generalization.'}
            save_checkpoint(folder / 'best.pt', model, metadata)
            return metadata
    if prior >= config['max_minutes'] * 60:
        raise RuntimeError('No training budget remains; do not silently reset the budget.')
    started = time.perf_counter()
    guard = CandidateGuard(device, folder, config['max_minutes'], prior)
    stop_reason = 'epoch_limit'
    for epoch in range(len(history) + 1, config['epochs'] + 1):
        began = time.perf_counter()
        rate = scheduled_lr(config, epoch)
        for group in optimizer.param_groups:
            group['lr'] = rate
        try:
            train = train_candidate_epoch(model, bundle.train, optimizer, device, config, guard)
            validation = epoch_pass(model, bundle.validation, device, guard=guard)
            clean = epoch_pass(model, bundle.clean_train, device, guard=guard)
        except (RuntimeError, FloatingPointError, KeyboardInterrupt) as error:
            elapsed = prior + time.perf_counter() - started
            atomic_torch_save({'state_dict': cpu_weights(model), 'optimizer': optimizer.state_dict(),
                               'rng': rng_snapshot(bundle.train, device), 'interrupted_epoch': epoch,
                               'config': config, 'seconds': elapsed}, folder / 'interrupted.pt')
            write_json(folder / 'interruption.json', {'epoch': epoch, 'seconds': elapsed, 'reason': str(error),
                       'resume_policy': 'Resume the complete preceding epoch snapshot; partial epoch is not treated as complete.'})
            if isinstance(error, TrainingBudgetExceeded) and best_state is not None:
                stop_reason = 'time_budget'
                break
            raise
        row = {'epoch': epoch, 'lr': rate, 'seconds': time.perf_counter() - began,
               'train_loss': train['loss'], 'train_accuracy': train['accuracy'],
               'validation_loss': validation['loss'], 'validation_accuracy': validation['accuracy'],
               'clean_train_loss': clean['loss'], 'clean_train_accuracy': clean['accuracy'],
               'clean_accuracy_gap': clean['accuracy'] - validation['accuracy']}
        history.append(row)
        key = (row['validation_accuracy'], -row['validation_loss'])
        if key > best_key:
            best_key, best_epoch, best_state, bad_epochs = key, epoch, cpu_weights(model), 0
        else:
            bad_epochs += 1
        snapshot = {'candidate_version': CANDIDATE_VERSION, 'dataset': metadata['dataset'], 'config': config, 'metadata': metadata,
                    'model': cpu_weights(model), 'optimizer': optimizer.state_dict(), 'history': list(history),
                    'best_state': best_state, 'best_key': best_key, 'best_epoch': best_epoch, 'bad_epochs': bad_epochs,
                    'seconds': prior + time.perf_counter() - started, 'rng': rng_snapshot(bundle.train, device),
                    'scheduler': {'kind': 'warmup_cosine', 'next_epoch': epoch + 1}}
        atomic_torch_save(snapshot, folder / 'resume.pt')
        write_json(folder / 'history.json', history)
        print(f"Epoch {epoch:03d}/{config['epochs']} | clean train {clean['accuracy']:.2%} | "
              f"validation {validation['accuracy']:.2%} | gap {row['clean_accuracy_gap']:.2%} | {row['seconds']:.1f}s", flush=True)
        target_accuracy = validation_target(config)
        if target_accuracy is not None:
            if row['validation_accuracy'] >= target_accuracy:
                stop_reason = f'target_validation_accuracy_{target_accuracy:.4f}'
                break
        if epoch >= config['minimum_epochs'] and bad_epochs >= config['patience']:
            stop_reason = 'early_stopping'
            break
    if best_state is None:
        raise RuntimeError('No full validation epoch completed; no model may be tested.')
    model.load_state_dict(best_state)
    metadata = {**metadata, 'training_status': 'complete', 'test_status': 'not_evaluated', 'best_epoch': best_epoch,
                'best_validation_accuracy': best_key[0], 'history': history, 'stop_reason': stop_reason,
                'training_seconds': prior + time.perf_counter() - started,
                'telemetry': guard.gpu.summary() if guard.gpu else None,
                'training_accuracy_kind': 'mixed-label weighted agreement in train mode; clean_train is unmixed evaluation-mode accuracy',
                'overfitting_claim': 'No guarantee of absent overfitting; inspect clean train/validation curves and final generalization.'}
    save_checkpoint(folder / 'best.pt', model, metadata)
    write_json(folder / 'training.json', metadata)
    return metadata


def validation_diagnostics(metadata, output, filename='validation_diagnostics.png'):
    """Comparable evaluation-mode curves on fixed clean training and validation samples."""
    frame = pd.DataFrame(metadata['history'])
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / 'history.csv', index=False)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric in zip(axes, ['accuracy', 'loss']):
        axis.plot(frame.epoch, frame[f'clean_train_{metric}'], label='clean training subset (eval mode)')
        axis.plot(frame.epoch, frame[f'validation_{metric}'], label='validation (eval mode)')
        axis.axvline(metadata['best_epoch'], color='gray', linestyle='--', label='selected epoch')
        axis.set(xlabel='Epoch', ylabel=metric, title=f'Comparable {metric}')
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    figure.suptitle('SYNTHETIC CHECK ONLY' if metadata['smoke'] else 'Generalization diagnostics; no guarantee of absent overfitting')
    save_fig(figure, output / filename)


def run_candidate(dataset, config, experiment_id='accuracy-v2', device_preference='auto', smoke=False, resume=False):
    if experiment_id == 'baseline' or not experiment_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in experiment_id):
        raise ValueError('Use a new simple experiment ID, never baseline.')
    config = copy.deepcopy(config)
    if smoke:
        config.update(epochs=2, batch_size=4, warmup_epochs=1, minimum_epochs=1, max_minutes=3)
    folder = Path('results/smoke/accuracy-v2') / dataset if smoke else Path('checkpoints') / experiment_id / dataset
    output = folder if smoke else Path('results') / experiment_id / dataset
    if (output / 'evaluation.json').exists():
        raise RuntimeError('This candidate has already been tested; training it further is prohibited.')
    recipe_path = output / 'frozen_recipe.json'
    if recipe_path.exists():
        previous_recipe = json.loads(recipe_path.read_text())
        if previous_recipe['config'] != config or previous_recipe['candidate_version'] != CANDIDATE_VERSION:
            raise ValueError('Frozen recipe differs. Use a new experiment ID.')
    if (folder / 'training.json').exists():
        previous = json.loads((folder / 'training.json').read_text())
        if previous['config'] != config:
            raise ValueError('Existing completed candidate has a different recipe.')
        print(f'{dataset}: candidate already trained; use explicit final evaluation after validation review.')
        return previous
    seed_everything(config['seed'])
    device, runtime = select_device(device_preference)
    bundle = prepare_candidate_data(dataset, Path('data'), config, smoke)
    model = make_candidate(dataset).to(device)
    metadata = {'implementation': IMPLEMENTATION_VERSION, 'candidate_version': CANDIDATE_VERSION,
                'dataset': dataset, 'architecture': config['architecture'], 'config': config, 'smoke': smoke,
                'recipe_sha256': recipe_signature(config), 'runtime': runtime, 'parameters': parameter_count(model),
                'classes': bundle.classes, 'mean': bundle.mean, 'std': bundle.std,
                'split_sha256': bundle.metadata['split_sha256'], 'seed': config['seed'],
                'benchmark_reuse': 'The baseline test scores were previously inspected; no test feedback enters this follow-up training.'}
    write_json(output / 'split.json', bundle.metadata)
    write_json(output / 'frozen_recipe.json', metadata)
    result = fit_candidate(model, bundle, device, config, folder, metadata, resume)
    write_json(output / 'training.json', result)
    validation_diagnostics(result, output)
    print('Validation-only training complete. Official test results remain pending.', flush=True)
    return result


def evaluate_candidate(dataset, config, experiment_id, device_preference='auto'):
    if experiment_id == 'baseline' or not experiment_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in experiment_id):
        raise ValueError('Use a new simple experiment ID, never baseline.')
    checkpoint = Path('checkpoints') / experiment_id / dataset / 'best.pt'
    output = Path('results') / experiment_id / dataset
    device, _ = select_device(device_preference)
    model = make_candidate(dataset)
    meta = load_checkpoint(checkpoint, model, device, {'config': config, 'dataset': dataset,
        'candidate_version': CANDIDATE_VERSION, 'training_status': 'complete', 'smoke': False})
    # This is the only code path that opens the official test partition.
    constructor = datasets.MNIST if dataset == 'mnist' else datasets.CIFAR10
    test = constructor('data', train=False, download=False,
                       transform=image_transform(dataset, meta['mean'], meta['std']))
    assert len(test) == 10000
    loader = DataLoader(test, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    evaluation = evaluate_once(model, loader, device, meta['classes'], checkpoint, output, 'torchvision-official')
    bundle = SimpleNamespace(test=loader, classes=meta['classes'], mean=meta['mean'], std=meta['std'])
    make_figures(meta, evaluation, bundle, output)
    # Mixed-label agreement is not comparable to ordinary validation accuracy.
    # Replace the generic plot with explicitly labeled, unmixed eval-mode curves.
    validation_diagnostics(meta, output, filename='learning_curves.png')
    if dataset == 'cifar10':
        grad_cam_figure(model, test[0][0].unsqueeze(0).to(device), meta['mean'], meta['std'], meta['classes'], output / 'gradcam.png')
    write_json(output / 'final_evaluation_record.json', {'recipe_sha256': meta['recipe_sha256'],
               'checkpoint_sha256': file_sha256(checkpoint), 'test_accuracy': evaluation['accuracy'],
               'baseline_benchmark_previously_seen': True, 'no_absence_of_overfitting_guarantee': True})
    return evaluation


def predict_candidate_file(path, checkpoint, dataset, device='auto', polarity='auto'):
    device, _ = select_device(device)
    model = make_candidate(dataset)
    metadata = load_checkpoint(checkpoint, model, device, {'dataset': dataset, 'candidate_version': CANDIDATE_VERSION})
    return predict_image(model, path, metadata, torch.device(device), polarity)


def preflight(configs):
    """Brief synthetic CUDA benchmark; no dataset access or accuracy measurement."""
    records = {}
    seed_everything(42)
    device, runtime = select_device('auto')
    for name, config in configs.items():
        model = make_candidate(name).to(device)
        optimizer = candidate_optimizer(model, config)
        shape = (1, 28, 28) if name == 'mnist' else (3, 32, 32)
        images = torch.randn(config['batch_size'], *shape, device=device)
        labels = torch.arange(config['batch_size'], device=device) % 10
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        guard = CandidateGuard(device, Path('results/accuracy_v2_preflight') / name, 3)
        timings = []
        for index in range(7):
            guard.check()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), labels)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            if index >= 2:
                timings.append(time.perf_counter() - started)
        seconds = float(np.median(timings))
        train_size = 54000 if name == 'mnist' else 45000
        records[name] = {'device': str(device), 'parameters': parameter_count(model), 'batch_size': config['batch_size'],
                         'median_step_seconds': seconds, 'timed_steps': len(timings),
                         'compute_only_epoch_minutes': seconds * math.ceil(train_size / config['batch_size']) / 60,
                         'compute_only_full_schedule_hours': seconds * math.ceil(train_size / config['batch_size']) * config['epochs'] / 3600,
                         'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else 0,
                         'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20 if device.type == 'cuda' else 0,
                         'telemetry': guard.gpu.summary() if guard.gpu else None,
                         'note': 'Synthetic compute estimate only; excludes augmentation, loading, validation, diagnostics, checkpoint I/O and sustained clock variation.'}
        del model, optimizer, images, labels, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    write_json(Path('results/accuracy_v2_preflight.json'), {'runtime': runtime, 'models': records})
    print(json.dumps(records, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--train', action='store_true', help='Start real training only after runtime approval.')
    mode.add_argument('--evaluate', action='store_true', help='Explicit final test after validation-based model selection.')
    parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'both'], default='both')
    parser.add_argument('--experiment-id', default='accuracy-v2')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--config', type=Path, default=Path('configs/accuracy_v2.json'))
    args = parser.parse_args()
    configs = json.loads(args.config.read_text())
    if args.preflight:
        preflight(configs)
        return
    names = ['mnist', 'cifar10'] if args.dataset == 'both' else [args.dataset]
    for name in names:
        if args.evaluate:
            result = evaluate_candidate(name, configs[name], args.experiment_id, args.device)
            print(f'{name}: measured test accuracy {result["accuracy"]:.2%}')
        else:
            run_candidate(name, configs[name], args.experiment_id, args.device, args.smoke, args.resume)


if __name__ == '__main__':
    main()
