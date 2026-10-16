"""Training, early stopping and strict state_dict checkpoints."""
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from src.utilities.runtime import IMPLEMENTATION_VERSION, write_json
from src.utilities.hardware import TrainingGuard, TrainingBudgetExceeded


def epoch_pass(model, loader, device, optimizer=None, guard=None) -> dict:
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            if guard is not None:
                guard.check()
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss; experiment aborted.')
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            count += labels.numel()
            total_loss += loss.item() * labels.numel()
            correct += int((logits.argmax(1) == labels).sum().item())
    if count == 0:
        raise ValueError('Empty data loader.')
    return {'loss': total_loss / count, 'accuracy': correct / count, 'count': count}


def save_checkpoint(path: Path, model, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save({'schema': 1, 'state_dict': state, 'metadata': metadata}, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, model, device, expected: dict | None = None) -> dict:
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema') != 1 or not isinstance(payload.get('metadata'), dict):
        raise ValueError('Unsupported checkpoint format.')
    meta = payload['metadata']
    if meta.get('implementation') != IMPLEMENTATION_VERSION:
        raise ValueError('Checkpoint implementation version mismatch; use a new experiment directory.')
    for key, value in (expected or {}).items():
        if meta.get(key) != value:
            raise ValueError(f'Checkpoint metadata mismatch: {key}')
    model.load_state_dict(payload['state_dict'], strict=True)
    model.to(device).eval()
    return meta


def mini_batch_check(model, loader, device) -> None:
    """Test a disposable copy, preserving the initialization for the real experiment."""
    candidate = copy.deepcopy(model).to(device)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    candidate.train()
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    logits = candidate(images)
    assert logits.shape == (len(labels), 10)
    loss = nn.functional.cross_entropy(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    del candidate, optimizer


def fit(model, bundle, device, config: dict, checkpoint_path: Path, metadata: dict, resume=False) -> dict:
    """Select by validation accuracy, breaking ties with lower validation loss."""
    if config['epochs'] < 1 or config['patience'] < 1 or config['max_minutes'] <= 0:
        raise ValueError('Epochs, patience and time budget must be positive.')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    history, best_key, bad_epochs = [], (-1.0, float('-inf')), 0
    prior_seconds, first_epoch = 0.0, 1
    best_metadata = None
    resume_path = checkpoint_path.parent / 'resume.pt'
    if resume:
        best_metadata = load_checkpoint(checkpoint_path, model, device, {'dataset': metadata['dataset']})
        if best_metadata.get('training_status') == 'complete':
            raise ValueError('A completed experiment must not be resumed after test evaluation.')
        if resume_path.exists():
            state = torch.load(resume_path, map_location='cpu', weights_only=True)
            assert state['config'] == config and state['dataset'] == metadata['dataset']
            model.load_state_dict(state['state_dict'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            history, bad_epochs = state['history'], state['bad_epochs']
            prior_seconds = state['training_seconds']
            torch.set_rng_state(state['torch_rng'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state_all(state['cuda_rng'])
            bundle.train.generator.set_state(state['loader_rng'])
            random.setstate(state['python_rng'])
            np_state = state['numpy_rng']
            np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
            metadata = state['metadata']
            telemetry_path = checkpoint_path.parent / 'gpu_telemetry.jsonl'
            if telemetry_path.exists():
                samples = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
                prior_seconds = max(prior_seconds, max(row['elapsed_seconds'] for row in samples)) + 5.0
            print(f'Resuming complete optimizer/RNG state after epoch {history[-1]["epoch"]}.', flush=True)
        else:
            # Legacy best-only checkpoints cannot recover AdamW momentum or the exact RNG stream.
            history = list(best_metadata['history'])
            prior_seconds = sum(row['seconds'] for row in history)
            telemetry_path = checkpoint_path.parent / 'gpu_telemetry.jsonl'
            if telemetry_path.exists():
                samples = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
                prior_seconds = max(prior_seconds, max(row['elapsed_seconds'] for row in samples))
            prior_seconds += 5.0  # conservative allowance for unsampled work before interruption
            finished_epoch = history[-1]['epoch']
            # Continue the original cosine schedule, but explicitly restart optimizer moments.
            for group in optimizer.param_groups:
                group['initial_lr'] = config['lr']
                group['lr'] = config['lr'] * (1 + math.cos(math.pi * (finished_epoch - 1) / config['epochs'])) / 2
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config['epochs'], last_epoch=finished_epoch - 1)
            metadata = {**metadata, 'recovery': {'method': 'best_weights_with_fresh_AdamW_moments_and_reseeded_RNG',
                'after_epoch': finished_epoch, 'prior_seconds_including_5s_allowance': prior_seconds,
                'limitation': 'Not an exact uninterrupted run; original cosine epoch schedule continues.'}}
            metadata['training_duration_note'] = 'Cumulative active training time, including a conservative 5-second interruption allowance; excludes the inactive gap.'
            best_metadata.update(recovery=metadata['recovery'])
            print(f'Recovering weights after epoch {finished_epoch}; optimizer moments restart, documented in metadata.', flush=True)
        first_epoch = history[-1]['epoch'] + 1
        selected = max(history, key=lambda row: (row['validation_accuracy'], -row['validation_loss']))
        best_key = (selected['validation_accuracy'], -selected['validation_loss'])
        bad_epochs = history[-1]['epoch'] - selected['epoch']
        if prior_seconds >= config['max_minutes'] * 60:
            raise RuntimeError('No training budget remains for resumption.')
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    stop_reason = 'epoch_limit'
    guard = TrainingGuard(checkpoint_path.parent / 'gpu_telemetry.jsonl', config['max_minutes'] - prior_seconds / 60,
                          elapsed_offset=prior_seconds) if device.type == 'cuda' and not metadata.get('smoke', False) else None
    for epoch in range(first_epoch, config['epochs'] + 1):
        epoch_start = time.perf_counter()
        lr = optimizer.param_groups[0]['lr']
        try:
            if guard is not None:
                train_metrics = epoch_pass(model, bundle.train, device, optimizer, guard=guard)
                validation_metrics = epoch_pass(model, bundle.validation, device, guard=guard)
            else:
                train_metrics = epoch_pass(model, bundle.train, device, optimizer)
                validation_metrics = epoch_pass(model, bundle.validation, device)
        except (RuntimeError, KeyboardInterrupt) as error:
            # Preserve current weights and optimizer state without replacing the best model.
            progress_meta = {**metadata, 'training_status': 'interrupted', 'interrupted_epoch': epoch,
                             'history': history, 'training_seconds': prior_seconds + time.perf_counter() - started,
                             'stop_reason': str(error), 'telemetry': guard.summary() if guard else None}
            progress_path = checkpoint_path.parent / 'interrupted.pt'
            save_checkpoint(progress_path, model, progress_meta)
            optimizer_path = checkpoint_path.parent / 'optimizer_interrupted.pt'
            torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                        'epoch': epoch, 'torch_rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                        'loader_rng': bundle.train.generator.get_state()}, optimizer_path)
            write_json(checkpoint_path.parent / 'interruption.json', progress_meta)
            if isinstance(error, TrainingBudgetExceeded) and best_metadata is not None:
                stop_reason = 'time_budget_at_batch_boundary'
                break
            raise
        scheduler.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        row = {'epoch': epoch, 'lr': lr, 'seconds': time.perf_counter() - epoch_start,
               'train_loss': train_metrics['loss'], 'train_accuracy': train_metrics['accuracy'],
               'validation_loss': validation_metrics['loss'], 'validation_accuracy': validation_metrics['accuracy']}
        history.append(row)
        key = (row['validation_accuracy'], -row['validation_loss'])
        if key > best_key:
            best_key, bad_epochs = key, 0
            best_metadata = {**metadata, 'best_epoch': epoch, 'best_validation_accuracy': key[0],
                             'history': list(history), 'training_status': 'in_progress'}
            save_checkpoint(checkpoint_path, model, best_metadata)
        else:
            bad_epochs += 1
        write_json(checkpoint_path.parent / 'history.json', history)
        # Atomic epoch-boundary recovery: weights, AdamW moments, schedule and random streams.
        np_state = np.random.get_state()
        recovery_state = {'state_dict': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'history': list(history), 'bad_epochs': bad_epochs, 'config': config, 'dataset': metadata['dataset'],
            'metadata': metadata, 'training_seconds': prior_seconds + time.perf_counter() - started,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            'loader_rng': bundle.train.generator.get_state(), 'python_rng': random.getstate(),
            'numpy_rng': (np_state[0], np_state[1].tolist(), *np_state[2:])}
        temporary = resume_path.with_suffix('.tmp')
        torch.save(recovery_state, temporary)
        temporary.replace(resume_path)
        print(f"Epoch {epoch:02d}/{config['epochs']} | train {row['train_accuracy']:.2%} | "
              f"validation {row['validation_accuracy']:.2%} | {row['seconds']:.1f}s", flush=True)
        if bad_epochs >= config['patience']:
            stop_reason = 'early_stopping'
            break
        if prior_seconds + time.perf_counter() - started >= config['max_minutes'] * 60:
            stop_reason = 'time_budget'
            break
    elapsed = prior_seconds + time.perf_counter() - started
    # Reload selected weights; preserve full history even when later epochs were worse.
    load_checkpoint(checkpoint_path, model, device, {'dataset': metadata['dataset']})
    best_metadata.update(history=history, training_status='complete', stop_reason=stop_reason,
                         training_seconds=elapsed,
                         peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0)
    if guard is not None:
        best_metadata['telemetry'] = guard.summary()
    save_checkpoint(checkpoint_path, model, best_metadata)
    return best_metadata
