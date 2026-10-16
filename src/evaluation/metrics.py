"""Held-out evaluation with checkpoint-bound cached predictions."""
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from src.utilities.runtime import file_sha256, write_json


def load_evaluation(cache, checkpoint, classes, count=10000, source='torchvision-official'):
    """Validate saved benchmark evidence without loading or downloading datasets."""
    import json
    result = json.loads(Path(cache).read_text(encoding='utf-8'))
    if (result['checkpoint_sha256'] != file_sha256(checkpoint) or result['source'] != source
            or result['count'] != count or result['classes'] != classes):
        raise ValueError('Saved evaluation does not match checkpoint, dataset or classes.')
    checked = classification_metrics(result['labels'], result['predictions'], classes)
    for key in ['count', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'confusion_matrix', 'report']:
        if checked[key] != result[key]:
            raise ValueError(f'Saved evaluation is inconsistent: {key}')
    return result


def classification_metrics(labels, predictions, classes) -> dict:
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if len(labels) == 0 or labels.shape != predictions.shape:
        raise ValueError('Expected matching nonempty labels and predictions.')
    report = classification_report(labels, predictions, labels=list(range(len(classes))),
                                   target_names=classes, output_dict=True, zero_division=0)
    return {'accuracy': float(np.mean(labels == predictions)), 'count': len(labels),
            'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall'], 'macro_f1': report['macro avg']['f1-score'],
            'report': report,
            'confusion_matrix': confusion_matrix(labels, predictions, labels=list(range(len(classes)))).tolist()}


def evaluate_once(model, loader, device, classes, checkpoint: Path, output_dir: Path, source: str):
    """Reuse predictions for the exact selected checkpoint; never select using test scores."""
    import json
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / 'evaluation.json'
    fingerprint = file_sha256(checkpoint)
    if cache.exists():
        result = json.loads(cache.read_text(encoding='utf-8'))
        if (result['checkpoint_sha256'] != fingerprint or result['source'] != source
                or result['count'] != len(loader.dataset) or result['classes'] != classes):
            raise ValueError('Stale evaluation cache. Choose a new experiment directory; do not mix runs.')
        checked = classification_metrics(result['labels'], result['predictions'], classes)
        for key in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'confusion_matrix', 'report']:
            if checked[key] != result[key]:
                raise ValueError(f'Cached evaluation is inconsistent: {key}')
        return result
    labels, predictions, confidence = [], [], []
    loss_sum = 0.0
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss_sum += torch.nn.functional.cross_entropy(logits, targets, reduction='sum').item()
            probabilities = logits.softmax(1)
            values, indices = probabilities.max(1)
            labels.extend(targets.cpu().tolist())
            predictions.extend(indices.cpu().tolist())
            confidence.extend(values.cpu().tolist())
    result = {**classification_metrics(labels, predictions, classes),
              'loss': loss_sum / len(labels), 'classes': classes,
              'labels': labels, 'predictions': predictions, 'confidence': confidence,
              'checkpoint_sha256': fingerprint, 'source': source}
    write_json(cache, result)
    return result
