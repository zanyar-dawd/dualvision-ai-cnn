"""Figures and error summaries derived only from recorded predictions/history."""
from pathlib import Path

import matplotlib
# Figures are saved and displayed as notebook PNGs; no desktop GUI/Tk is needed.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay


def denormalize(image, mean, std):
    image = image.detach().cpu()
    pixels = image * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]
    return pixels.clamp(0, 1).permute(1, 2, 0).numpy().squeeze()


def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def make_figures(metadata, evaluation, bundle, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(metadata['history'])
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ['loss', 'accuracy']):
        for split in ['train', 'validation']:
            ax.plot(history.epoch, history[f'{split}_{metric}'], marker='o', markersize=3, label=split)
        ax.axvline(metadata['best_epoch'], color='gray', linestyle='--', label='selected epoch')
        ax.set(xlabel='Epoch', ylabel=metric.capitalize(), title=f"{metadata['dataset'].upper()} {metric}")
        ax.legend()
        if len(history) <= 15:
            ax.set_xticks(history.epoch)
        ax.grid(alpha=0.2)
    fig.suptitle('SYNTHETIC SMOKE TEST — NOT BENCHMARK RESULTS' if metadata['smoke'] else 'Learning curves')
    paths.append(output_dir / 'learning_curves.png')
    save_fig(fig, paths[-1])
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(np.asarray(evaluation['confusion_matrix']), display_labels=bundle.classes).plot(
        ax=ax, cmap='Blues', colorbar=False, xticks_rotation=45)
    ax.set_title('Synthetic smoke predictions' if metadata['smoke'] else 'Held-out test confusion matrix')
    paths.append(output_dir / 'confusion_matrix.png')
    save_fig(fig, paths[-1])
    truth, predicted = np.array(evaluation['labels']), np.array(evaluation['predictions'])
    for correct, filename in [(True, 'correct_samples.png'), (False, 'mistakes.png')]:
        chosen = np.flatnonzero((truth == predicted) == correct)[:8]
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for ax in axes.flat:
            ax.axis('off')
        for ax, index in zip(axes.flat, chosen):
            image, _ = bundle.test.dataset[int(index)]
            ax.imshow(denormalize(image, bundle.mean, bundle.std), cmap='gray' if len(bundle.mean) == 1 else None)
            ax.set_title(f"True: {bundle.classes[truth[index]]}\nPred: {bundle.classes[predicted[index]]} "
                         f"({evaluation['confidence'][index]:.1%})", fontsize=10)
        heading = 'Correct predictions' if correct else 'Observed mistakes'
        if len(chosen) == 0:
            heading += ' — no examples in this evaluation'
        if metadata['smoke']:
            heading = 'SYNTHETIC: ' + heading
        fig.suptitle(heading)
        paths.append(output_dir / filename)
        save_fig(fig, paths[-1])
    pd.DataFrame(evaluation['report']).T.to_csv(output_dir / 'per_class_metrics.csv')
    history.to_csv(output_dir / 'history.csv', index=False)
    return paths


def observed_errors(evaluation, classes, limit: int = 5) -> str:
    cm = np.asarray(evaluation['confusion_matrix']).copy()
    np.fill_diagonal(cm, 0)
    indices = np.argsort(cm.ravel())[::-1]
    pairs = [(i // len(classes), i % len(classes), int(cm.ravel()[i])) for i in indices if cm.ravel()[i] > 0][:limit]
    if not pairs:
        return 'No misclassifications were observed in this evaluation.'
    return '\n'.join(f'- {classes[a]} -> {classes[b]}: {n} images' for a, b, n in pairs)
