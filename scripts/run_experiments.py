"""Explicit CLI: real data download and training are separate opt-in flags."""
import argparse
import json
from pathlib import Path
from src.workflow import run_experiment, comparison_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'both'], default='both')
    parser.add_argument('--download', action='store_true', help='Allow official dataset downloads.')
    parser.add_argument('--train', action='store_true', help='Allow bounded training on real data.')
    parser.add_argument('--smoke', action='store_true', help='Synthetic one-epoch check; no downloads.')
    parser.add_argument('--retrain', action='store_true', help='Train a new experiment ID; never overwrite a previous run.')
    parser.add_argument('--resume', action='store_true', help='Resume an incomplete training run within its original total budget.')
    parser.add_argument('--experiment-id', default='baseline')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--config', type=Path, default=Path('configs/experiments.json'))
    args = parser.parse_args()
    configs = json.loads(args.config.read_text(encoding='utf-8'))
    names = ['mnist', 'cifar10'] if args.dataset == 'both' else [args.dataset]
    experiments = {name: run_experiment(name, experiment_id=args.experiment_id, config=configs[name],
                    allow_training=args.train, download=args.download, retrain=args.retrain,
                    smoke=args.smoke, device_preference=args.device, resume=args.resume) for name in names}
    table = comparison_table(experiments)
    print(table.to_string(index=False))
    destination = Path('results') / ('smoke' if args.smoke else '') / args.experiment_id
    destination.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination / 'comparison.csv', index=False)


if __name__ == '__main__':
    main()
