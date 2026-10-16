"""Download and verify official dataset partitions; does not train models."""
from pathlib import Path
from torchvision.datasets import MNIST, CIFAR10
from src.utilities.runtime import write_json

if __name__ == '__main__':
    sizes = {}
    for name, constructor in [('mnist', MNIST), ('cifar10', CIFAR10)]:
        training = constructor('data', train=True, download=True)
        test = constructor('data', train=False, download=True)
        expected = 60000 if name == 'mnist' else 50000
        assert len(training) == expected and len(test) == 10000
        sizes[name] = {'official_training_count': len(training), 'official_test_count': len(test),
                       'torchvision_download_integrity_checks': 'passed'}
        print(f'{name}: verified {len(training)} official training and {len(test)} test examples.', flush=True)
    write_json(Path('results/dataset_verification.json'), sizes)
