"""Recover the official CIFAR archive from a pinned academic mirror via torchvision."""
import hashlib
from pathlib import Path
from torchvision.datasets import CIFAR10, MNIST
from src.utilities.runtime import write_json

MIRROR = 'https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python/resolve/a48007227f9e2cd0af96175f4afb5ac4965e261b/cifar-10-python.tar.gz'

if __name__ == '__main__':
    CIFAR10.url = MIRROR
    train = CIFAR10('data', train=True, download=True)
    test = CIFAR10('data', train=False, download=False)
    assert len(train) == 50000 and len(test) == 10000
    archive = Path('data/cifar-10-python.tar.gz').read_bytes()
    digest = hashlib.md5(archive).hexdigest()
    assert digest == CIFAR10.tgz_md5 == 'c58f30108f718f92721af3b95e74349a'
    assert len(MNIST('data', train=True)) == 60000 and len(MNIST('data', train=False)) == 10000
    write_json(Path('results/dataset_verification.json'), {
        'mnist': {'official_training_count': 60000, 'official_test_count': 10000,
                  'torchvision_download_integrity_checks': 'passed'},
        'cifar10': {'official_training_count': 50000, 'official_test_count': 10000,
                    'torchvision_download_integrity_checks': 'passed', 'archive_md5': digest,
                    'archive_bytes': len(archive), 'recovery_mirror': MIRROR,
                    'note': 'Unmodified official archive, recovered after original host transfer failed.'}})
    print('PASS: both official datasets verified; CIFAR-10 archive checksum matches the original distribution.', flush=True)
