"""Reproducibility, device verification and atomic artifact utilities."""
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time
import warnings
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLCONFIGDIR', str(Path.cwd() / '.downloads' / 'matplotlib'))
import numpy as np
import torch

IMPLEMENTATION_VERSION = 'dualvision-v1'


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def verify_device_operation(device):
    """Verify CUDA initialization, matrix multiplication and the CNN backend."""
    x = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    expected = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    torch.testing.assert_close((x @ x.T).cpu(), expected @ expected.T)
    pixels = torch.ones(1, 1, 5, 5, device=device)
    kernel = torch.ones(1, 1, 3, 3, device=device)
    actual = torch.nn.functional.conv2d(pixels, kernel)
    torch.testing.assert_close(actual.cpu(), torch.full((1, 1, 3, 3), 9.0))
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def select_device(preference: str = 'auto') -> tuple[torch.device, dict]:
    preference = str(preference)
    if preference == 'cuda:0':
        preference = 'cuda'
    if preference not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('Device must be auto, cpu, or cuda.')
    available = torch.cuda.is_available()
    if preference == 'cuda' and not available:
        raise RuntimeError('CUDA was requested but is unavailable.')
    device = torch.device('cuda' if preference != 'cpu' and available else 'cpu')
    fallback_reason, gpu = None, {}
    if device.type == 'cuda':
        try:
            verify_device_operation(device)
            gpu = {'gpu': torch.cuda.get_device_name(0),
                   'gpu_capability': list(torch.cuda.get_device_capability(0)),
                   'compiled_cuda_architectures': torch.cuda.get_arch_list(),
                   'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory}
        except (RuntimeError, AssertionError, OSError) as error:
            if preference == 'cuda':
                raise RuntimeError(f'Explicit CUDA verification failed: {error}') from error
            fallback_reason = f'{type(error).__name__}: {error}'
            warnings.warn(f'CUDA verification failed; using CPU. {fallback_reason}', RuntimeWarning, stacklevel=2)
            device = torch.device('cpu')
    if device.type == 'cpu':
        verify_device_operation(device)
    record = {
        'python': platform.python_version(), 'platform': platform.platform(),
        'device': str(device), 'requested_device': preference, 'cuda_available': available,
        'cuda_verified': device.type == 'cuda', 'fallback_reason': fallback_reason,
        'torch_cuda_runtime': torch.version.cuda,
        'tensor_operation_verified': True,
        'gpu': None, 'gpu_capability': None,
        'compiled_cuda_architectures': [], 'gpu_memory_bytes': None,
        **gpu,
        'versions': {p: importlib.metadata.version(p) for p in
                     ['torch', 'torchvision', 'numpy', 'scikit-learn', 'matplotlib', 'Pillow']},
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cpu_threads': torch.get_num_threads(),
    }
    return device, record


def write_json(path: Path, value: dict | list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
