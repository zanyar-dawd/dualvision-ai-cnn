"""Reproducibility, device verification and atomic artifact utilities."""
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time
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


def select_device(preference: str = 'auto') -> tuple[torch.device, dict]:
    if preference not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('Device must be auto, cpu, or cuda.')
    if preference == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')
    device = torch.device('cuda' if preference != 'cpu' and torch.cuda.is_available() else 'cpu')
    # Exercise matrix multiplication, not just device enumeration. Fail visibly if broken.
    x = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    actual = x @ x.T
    expected = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    torch.testing.assert_close(actual.cpu(), expected @ expected.T)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    record = {
        'python': platform.python_version(), 'platform': platform.platform(),
        'device': str(device), 'cuda_available': torch.cuda.is_available(),
        'torch_cuda_runtime': torch.version.cuda,
        'tensor_operation_verified': True,
        'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        'gpu_capability': list(torch.cuda.get_device_capability(0)) if device.type == 'cuda' else None,
        'compiled_cuda_architectures': torch.cuda.get_arch_list() if device.type == 'cuda' else [],
        'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory if device.type == 'cuda' else None,
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
