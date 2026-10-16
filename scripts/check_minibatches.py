"""Verify the actual configured batch sizes with synthetic tensors; no datasets."""
import json
import time
from pathlib import Path
import torch
from src.models.cnn import make_model
from src.utilities.runtime import seed_everything, select_device, parameter_count, write_json
from src.workflow import DEFAULT_CONFIGS


def main():
    seed_everything()
    device, runtime = select_device()
    rows = []
    for name, shape in [('mnist', (1, 28, 28)), ('cifar10', (3, 32, 32))]:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model = make_model(name).to(device).train()
        batch_size = DEFAULT_CONFIGS[name]['batch_size']
        images = torch.randn(batch_size, *shape, device=device)
        labels = torch.arange(batch_size, device=device) % 10
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        before = next(model.parameters()).detach().clone()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        assert logits.shape == (batch_size, 10)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        assert not torch.equal(before, next(model.parameters()))
        if device.type == 'cuda':
            torch.cuda.synchronize()
        rows.append({'dataset_shape': name, 'source': 'SYNTHETIC_BATCH_ONLY', 'batch_size': batch_size,
                     'output_shape': list(logits.shape), 'parameters': parameter_count(model),
                     'device': str(device), 'optimizer_step_verified': True,
                     'seconds': time.perf_counter() - started,
                     'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0,
                     'peak_reserved_bytes': torch.cuda.max_memory_reserved() if device.type == 'cuda' else 0})
        del model, optimizer, images, labels, logits, loss, before
    result = {'runtime': runtime, 'batches': rows}
    write_json(Path('results/minibatch_verification.json'), result)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
