"""Class-specific Grad-CAM, with an independent GAP/linear-head identity check."""
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from src.visualization.plots import denormalize, save_fig


def grad_cam(model, image, target_class: int | None = None):
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError('Grad-CAM expects one NCHW image.')
    model.eval()
    captured = []
    handle = model.stage3.register_forward_hook(lambda module, inputs, output: captured.append(output))
    try:
        with torch.enable_grad():
            logits = model(image.detach().requires_grad_(True))
            target = int(logits.argmax(1)) if target_class is None else int(target_class)
            if not 0 <= target < 10:
                raise ValueError('Target class must lie in [0, 9].')
            activation = captured[0]
            gradients, = torch.autograd.grad(logits[0, target], activation)
            raw = (gradients.mean((2, 3), keepdim=True) * activation).sum(1, keepdim=True).detach()
            # In eval mode the head is GAP + Linear. Its analytical CAM must match
            # the gradient implementation up to the spatial averaging factor.
            weights = model.head[-1].weight[target].detach()[None, :, None, None]
            analytic = (weights * activation.detach()).sum(1, keepdim=True) / (activation.shape[2] * activation.shape[3])
            torch.testing.assert_close(raw, analytic, atol=1e-5, rtol=1e-4)
        heatmap = F.interpolate(raw.cpu().relu(), image.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
        heatmap -= heatmap.min()
        span = heatmap.max()
        if span > 1e-12:
            heatmap /= span
        if not torch.isfinite(heatmap).all():
            raise FloatingPointError('Non-finite Grad-CAM.')
        return heatmap.numpy(), target, logits.detach().softmax(1)[0].cpu().numpy()
    finally:
        handle.remove()


def grad_cam_figure(model, image, mean, std, classes, path, target_class=None, smoke=False):
    heatmap, target, probabilities = grad_cam(model, image, target_class)
    original = denormalize(image[0], mean, std)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(original)
    axes[0].set_title('Input image')
    axes[1].imshow(heatmap, cmap='inferno', vmin=0, vmax=1)
    axes[1].set_title('Grad-CAM (relative scale)')
    axes[2].imshow(original)
    axes[2].imshow(heatmap, cmap='inferno', alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title(f'{classes[target]}: {probabilities[target]:.1%}')
    for ax in axes:
        ax.axis('off')
    title = 'SYNTHETIC SMOKE TEST — NOT AN OBJECT EXPLANATION' if smoke else ''
    if np.ptp(heatmap) < 1e-12:
        title += '\nDegenerate map: no positive spatial contrast for this class'
    if title:
        fig.suptitle(title.strip())
    save_fig(fig, path)
    return heatmap
