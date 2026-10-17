# DualVision AI status

Verified on 2026-10-17. The optimized MNIST and CIFAR-10 experiments, official test evaluations, figures, interactive inference, CPU/CUDA portability checks, and self-contained notebook are complete. The required submission is `Homework_5_DualVision.ipynb`; its first cell contains only the requested student name, Zanyar Dawd.

## Final verified results

| Model | Parameters | Best validation accuracy | Official test accuracy | Test loss | Macro F1 | Training time | Selected epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| DigitNetPlus / MNIST | 435,754 | 99.70% | **99.65%** (9,965 / 10,000) | 0.01214 | 0.99649 | 1,480.27 s (24 min 40 s) | 45 / 60 |
| VisionNetWide / CIFAR-10 | 5,849,050 | 94.12% | **93.32%** (9,332 / 10,000) | 0.20026 | 0.93328 | 26,827.09 s (7 h 27 min) | 148 / 200 |

The official test accuracies, 99.65% and 93.32%, are the final numbers to report. The 99.70% MNIST and 94.12% CIFAR-10 values are validation accuracies used for checkpoint selection. Test labels were not used to choose checkpoints.

The abandoned `accuracy-mnist-998-20261017` run was stopped at epoch 3 at the user's request. It was never evaluated on the official test set and is not used by the notebook.

## Selected artifacts

- MNIST checkpoint: `checkpoints/accuracy-mnist-997-20261017/mnist/best.pt`
- MNIST report: `results/accuracy-mnist-997-20261017/mnist/`
- CIFAR-10 checkpoint: `checkpoints/accuracy-max-20261017/cifar10/best.pt`
- CIFAR-10 report: `results/accuracy-max-20261017/cifar10/`
- Machine-readable selection: `results/final-model-selection.json`

The final notebook embeds both checkpoints, evaluation JSON files, per-class reports, curves, confusion matrices, correct/mistaken examples, and the CIFAR-10 Grad-CAM image. It restores them into an empty folder and loads them with `map_location=device`. Training and downloads remain disabled by default.

## Hardware and training safeguards

- Environment: project-local Python 3.12.11, torch 2.8.0+cu126, torchvision 0.23.0+cu126.
- GPU: NVIDIA Quadro T1000, 4 GB; actual CUDA tensor operations were verified.
- MNIST peak: 70 C and 473 MiB sampled device memory.
- CIFAR-10 peak: 77 C and 1,806 MiB sampled device memory.
- Both remained below the project stop limits of 85 C and 95% device memory.
- Epoch-boundary recovery checkpoints, early stopping, validation selection, and best-weight checkpointing were active.

## Notebook portability and verification

- Automatic mode uses `torch.cuda.is_available()` and verifies real CUDA matrix multiplication and convolution. Failed or unavailable CUDA falls back to CPU. `DEVICE_PREFERENCE="cpu"` provides a manual override.
- The complete notebook passed execution from an empty CPU-only folder with no datasets, source modules, downloads, or retraining.
- The same notebook passed execution from an empty folder on the actual CUDA GPU.
- Both interactive upload callbacks, path-based inference, checkpoint loading, softmax probabilities, and CIFAR-10 Grad-CAM passed on the portability paths.
- The complete 38-test suite covering notebook structure, assets, core models, checkpoints, metrics, Grad-CAM, portability, and optimized experiment safeguards passed.
- `results/real_experiment_audit.json` confirms checkpoint metadata, validation selection, 10,000-image official test counts, cached predictions, metrics, history CSVs, class reports, figures, and telemetry.

## Final checklist

- **Complete:** both custom PyTorch CNNs, real training, validation-only checkpoint selection, official test metrics, checkpoints, plots, confusion matrices, per-class reports, sample predictions, error analysis, Grad-CAM, upload inference, automatic CPU/CUDA selection, CPU fallback, documentation, and executed self-contained notebook.
- **Complete:** student name is filled in; unrequested ID, course, instructor, and date placeholders are removed.
- **Complete:** datasets, virtual environments, and standalone checkpoint files remain excluded from Git. The selected checkpoints are carried inside the notebook archive for teacher portability.
- **Submission file:** `Homework_5_DualVision.ipynb` or the identical copy under `teacher_submission/`.

No additional training is running or planned. Default notebook settings are `ALLOW_TRAINING=False` and `DOWNLOAD_DATA=False`.
