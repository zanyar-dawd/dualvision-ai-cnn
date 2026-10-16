# Accuracy upgrade experiment — prepared, not yet trained

Prepared on 2026-10-16. Requested targets are **99.8% MNIST** and **95.5–96% CIFAR-10**. These are aspirations, not measured results or guarantees. No new extended training has started. The completed submission still reports the actual baseline scores: **99.25% MNIST** and **90.92% CIFAR-10**.

## Proposed, bounded experiment

| Setting | MNIST | CIFAR-10 |
|---|---|---|
| Model | DigitNetPlus: six small convolutions, spatial classification head | VisionNetWide: custom WRN-28-4 style preactivation residual CNN |
| Parameters | 435,754 | 5,849,050 |
| Training / validation examples | 54,000 / 6,000 | 45,000 / 5,000 |
| Training augmentation | Mild affine rotation, translation, scale and shear | Random crop, horizontal flip, CIFAR-10 AutoAugment policy |
| Regularization | Dropout, AdamW weight decay | Dropout, SGD weight decay, mixup alpha 0.2 on half of batches |
| Optimizer | AdamW, initial maximum LR 0.001 | Nesterov SGD, momentum 0.9, initial maximum LR 0.05 |
| Schedule | 3-epoch warmup, cosine decay | 5-epoch warmup, cosine decay |
| Maximum epochs / batch size | 40 / 128 | 200 / 64 |
| Early stopping | 12 stale epochs, earliest stop epoch 15 | 35 stale epochs, earliest stop epoch 100 |
| Proposed additional active training cap | **45 minutes** | **12 hours** |

Both models train from scratch using existing local datasets. No new dataset, pretrained weights, dependency installation or system change is needed. Exact settings are in `configs/accuracy_v2.json`; implementation is in `experiments/`. Existing baseline weights and the executed main notebook are preserved.

The longer CIFAR schedule is a bounded attempt at a challenging target, not a promise that more training will reach it. Early stopping is delayed to allow the warmup/cosine schedule time to improve generalization. A time limit or hardware stop can prevent completion of that schedule. Do not extend the budget simply because a target was missed.

## What “without overfitting” can mean here

No finite experiment can guarantee absence of overfitting. We will report validation and final test outcomes honestly, including misses of the requested targets. The controls are:

- Reuse the fixed stratified training/validation split and calculate normalization from training indices only. Augmentation is training-only.
- Select the best epoch by validation accuracy, breaking ties with validation loss. Neither training nor model selection opens the official test partition.
- Measure a fixed 2,048-image clean training subset in evaluation mode each epoch, alongside validation. These curves reveal generalization gaps without confusing augmented/mixed training accuracy with clean validation accuracy. The training subset is a diagnostic sample, not a full-training-set estimate.
- Inspect validation loss as well as accuracy; rising validation loss with improving clean training performance is evidence of overfitting. A small gap alone is not proof of good generalization.
- Freeze this recipe before running; do not select seeds, epochs or architectures using test scores. The baseline test scores and errors have already been seen, so the reused benchmark cannot be described as entirely unseen throughout project development.
- Use a separate explicit final test operation after validation review. Cache its predictions for plots and reporting; block further training of an evaluated experiment.

Mixup training accuracy is explicitly labeled as mixed-label weighted agreement. The candidate report's learning curves use comparable, unmixed evaluation-mode measurements. The original augmented/mixed metrics remain in the history CSV for transparency.

## Measured preparation checks

`results/accuracy_v2_preflight.json` records real Quadro T1000 CUDA forward/backward/optimizer operations at the proposed full batch sizes. After two warmup steps, five steps were timed per model:

| Measurement | MNIST | CIFAR-10 |
|---|---:|---:|
| Median synthetic optimizer step | 0.01869 s | 0.12640 s |
| Peak PyTorch allocated GPU memory | 200.30 MiB | 738.47 MiB |
| Peak PyTorch reserved GPU memory | 396 MiB | 772 MiB |
| Compute-only extrapolation for all planned epochs | 5.26 min | 4.94 hours |

These are short synthetic resource measurements, **not actual full-training durations or accuracy results**. The extrapolation excludes image preprocessing, data loading, validation, clean-training diagnostics, checkpoint writes and sustained clock changes. Device-wide memory also includes CUDA context and other applications. The proposed caps allow overhead and bound the work; they do not guarantee all epochs will finish.

Strict deterministic algorithms remain enabled. A CUDA-incompatible adaptive pooling backward operation found during preflight was replaced by fixed pooling. Both full candidate pipelines completed two-epoch CPU synthetic smoke checks. Their synthetic scores are not benchmark results.

`results/accuracy_v2_full_tests.xml`: **32 passed**, no skipped GPU tests, 14 upstream Matplotlib/pyparsing deprecation warnings. Coverage includes both CPU/CUDA optimizer steps, transforms, mixup loss, schedule, analytical Grad-CAM validation, exact interrupted-run recovery, checkpoint reload and baseline overwrite prevention. `scripts.verify_submission` also passed again for the existing completed notebook.

## Monitoring and recovery

Training samples NVIDIA temperature, device memory, utilization and power every five seconds. The project stops at **85 C**, **95% device-memory use**, failed telemetry, CUDA out-of-memory, non-finite loss/gradients or its cumulative time cap. These are project policies, not a claim about manufacturer temperature limits.

Every completed epoch atomically saves `resume.pt`, containing current weights, optimizer, schedule position, RNG states, validation-best weights, history and cumulative time. On a caught interruption, `interrupted.pt` saves partial progress; resumption uses the preceding complete epoch and counts interrupted work against the cap. `best.pt` is exported when training is finalized. A hard power/process failure can lose work since the last completed epoch. CPU and CUDA inference use the same selected state dictionaries.

## Commands after runtime approval

From the repository root with the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -u -m experiments.train_v2 --train --dataset mnist --device cuda --experiment-id accuracy-v2
.\.venv\Scripts\python.exe -u -m experiments.train_v2 --train --dataset cifar10 --device cuda --experiment-id accuracy-v2
```

Run sequentially to avoid competing for the 4 GB GPU. Add `--resume` only when recovering an interrupted candidate. Do not change the recipe or reset the time budget during a run. New results will be under `results/accuracy-v2/`; checkpoints under `checkpoints/accuracy-v2/`.

After validation-only review and final model selection:

```powershell
.\.venv\Scripts\python.exe -m experiments.train_v2 --evaluate --dataset both --device cpu --experiment-id accuracy-v2
```

The baseline and candidate selection must be based on validation evidence, not on whichever final test score is higher. Keep and report the final test result even if it is disappointing. A further study requires a new experiment ID and disclosure of previous test exposure.

## Remaining after approval

1. Run the bounded real-data training, monitor resource use and inspect clean training/validation diagnostics.
2. Freeze selected models and generate true test reports, errors, plots and Grad-CAM.
3. Integrate the selected candidate architectures and inference loading into the self-contained `Homework_5_DualVision.ipynb`; include the actual recipe, results and limitations. Candidate code is not yet part of that baseline submission.
4. Repackage selected weights, verify both upload callbacks and execute the notebook in an empty CPU environment without retraining or datasets. Keep training optional.
5. Update `STATUS.md` with measured outcomes and submission verification. Never replace measured scores with requested targets.

The main notebook currently remains a complete, verified report of the **baseline** experiment. The requested accuracy upgrade remains **pending real training, evaluation and notebook integration**.

## Method sources

The architectures and augmentations draw on [Wide Residual Networks](https://arxiv.org/abs/1605.07146), [mixup](https://arxiv.org/abs/1710.09412), and [AutoAugment](https://arxiv.org/abs/1805.09501). The augmentation implementation uses the installed [torchvision 0.23 AutoAugment API](https://docs.pytorch.org/vision/0.23/generated/torchvision.transforms.AutoAugment.html). Published results are not this project's results, and these methods do not guarantee the requested accuracy on this run.
