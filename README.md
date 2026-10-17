# DualVision AI — Intelligent Digit and Object Recognition

Homework 5: two custom PyTorch CNNs and a self-contained Jupyter Notebook report.

**Status:** both optimized real experiments are complete: **MNIST 99.65%** and **CIFAR-10 93.32%** held-out test accuracy (10,000 test images each). The notebook embeds the validation-selected weights, measured report, figures, Grad-CAM output, and inference demonstrations. See [STATUS.md](STATUS.md) for verification evidence and limitations. The student name is filled in and the unrequested information fields have been removed.

Open **[Homework_5_DualVision.ipynb](Homework_5_DualVision.ipynb)**. It includes the actual architectures, data pipeline, training, checkpointing, evaluation, plots, Grad-CAM, upload interface and image inference code. It does not import repository modules or invoke a hidden training script. The `.ipynb` is the required submission.

For your teacher's CPU-only laptop, see [TEACHER_QUICKSTART.md](TEACHER_QUICKSTART.md). The final packaged notebook embeds compact checkpoints and report assets so default execution needs neither datasets nor retraining. GPU installation below is for reproducing the original training workstation.

## Windows 11 setup

Use Python **3.12**, separate from Python 3.14 and ArcGIS Python 2.7. Always invoke the explicit environment executable below. Do not change ArcGIS, system Python, PATH or the CUDA Toolkit.

The standard Python 3.12.10 MSI installer failed on this machine with Windows Installer 2502/1603. The alternate script uses pinned project-local `uv` and CPython 3.12.11, disables executable/registry registration, then creates `.venv`. Run only after approving installation and multi-GB downloads:

```powershell
.\scripts\setup-portable.ps1
```

With an existing Python 3.12 executable:

```powershell
.\scripts\setup.ps1 -Python312 'C:\path\to\Python312\python.exe'
```

Manual equivalent:

```powershell
& 'C:\path\to\Python312\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m scripts.verify_environment
```

The official CUDA 12.6 wheel pair is pinned deliberately. Driver-reported CUDA 13.2 describes driver support, not an installed Toolkit. PyTorch wheels supply runtime libraries. Automatic device selection checks `torch.cuda.is_available()` and verifies matrix multiplication and convolution. Working CUDA uses the NVIDIA GPU; unavailable or failed CUDA verification falls back to CPU, with the failure recorded and displayed. Explicit `--device cuda` remains a strict diagnostic option that raises if CUDA fails. Mid-training hardware faults still stop and save progress.

The notebook, saved-checkpoint inference, upload UI, baseline workflow and candidate prediction functions default to automatic selection. To force CPU, set `DEVICE_PREFERENCE = "cpu"` in the notebook, use `--device cpu` in training CLIs, or pass `device="cpu"` to prediction functions. `DUALVISION_DEVICE=cpu` overrides the notebook preference through the environment. The notebook displays the selected device and GPU name. Checkpoint tensors load with `map_location=device`; model inputs move to that device. Default execution loads pretrained checkpoints without retraining or downloading data. Explicit resume snapshots deserialize RNG bookkeeping on CPU before restoring the selected device's training state.

In VS Code select **Select Kernel → Python Environments → `.venv/Scripts/python.exe`**. No activation or global kernelspec installation is necessary. Alternatively:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab Homework_5_DualVision.ipynb
```

## Verify before downloading datasets

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m scripts.check_minibatches
.\.venv\Scripts\python.exe -m scripts.run_experiments --smoke --experiment-id smoke-01
.\.venv\Scripts\python.exe -m scripts.smoke_notebook
```

Smoke mode uses tiny synthetic datasets, never downloads data, and writes under `results/smoke/`. Checks exercise both models, backward/optimizer steps, checkpoint reload, cache reuse, plots and Grad-CAM. The notebook runner launches a fresh kernel in an empty directory without repository source modules. This checks notebook independence, not installation on every possible computer.

## Approved real experiments

Run only after dataset downloads and extended training are confirmed. Compressed downloads total approximately 175 MB, plus extracted data. Flags explicitly authorize downloads and training:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_experiments --dataset mnist --download --train
.\.venv\Scripts\python.exe -m scripts.run_experiments --dataset cifar10 --download --train
```

Then run the notebook with default flags to load the checkpoints and populate the report. Alternatively set `ALLOW_TRAINING=True` and `DOWNLOAD_DATA=True` in the notebook and run everything there. Matching checkpoints load automatically, preventing duplicate training.

| Setting | DigitNet / MNIST | VisionNet / CIFAR-10 |
|---|---:|---:|
| Train / validation / test | 54,000 / 6,000 / 10,000 | 45,000 / 5,000 / 10,000 |
| Batch size | 128 | 64 |
| Maximum epochs | 15 | 50 |
| AdamW learning rate | 0.001 | 0.001 |
| Weight decay | 0.0001 | 0.0005 |
| Early-stopping patience | 5 | 12 |
| Time budget | 30 minutes | 115 minutes + 5 minutes reserved for interrupted attempt |
| Augmentation | none | padded random crop + horizontal flip |

The first CIFAR-10 process ended externally after three epochs, before any test evaluation. Its weights and telemetry are preserved under `interrupted-cifar10-20261015`; the final experiment starts afresh with seed 42. `results/cifar10_recovery.json` records the five-minute allowance for that attempt. Budgets are not predicted runtimes. Real GPU training checks time at batch boundaries and logs GPU temperature/memory/utilization/power about every five seconds. The conservative project stop policies are 85°C and 95% device-memory use; they are not hardware ratings. A safety stop preserves the best checkpoint plus interrupted weights and optimizer state, and propagates the error. Training uses float32, at most four CPU threads, and zero loader workers for Windows reliability. Parameter counts and peak allocated VRAM are recorded.

CLI configuration is `configs/experiments.json`; standalone notebook configuration is its `CONFIGS` cell. Keep these aligned. Metadata mismatches refuse incompatible checkpoint reuse. For changed settings use a new ID, e.g. `--experiment-id validation-study-02 --retrain --train`; existing experiments cannot be overwritten. Interrupted checkpoints remain `in_progress` and are not test-evaluated. Automatic resumption is not implemented; restart with a new ID. Never select settings using test results.

## Scientific safeguards

- Official torchvision partition integrity/size checks; validation is a stratified training-only split with seed 42.
- Normalization statistics come only from selected training pixels. Augmentation is training-only.
- Validation accuracy selects checkpoints; validation loss breaks ties. Early stopping and scheduling never inspect test data.
- Final predictions are cached against a SHA-256 checkpoint digest; reruns reuse them. Cached reports are verified against stored labels/predictions.
- Accuracy is total correct / total samples. Precision/recall/F1 are **macro** averages; reports also include per-class support and weighted averages.
- Split indices, fingerprint, versions, device, hyperparameters, timing, precision and peak VRAM accompany artifacts.
- Cross-dataset accuracy concerns different tasks, not a controlled architecture comparison. Targets are not guarantees.

## Inference and explainability

The notebook enables `ipywidgets` uploads after real training. Digit processing supports ink polarity overrides, bounding-box resizing and centroid alignment. Use one digit on a plain background. Color uploads become 32×32 RGB. CIFAR-10 recognizes only its ten predefined classes; unfamiliar images still receive one of those labels. Softmax scores are not calibrated certainty.

Path-based fallback, also defined inline in the notebook:

```python
from src.utilities.inference import predict_file
result = predict_file('digit.png', 'checkpoints/accuracy-mnist-997-20261017/mnist/best.pt', 'mnist')
print(result['top_classes'])
```

Grad-CAM uses the last residual stage. An independent global-pooling/linear-head identity validates its gradients against classifier-weight maps. Tests check class sensitivity, finite range, hook cleanup and zero maps. This is associative visualization, not proof of causality.

## Organization and reproduction

```text
Homework_5_DualVision.ipynb   Required self-contained submission
src/models/                 Both CNNs
src/training/               Epoch loops, checkpoint handling
src/evaluation/             Metrics, cached test evaluation
src/visualization/          Figures, Grad-CAM
src/utilities/              Runtime, data, image inference
src/workflow.py             Explicit experiment orchestration
configs/                    Hyperparameters
scripts/                    Setup, GPU check, CLI, notebook builder/smoke
tests/                      Practical model/pipeline tests
data/                       Ignored official datasets
checkpoints/<id>/<dataset>/  Ignored state_dict checkpoints
results/<id>/<dataset>/     JSON/CSV evidence and PNG figures after real training
results/smoke/              Ignored synthetic artifacts
```

Real artifacts include `training.json`, `split.json`, `evaluation.json`, per-class/history CSVs, learning curves, confusion matrices, correct/mistaken samples, and CIFAR-10 Grad-CAM. Environment evidence is `results/environment.json`; installed package versions are recorded after setup. Data, runtime, environment and model weights are Git-ignored.

`results/environment-freeze.txt` contains the exact installed package snapshot. Install the CUDA wheel pair first, then `python -m pip install -r results/environment-freeze.txt` to reproduce that snapshot. Core dependencies are pinned in `requirements.txt` for a portable minimal specification. The local verification completed with 23 passing tests and no skipped GPU tests; remaining upstream plotting deprecation warnings are documented in `STATUS.md`.

After source edits run `python -m scripts.build_notebook`, then tests and notebook smoke execution. **Rebuilding resets notebook outputs and the embedded archive**. After real results are finalized, run `python -m scripts.verify_trained_inference`, `python -m scripts.audit_results`, and `python -m scripts.package_submission`. Run `python -m scripts.execute_submission --device cpu` and `python -m scripts.execute_submission --device cuda`, then `python -m scripts.verify_submission`. Both executions use empty folders and embedded assets, test automatic selection, both upload callbacks and Grad-CAM, and require no datasets or retraining. The CPU run hides CUDA and saves executed notebook outputs; the GPU run verifies the same notebook on real hardware without replacing those portable report outputs. Final portability tests passed: **38 tests, zero failures or skips**, with 14 upstream plotting deprecation warnings (`results/portability_tests.xml`).

The original CIFAR-10 host transfer failed integrity verification. Recovery used torchvision with the [MIT Open Learning mirror](https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python), pinned to revision `a48007227f9e2cd0af96175f4afb5ac4965e261b`. Its 170,498,071-byte archive matches the original MD5 `c58f30108f718f92721af3b95e74349a`; the data were not altered. Provenance is recorded in `results/dataset_verification.json`.

The saved report includes actual confusion patterns, inspected error images, checked Grad-CAM, and documented training interruptions. The first cell identifies the student as Zanyar Dawd. Its default execution loads embedded checkpoints and saved results; no dataset download or retraining is required.

References: [teacher MNIST](https://github.com/moelgendy/deep_learning_for_vision_systems/blob/master/chapter_03/mnist_cnn.ipynb), [teacher CIFAR-10](https://github.com/moelgendy/deep_learning_for_vision_systems/blob/master/chapter_04/chapter_4_improve_cifar_90_percent.ipynb), [PyTorch wheel combinations](https://pytorch.org/get-started/previous-versions/), [uv Python setup](https://docs.astral.sh/uv/guides/install-python/), [Grad-CAM](https://arxiv.org/abs/1610.02391). Reference scores are not this project's results.

Recovery after interruption: `python -m scripts.run_experiments --dataset cifar10 --train --resume --device cuda` restores `resume.pt` (weights, optimizer, scheduler and RNG) when available. It preserves the cumulative budget. A legacy best-only checkpoint restarts optimizer moments and documents that limitation. The measured CIFAR-10 run required this legacy recovery after epoch 4; see `results/cifar10_recovery.json` and the notebook discussion. Future uninterrupted default runs need not reproduce that warm-restarted trajectory exactly. Completed/evaluated experiments are loaded rather than trained further.

## Measured results

The accuracy upgrade is complete. Candidate architectures, augmentation, regularization, full-state recovery, validation diagnostics, and separate final test evaluation are implemented in `experiments/`. The selected models were chosen by validation accuracy and evaluated once on the official test partitions. The notebook loads these checkpoints automatically and never starts training by default.

| Model | Parameters | Test accuracy | Macro precision | Macro recall | Macro F1 | Active training | Selected epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| DigitNetPlus / MNIST | 435,754 | 99.65% | 0.99651 | 0.99647 | 0.99649 | 24 min 40 s | 45 / 60 |
| VisionNetWide / CIFAR-10 | 5,849,050 | 93.32% | 0.93426 | 0.93320 | 0.93328 | about 7 h 27 min | 148 / 200 |

CIFAR-10 duration includes logged interrupted work and conservative recovery allowances, but excludes inactive gaps and the discarded initial attempt. Adding its reserved five minutes gives about 59 min 46 s of the approved 120-minute allowance. Neither run triggered a temperature or memory safety stop. These cross-dataset scores measure different tasks, not a controlled architecture ranking.
