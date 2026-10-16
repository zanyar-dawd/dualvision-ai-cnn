"""Build the submission with complete inline code, using only the standard library."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def inline_module(relative_path):
    source = (ROOT / relative_path).read_text(encoding='utf-8')
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    excluded = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('src.'):
            excluded.update(range(node.lineno - 1, node.end_lineno))
    return ''.join(line for index, line in enumerate(lines) if index not in excluded)


def build():
    cells = []
    def cell(kind, source):
        value = {'cell_type': kind, 'id': f'cell-{len(cells):03}', 'metadata': {}, 'source': source.strip() + '\n'}
        if kind == 'code':
            value.update(execution_count=None, outputs=[])
        cells.append(value)
    def md(source):
        cell('markdown', source)
    def code(source):
        cell('code', source)

    md('''# DualVision AI — Intelligent Digit and Object Recognition
## Homework 5 — Building a Model
**Student:** [Your name] · **Student ID:** [Your ID]  
**Course:** [Course name] · **Instructor:** [Instructor name] · **Submission date:** [Date]

**Submission status:** The implementation is complete; real-data experimental results are **pending** until the explicitly enabled training cells finish. Synthetic smoke-test results are never benchmark evidence. Fill in student details and execute the real experiments before submission.
''')
    md('''## 1. Abstract
This project implements two custom PyTorch convolutional neural networks: DigitNet for MNIST handwritten digits and VisionNet for CIFAR-10 object categories. A stratified validation partition selects checkpoints without using test labels. Training records loss, accuracy, runtime and model size; final evaluation reports confusion matrices and macro/per-class precision, recall and F1. An upload interface and a mathematically checked Grad-CAM implementation support inference and interpretation. No accuracy claim is made before a real experiment has completed. Targets of approximately 99% for MNIST and 85–90% for CIFAR-10 are aspirations, not measured results or guarantees.
''')
    md(r'''## 2. Introduction to computer vision and CNNs
Image recognition maps pixel arrays to discrete labels. Convolution shares local filters over an image, building representations from edges and textures to shapes. ReLU introduces nonlinearity; pooling or strided convolution reduces spatial size. Batch normalization stabilizes feature scales and dropout regularizes the classifier. Residual connections help gradients pass through deeper networks.

Both models minimize multiclass cross-entropy on logits:
\[
\mathcal{L}=-\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(z_{i,y_i})}{\sum_{c=1}^{10}\exp(z_{i,c})}.
\]
Softmax is applied only for inference probabilities. Test accuracy is the total number of correct predictions divided by the total test count, not an unweighted mean of batch accuracies.
''')
    md('''## 3. Project objectives
1. Implement and verify two original CNNs using official torchvision datasets.
2. Separate training, validation and final test evaluation; record reproducible settings.
3. Analyze actual errors and report macro and per-class metrics.
4. Provide portable notebook-local inference, uploads and Grad-CAM.
5. Respect a 4 GB GPU and bounded experiment duration.
''')
    md('''## 4. Dataset descriptions
| Dataset | Input | Official training | Training subset | Validation | Held-out test | Classes |
|---|---|---:|---:|---:|---:|---|
| MNIST | 1 × 28 × 28 | 60,000 | 54,000 | 6,000 | 10,000 | Digits 0–9 |
| CIFAR-10 | 3 × 32 × 32 | 50,000 | 45,000 | 5,000 | 10,000 | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |

Validation uses a stratified 10% split from the official training partition with seed 42. Per-channel population mean and standard deviation are measured **only on the selected training subset**, before augmentation. The same values normalize validation, test and uploads. CIFAR-10 training uses random crops with four-pixel padding and horizontal flips; validation/test preprocessing is deterministic. The test partition is never used by training, scheduling or checkpoint selection.
''')
    md('''## Environment and execution controls
Use **Python 3.12** and select the project environment as the notebook kernel. For your teacher's **CPU-only laptop**, install:

```powershell
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.2.6 pandas==2.2.3 matplotlib==3.10.3 scikit-learn==1.6.1 Pillow==11.2.1 jupyterlab==4.4.3 ipywidgets==8.1.7 ipykernel==6.29.5 nbformat==5.10.4 nbclient==0.10.2 pytest==8.3.5
```

The final notebook embeds compact trained checkpoints, saved test metrics, figures and demonstration images. Run all cells in a fresh writable folder: they restore and verify these assets, load weights on CPU when CUDA is absent, and display the measured report **without downloading datasets or retraining**. Inference on uploads still executes the real trained model. For the original GPU training workstation, use the same PyTorch pair from the `cu126` index instead of `cpu`; `nvidia-smi` reports driver support, not Toolkit installation. A broken CUDA operation raises an error instead of silently claiming acceleration.

Run all cells in order. The full implementation below needs no `src/` imports or hidden training script. By default existing compatible checkpoints load and missing models stay pending. For a new real experiment, explicitly set `ALLOW_TRAINING=True` and `DOWNLOAD_DATA=True` after approving downloads and runtime. For another training run use a **new** `EXPERIMENT_ID` and set `RETRAIN=True`; existing checkpoints are never overwritten by retraining. Keep the baseline settings fixed before examining test results.

Default maximums: DigitNet 15 epochs / 30 minutes; VisionNet 50 epochs / 115 minutes. Five additional minutes of the approved 120-minute CIFAR-10 allowance were reserved for an interrupted preliminary attempt; the final run starts afresh with seed 42. Real GPU training checks the time budget at batch boundaries, samples telemetry every five seconds, and stops at the project's conservative 85°C or 95% device-memory policy. These are project safeguards, not manufacturer-rated limits. It preserves the validation-best checkpoint and interrupted state on a safety stop. Early stopping may end sooner. Float32 training and batches of 128/64 avoid mixed-precision assumptions about the T1000. Workers are zero for Windows notebook reliability. Reduce the batch size in a new validation experiment if another application consumes VRAM.
''')
    code('''import os
from pathlib import Path

# These defaults never start real-data training or dataset downloads.
ALLOW_TRAINING = False
DOWNLOAD_DATA = False
RETRAIN = False
EXPERIMENT_ID = "baseline"
DEVICE_PREFERENCE = os.environ.get("DUALVISION_DEVICE", "auto")  # CPU portability is explicitly verified
SEED = 42
# The smoke runner sets these environment variables in its child process only.
SMOKE_TEST = os.environ.get("DUALVISION_SMOKE", "0") == "1"
PROJECT_ROOT = Path(os.environ.get("DUALVISION_ROOT", ".")).resolve()
if SMOKE_TEST:
    EXPERIMENT_ID = "notebook-smoke"
    print("SYNTHETIC SMOKE TEST: all displayed scores are software checks, NOT MNIST/CIFAR-10 results.")
''')
    code(inline_module('src/utilities/assets.py'))
    code('''# Populated by scripts.package_submission after real experiments finish.
SUBMISSION_ARCHIVE_B64 = ""
if SUBMISSION_ARCHIVE_B64 and not SMOKE_TEST and EXPERIMENT_ID == "baseline":
    restore_submission_assets(SUBMISSION_ARCHIVE_B64, PROJECT_ROOT)
''')
    code(inline_module('src/utilities/runtime.py'))
    code('''seed_everything(SEED)
DEVICE, RUNTIME = select_device(DEVICE_PREFERENCE)
print(json.dumps(RUNTIME, indent=2))
print("GPU tensor computation verified" if DEVICE.type == "cuda" else "CPU tensor computation verified")
''')
    md('''## 5. MNIST methodology
DigitNet has two 3×3 convolutional stages (32 and 64 channels), each with batch normalization, ReLU and 2×2 max-pooling. Dropout2d regularizes features; a 128-unit hidden classifier uses ReLU and dropout before ten logits. This modest architecture retains spatial information needed for handwritten digits. AdamW uses learning rate 0.001 and weight decay 0.0001. A cosine schedule lowers the learning rate; validation accuracy selects the best epoch with validation loss as a tie-breaker. Patience is five epochs.

The next cells also define reusable routines for both experiments. Source-equivalent implementations are included inline to make this notebook independently executable.
''')
    md('''## 6. Complete executable MNIST implementation
### Architectures
Both model classes are defined here so shared experiment helpers can instantiate either model. VisionNet's methodology is explained in Section 8.
''')
    code(inline_module('src/models/cnn.py'))
    md('''### Data partitions and preprocessing
Official partition sizes and class ordering are checked. Split indices and their SHA-256 fingerprint are recorded with each experiment. Synthetic smoke mode uses disjoint FakeData offsets and writes to a separate directory.
''')
    code(inline_module('src/utilities/data.py'))
    md('''### Training, mini-batch verification and checkpointing
Training and validation explicitly use `train()` and `eval()`; validation disables gradients. Loss is weighted by batch sample count. Mini-batch checks use a disposable model; RNG and loader shuffle state are reset before actual training. Checkpoints store CPU `state_dict` tensors and metadata; loading uses `weights_only=True`, strict architecture matching and experiment metadata checks. Atomic recovery snapshots save optimizer, scheduler and random streams after each complete epoch. Explicit resumption stays within the cumulative time budget. Legacy best-weight-only recovery restarts optimizer moments and is recorded as a methodological limitation. An interrupted training run is not eligible for test evaluation until bounded training is finalized.
''')
    code(inline_module('src/utilities/hardware.py'))
    code(inline_module('src/training/engine.py'))
    md('''### Evaluation and reporting helpers
Final test predictions are cached against the selected checkpoint's SHA-256 hash and reused on reruns. The cache validates its metrics against stored predictions. Precision, recall and F1 are **macro-averaged** across all ten classes (equal class weight); the full report also includes support and weighted averages. Undefined per-class metrics use zero explicitly.
''')
    code(inline_module('src/evaluation/metrics.py'))
    code(inline_module('src/visualization/plots.py'))
    md(r'''### Explainability helper: Grad-CAM
For class score $y^c$ and final spatial features $A^k$, the weights and map are
\[
\alpha_k^c=\frac{1}{HW}\sum_{i,j}\frac{\partial y^c}{\partial A_{ij}^k},\qquad
L^c=\operatorname{ReLU}\left(\sum_k\alpha_k^c A^k\right).
\]
VisionNet's evaluation head is global average pooling followed by a linear layer. Consequently, the raw gradient-based map must equal the classifier-weight CAM divided by the spatial area. The implementation checks this identity for every map. Tests also check shape, range, finite values, class sensitivity, zero-weight degeneracy and hook removal. These validate the calculation; they do not prove causal explanation. A zero map is explicitly reported instead of being replaced with arbitrary colors.
''')
    code(inline_module('src/visualization/gradcam.py'))
    code(inline_module('src/workflow.py'))
    code('''CONFIGS = {name: dict(config) for name, config in DEFAULT_CONFIGS.items()}
print(json.dumps(CONFIGS, indent=2))
experiments = {}
experiments["mnist"] = run_experiment(
    "mnist", root=PROJECT_ROOT, experiment_id=EXPERIMENT_ID, config=CONFIGS["mnist"],
    allow_training=ALLOW_TRAINING, download=DOWNLOAD_DATA, retrain=RETRAIN,
    smoke=SMOKE_TEST, device_preference=DEVICE_PREFERENCE, seed=SEED)
''')
    md('''## 7. MNIST experimental results
The following cell displays only measured results from the current experiment. **Pending if no real checkpoint exists.** Learning curves include the validation-selected epoch. Example images are the first matching test indices, avoiding cherry-picking; a panel states when no correct or incorrect examples exist.
''')
    code('''from IPython.display import display, Markdown, Image as DisplayImage

def display_experiment(experiment):
    if experiment is None:
        display(Markdown("**PENDING:** no completed real-data training/evaluation is available."))
        return
    meta, metrics = experiment["metadata"], experiment["evaluation"]
    if meta["smoke"]:
        display(Markdown("**SYNTHETIC SMOKE ONLY — these are not dataset results.**"))
    display(pd.DataFrame([{key: metrics[key] for key in
        ["count", "accuracy", "loss", "macro_precision", "macro_recall", "macro_f1"]}]))
    display(pd.DataFrame(metrics["report"]).T)
    print(f"Parameters: {meta['parameters']:,}; training: {meta['training_seconds']:.1f}s; "
          f"selected epoch: {meta['best_epoch']}; stop: {meta['stop_reason']}")
    print("Training-subset normalization:", meta["mean"], meta["std"])
    if meta.get("telemetry"):
        print(f"Original training device: {meta['runtime']['gpu']}; "
              f"sampled peak temperature: {meta['telemetry']['peak_temperature_c']:.0f} C; "
              f"sampled peak device memory: {meta['telemetry']['peak_device_memory_used_mib']:.0f} MiB")
    for path in experiment["figures"]:
        display(DisplayImage(filename=str(path)))
    display(Markdown("Observed confusion pairs:\\n" + observed_errors(metrics, meta["classes"])))

display_experiment(experiments["mnist"])
''')
    md('''## 8. CIFAR-10 methodology
VisionNet is an original narrow residual CNN, trained from scratch without pretrained weights. A 32-channel stem feeds three stages at widths 32, 64 and 128, with two residual blocks per stage. Each block contains two 3×3 convolutions and batch normalization; downsampling uses stride 2 with a 1×1 projected shortcut. Global average pooling and dropout (0.25) feed ten logits. There is no ImageNet pretraining or test-time augmentation.

AdamW (learning rate 0.001, weight decay 0.0005) and cosine scheduling provide a simple bounded baseline. Batch size is 64, maximum epochs 50, patience 12, and time budget 115 minutes (plus five minutes reserved for the interrupted preliminary attempt). This is a practical experiment budget, not a promise of 85–90% accuracy. Stronger augmentation, longer schedules or wider models are future validation-only investigations rather than changes made in response to this test set.
''')
    md('''## 9. Complete executable CIFAR-10 implementation
`ResidualBlock`, `VisionNet`, data augmentation, optimizer/scheduler, training, evaluation and Grad-CAM are fully implemented above in notebook-local cells. This call runs the distinct CIFAR-10 experiment using those definitions. Its weights, split, curves and cached test predictions are stored independently from DigitNet.
''')
    code('''experiments["cifar10"] = run_experiment(
    "cifar10", root=PROJECT_ROOT, experiment_id=EXPERIMENT_ID, config=CONFIGS["cifar10"],
    allow_training=ALLOW_TRAINING, download=DOWNLOAD_DATA, retrain=RETRAIN,
    smoke=SMOKE_TEST, device_preference=DEVICE_PREFERENCE, seed=SEED)
''')
    md('''## 10. CIFAR-10 experimental results
The cell reports the selected model on untouched official test images once, or loads predictions for that exact checkpoint. **Pending until training completes.** The Grad-CAM example uses fixed test index zero; its color intensity is relative within that image, not a calibrated importance score.
''')
    code('display_experiment(experiments["cifar10"])')
    md('''## 11. Visualizations, error analysis and prediction demonstrations
Confusion matrices, mistakes and correct examples above derive from the saved test predictions. The listed confusion counts describe actual outcomes. Similar outlines, handwriting ambiguity, low resolution, clutter or pose are **possible hypotheses to inspect in the displayed mistakes**, not established explanations. If results are pending, no empirical mistake pattern can be stated. If no mistakes occur in a small smoke fixture, that says nothing about benchmark performance.

For a digit upload, use a single clearly written digit with a plain background. Processing removes background intensity, handles ink polarity (with manual override), fits a glyph inside 20×20 pixels, pads to 28×28, and centers its intensity centroid. Color uploads are converted to RGB and resized to 32×32. This can remove useful detail. CIFAR-10 recognizes only ten predefined classes and will still assign an unfamiliar image one of those labels. Softmax scores are not calibrated certainty.
''')
    code(inline_module('src/utilities/inference.py'))
    code('''dashboard = prediction_dashboard(experiments, DEVICE)

# Reliable non-interactive fallback after completing training:
# result = predict_file("my_digit.png", PROJECT_ROOT / "checkpoints" / EXPERIMENT_ID / "mnist" / "best.pt", "mnist", DEVICE)
# display(result["processed_image"])
# print(result["top_classes"])
# result = predict_file("my_object.jpg", PROJECT_ROOT / "checkpoints" / EXPERIMENT_ID / "cifar10" / "best.pt", "cifar10", DEVICE)
# print(result["top_classes"])
''')
    code('''# Fixed official test index 0 for each dataset; included as small demonstration PNGs.
# This runs actual inference on the current device, even when saved report metrics are reused.
for name, experiment in experiments.items():
    if experiment is None or experiment["metadata"]["smoke"]:
        continue
    example = experiment["output_dir"] / "examples" / "test_index_0.png"
    if example.exists():
        prediction = predict_file(example, experiment["checkpoint"], name, DEVICE)
        print(f"{name}: real checkpoint inference on {DEVICE}; top classes {prediction['top_classes']}")
        display(prediction["processed_image"].resize((168, 168)))
''')
    md('''## 12. Comparative discussion
MNIST has centered grayscale symbols and limited background variation. CIFAR-10 has color, viewpoint changes, backgrounds and more diverse within-class appearances at only 32×32 pixels. VisionNet therefore uses more feature processing and residual connections. Parameter count, measured training time and macro metrics below summarize each experiment.

**Cross-dataset accuracies measure different tasks and must not be interpreted as a controlled ranking of architectures.** Neither accuracy targets nor synthetic scores belong in the experimental results. A fair architecture comparison would train both architectures on the same dataset with a controlled budget and validation protocol.
''')
    code('''comparison = comparison_table(experiments)
display(comparison)
if all(exp is not None and not exp["metadata"]["smoke"] for exp in experiments.values()):
    for name, exp in experiments.items():
        print(f"Measured {name}: {exp['evaluation']['accuracy']:.2%} accuracy; macro F1 {exp['evaluation']['macro_f1']:.4f}.")
else:
    print("Real cross-dataset results are pending; no empirical accuracy comparison is claimed.")
''')
    md('''## 13. Limitations and possible improvements
This is a single-seed baseline, not a statistically comprehensive study. Fixed seeds improve repeatability, but exact results can differ across PyTorch versions, hardware and numerical kernels. The runtime record captures versions and deterministic settings. Training time is hardware-dependent and excludes dataset preparation and final evaluation. The wall-clock budget may stop before cosine annealing finishes. If it does, report that fact and the observed scores.

Training metrics are collected with dropout enabled and, for CIFAR-10, random augmentation. Validation metrics use deterministic preprocessing and evaluation mode. Their gap therefore reflects different measurement conditions as well as possible generalization differences.

Uploaded images may differ substantially from the training distribution. Neither model has an unknown class. Grad-CAM is coarse and highlights associations rather than proving causation. Test metrics have sampling uncertainty; repeated seeds and confidence intervals would strengthen conclusions. Future changes should be selected using validation data and a separately planned evaluation protocol. Potential improvements include small digit translations, CIFAR-10 color augmentation, label smoothing, validation-calibrated probabilities and repeated-seed experiments within the hardware budget.
''')
    md('''## 14. Conclusion
The notebook supplies two custom CNN implementations and a reproducible procedure for model selection, evaluation, visualization and inference. Scientific conclusions depend on completed real-data experiments. Until those runs finish, the submission remains an implemented and smoke-tested pipeline with **pending experimental results**, not a completed empirical report.
''')
    code('''real_complete = all(exp is not None and not exp["metadata"]["smoke"] for exp in experiments.values())
print("REAL EXPERIMENTS COMPLETE — inspect the measured results above and fill in student details."
      if real_complete else "SUBMISSION INCOMPLETE: real MNIST/CIFAR-10 training, test results and scientific conclusions are pending.")
''')
    md('''## 15. Reproduction and submission checklist
- Restart the Python 3.12 kernel and run all cells; ensure no errors.
- Confirm actual CUDA tensor verification (or honestly report CPU execution).
- Confirm both official datasets, training-only normalization, disjoint validation indices and separate checkpoints.
- Complete real training and cached final evaluation for both models; synthetic smoke mode is insufficient.
- Confirm curves/metrics come from the same checkpoint-bound artifacts.
- Inspect confusion pairs and mistakes; add observations supported by the actual images.
- Fill in student details and retain executed real outputs in this `.ipynb`.
- Submit **Homework_5_DualVision.ipynb**, not a replacement `.py` file.
- Do not commit datasets, virtual environments or standalone checkpoint files. This notebook intentionally embeds the two compact selected state dictionaries so it can run independently on CPU.

Repository QA commands (not required by this standalone notebook): `python -m pytest`, `python -m scripts.run_experiments --smoke`, and `python -m scripts.smoke_notebook`. The notebook smoke runner copies this notebook into an empty temporary directory, executes all cells against small synthetic fixtures, and checks that no local source modules are needed.
''')
    md('''## 16. References
1. Teacher MNIST learning resource: [Elgendy, chapter 3 notebook](https://github.com/moelgendy/deep_learning_for_vision_systems/blob/master/chapter_03/mnist_cnn.ipynb).
2. Teacher CIFAR-10 learning resource: [Elgendy, chapter 4 notebook](https://github.com/moelgendy/deep_learning_for_vision_systems/blob/master/chapter_04/chapter_4_improve_cifar_90_percent.ipynb). These resources inform the task; their code and reported scores are not copied as this project's results.
3. [PyTorch official installation combinations](https://pytorch.org/get-started/previous-versions/) and [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).
4. [torchvision MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) and [CIFAR-10](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html).
5. [CIFAR-10 dataset, Krizhevsky](https://www.cs.toronto.edu/~kriz/cifar.html).
6. Selvaraju et al., [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391), ICCV 2017.
7. He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), CVPR 2016.
8. [scikit-learn classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html).
9. [MIT Open Learning mirror of the unmodified official CIFAR-10 archive](https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python). Used after the original host transfer failed; pinned revision `a48007227f9e2cd0af96175f4afb5ac4965e261b`, verified MD5 `c58f30108f718f92721af3b95e74349a`. torchvision still performs the integrity checks.
''')
    result_paths = {name: ROOT / 'results' / 'baseline' / name / 'evaluation.json' for name in ['mnist', 'cifar10']}
    if all(path.exists() for path in result_paths.values()):
        measured = {name: json.loads(path.read_text(encoding='utf-8')) for name, path in result_paths.items()}
        trained = {name: json.loads((path.parent / 'training.json').read_text(encoding='utf-8')) for name, path in result_paths.items()}
        if all(value['source'] == 'torchvision-official' for value in measured.values()):
            summary = ('Measured held-out test accuracy: **DigitNet '
                       f"{measured['mnist']['accuracy']:.2%}**, **VisionNet {measured['cifar10']['accuracy']:.2%}** "
                       '(10,000 test images each). Both models were trained from scratch; test scores were not used for selection.')
            for item in cells:
                if item['cell_type'] != 'markdown':
                    continue
                if item['source'].startswith('# DualVision'):
                    item['source'] = item['source'].split('**Submission status:**')[0] + '**Submission status:** Real experiments complete. Fill in student/course details before submitting.\n\n' + summary + '\n'
                if item['source'].startswith('## 1. Abstract'):
                    item['source'] = ('## 1. Abstract\n'
                        'Two custom PyTorch convolutional networks were trained from scratch on MNIST and CIFAR-10. '
                        'A stratified training-only validation split selected checkpoints; the official test partitions '
                        'were evaluated after selection. The report includes loss and accuracy curves, confusion matrices, '
                        'macro/per-class metrics, observed mistakes, and independently checked Grad-CAM calculations. '
                        'The notebook embeds selected weights and measured artifacts, and supports CPU inference without '
                        'retraining or dataset downloads.\n\n' + summary + '\n')
                if item['source'].startswith('## 7. MNIST experimental results'):
                    item['source'] = item['source'].replace('**Pending if no real checkpoint exists.**',
                        f"**Completed:** {measured['mnist']['accuracy']:.2%} test accuracy, loss {measured['mnist']['loss']:.5f}; selected epoch {trained['mnist']['best_epoch']}; {trained['mnist']['training_seconds']/60:.2f} minutes training.")
                if item['source'].startswith('## 10. CIFAR-10 experimental results'):
                    item['source'] = item['source'].replace('**Pending until training completes.**',
                        f"**Completed:** {measured['cifar10']['accuracy']:.2%} test accuracy, loss {measured['cifar10']['loss']:.5f}; selected epoch {trained['cifar10']['best_epoch']}; {trained['cifar10']['training_seconds']/60:.2f} minutes training.")
                if item['source'].startswith('## 12. Comparative discussion'):
                    discussion = ROOT / 'results/baseline/discussion.md'
                    if discussion.exists():
                        item['source'] += '\n' + discussion.read_text(encoding='utf-8') + '\n'
                if item['source'].startswith('## 14. Conclusion'):
                    item['source'] = '## 14. Conclusion\n' + summary + '\n\nThe saved validation-selected weights, measured figures and CPU-executable inference are included in this notebook. The results support the expected difference in task difficulty, while this single-seed cross-dataset study does not isolate architecture quality. No further tuning was performed using test scores.\n'
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3 (ipykernel)',
                'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.12.11'}},
                'nbformat': 4, 'nbformat_minor': 5}
    path = ROOT / 'Homework_5_DualVision.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'Built {path.name}: {len(cells)} cells, complete inline implementations.')


if __name__ == '__main__':
    build()
