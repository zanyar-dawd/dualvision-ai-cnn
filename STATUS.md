# DualVision AI status

Verified on 2026-10-16. **The baseline implementation and empirical report are complete. The requested higher-accuracy experiment is prepared but not yet trained.** The required submission is `Homework_5_DualVision.ipynb`. Fill in its student/course/instructor/date placeholders before submitting.

## Requested accuracy upgrade — pending additional runtime approval

Targets: MNIST 99.8%, CIFAR-10 95.5–96%. These have **not** been achieved or guaranteed. The new `experiments/` implementations and `configs/accuracy_v2.json` are isolated from the completed baseline. See [ACCURACY_UPGRADE_PLAN.md](ACCURACY_UPGRADE_PLAN.md) for the concrete recipe, safeguards and commands.

- **Completed preparation:** DigitNetPlus and VisionNetWide, training-only augmentation, mixup for CIFAR-10, warmup/cosine schedules, clean training/validation diagnostics, early stopping, full-state recovery, separate final test evaluation, and CPU checkpoint inference support.
- **Verified:** 32 tests passed, including CPU/CUDA optimizer steps and exact recovery; both synthetic CPU training pipelines completed. Full-batch GPU preflight measured 200.30 MiB and 738.47 MiB peak allocated memory. Five timed steps per model project about 5.26 minutes and 4.94 hours of computation alone, excluding preprocessing and validation. This is not a full training measurement.
- **Pending approval:** additional caps of 45 minutes MNIST and 12 hours CIFAR-10, running sequentially with temperature/memory monitoring. The user's original instruction requires confirmation before extended training.
- **Not completed for candidates:** real training, final accuracy measurements, trained-model upload demonstrations and integration into the final self-contained notebook. Candidate synthetic scores are not experimental results.

The completed baseline notebook passed submission verification again; its measured scores below are unchanged. `results/accuracy_v2_full_tests.xml` records the new 32-test run. The following phase table and submission checklist describe the completed baseline, not the pending upgrade.

## Real experimental results

| Model | Test accuracy | Test loss | Macro F1 | Active training | Epochs completed | Selected epoch |
|---|---:|---:|---:|---:|---:|---:|
| DigitNet / MNIST | 99.25% (9,925 / 10,000) | 0.02677 | 0.99247 | 320.85 s (5 min 21 s) | 15 | 11 |
| VisionNet / CIFAR-10 | 90.92% (9,092 / 10,000) | 0.39882 | 0.90920 | about 3,285.77 s (54 min 46 s) | 50 | 49 |

Both models stopped at their planned epoch limits. Early stopping was enabled but did not trigger. Validation alone selected checkpoints; final test predictions were saved after selection and reused for report execution. Cross-dataset scores describe different tasks and are not a controlled ranking of architectures.

CIFAR-10 duration is cumulative active time across recoveries, including logged incomplete-epoch work and conservative five-second allowances per recovery; it excludes inactive gaps. Five further minutes were reserved for a discarded initial attempt, giving about 59 min 46 s against the approved 120-minute total. Its epoch-4 recovery restarted AdamW moments and the RNG stream; the later epoch-23 recovery restored full optimizer/scheduler/RNG state. These deviations are disclosed in the notebook. A simple uninterrupted default run is not claimed to reproduce the recovered trajectory exactly.

## Phase completion

| Phase | Verified outcome |
|---|---|
| 1. Environment inspection | Complete: Windows, Quadro T1000 4 GB, driver 597.16, existing Python 3.14 and ArcGIS installation inspected. |
| 2. Isolated environment | Complete: project-local Python 3.12.11 and `.venv`; torch 2.8.0+cu126 / torchvision 0.23.0+cu126; dependency check and actual CUDA matrix operation passed. |
| 3. DigitNet implementation | Complete: configured batch 128 forward/backward/optimizer check, CPU/CUDA tests, synthetic pipeline and checkpoint tests passed. |
| 4. DigitNet experiment | Complete: official MNIST, 54,000/6,000/10,000 train/validation/test split, saved validation-best weights and measured results. |
| 5. VisionNet implementation | Complete: custom residual CNN, configured batch 64 forward/backward/optimizer check, CPU/CUDA tests and synthetic pipeline passed. |
| 6. VisionNet experiment | Complete: verified official CIFAR-10, 45,000/5,000/10,000 split, bounded monitored training, selected weights and measured results. |
| 7. Explainability and inference | Complete: analytical Grad-CAM identity checked; real Grad-CAM executed on CPU; both upload-button callbacks and file fallbacks passed using fixed official test images. |
| 8. Real figures and error analysis | Complete: curves, confusion matrices, correct/mistaken images, per-class reports, CIFAR-10 Grad-CAM and evidence-based discussion. All nine report PNGs match their notebook outputs byte-for-byte. |
| 9. Self-contained notebook | Complete: valid 44-cell `.ipynb`; complete inline implementation, 27 embedded assets, actual results and report sections. All 22 code cells executed successfully in a fresh CPU kernel. |
| 10. Submission QA | Complete: 23 unit/integration tests, isolated notebook smoke, real artifact audit, CPU-only report execution and final submission verification passed. No dataset or standalone weights tracked by Git. |

## Verification evidence

- `results/pytest.xml`: **23 passed**, no skipped GPU checks, 14 upstream Matplotlib/pyparsing deprecation warnings. A separate final notebook structure check passed; sandbox cache-write warnings did not affect assertions.
- `results/minibatch_verification.json`: both full configured training batches verified before training.
- `results/dataset_verification.json`: official partition counts and archive integrity verified. The original CIFAR-10 transfer failed; recovery from the pinned MIT Open Learning mirror matched the original MD5 exactly.
- `results/real_experiment_audit.json`: checkpoint metadata, validation selection, saved predictions, metrics, history CSVs, class reports and telemetry agree.
- `results/inference_verification.json`: both real CPU upload callbacks match the path-based prediction fallback. Fixed test index 0 predicts digit 7 and cat correctly; this is an interface check, not a new accuracy estimate.
- `results/cpu_notebook_verification.json`: all 22 code cells executed from an empty directory with CUDA hidden, no source modules, no datasets and no retraining. Real CPU Grad-CAM and both upload callbacks passed.
- `results/submission_verification.json`: all 27 embedded assets hash-verified, nine displayed report figures match saved PNGs, valid executed notebook with zero error outputs, technical readiness confirmed.
- `results/baseline/comparison.csv` and per-dataset JSON/CSV/PNG files contain the real measured results. Synthetic outputs remain isolated under ignored `results/smoke/`.

An initial final-notebook verification completed its cells but hit a Windows SQLite history-file lock during cleanup. The runners now use in-memory history and graceful kernel shutdown. The final verification was rerun and exited successfully, including cleanup; the exception was not suppressed.

## Hardware and environment

The reported MNIST run peaked at 70 C and 247 MiB sampled device-wide memory; CIFAR-10 peaked at 73 C and 619 MiB. These measurements may include concurrent device activity. Both stayed below the project's 85 C and 95% memory stop policies; neither reported run triggered a hardware stop. Raw telemetry is saved beside each checkpoint.

The standard Python 3.12.10 MSI failed with Windows Installer 2502/1603. A project-local uv runtime installed CPython 3.12.11 without executable/registry registration. Existing Python and ArcGIS installations were not modified. Driver-reported CUDA 13.2 was not treated as evidence of a Toolkit installation. `results/environment.json` records real CUDA computation, compute capability 7.5 and compiled sm_75 support; `results/environment-freeze.txt` records installed versions.

## Final checklist and limits

- **Completed:** both trained CNNs, true test metrics, saved checkpoints, plots, per-class reports, Grad-CAM, upload inference, comparison/error discussion, reproducibility settings, tests, documentation and the executed self-contained notebook.
- **Partially verified:** teacher portability has been verified on the CPU execution path with CUDA hidden and an empty working directory. A separate physical teacher laptop and a fresh CPU-only wheel installation were not available for testing. CPU installation instructions are in `TEACHER_QUICKSTART.md` and the notebook.
- **Remaining before baseline submission:** fill the student/course/instructor/date placeholders and submit the `.ipynb`. The baseline experiment is complete; the newly requested accuracy upgrade has the pending work listed above.

Default notebook settings are `ALLOW_TRAINING=False` and `DOWNLOAD_DATA=False`. Rebuilding the notebook resets its embedded archive and saved outputs; after source changes, rebuild, package and execute the submission again. Dataset folders, virtual environments, recovery snapshots and standalone checkpoints are Git-ignored. The notebook intentionally embeds only compact selected weights and report assets for independent CPU use.
