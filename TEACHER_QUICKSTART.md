# Run the submitted notebook on a CPU-only laptop

The final `Homework_5_DualVision.ipynb` embeds both compact trained checkpoints, measured test reports, figures and two small demonstration images. The full implementation is also inline. You do not need the repository, datasets, a GPU, a CUDA Toolkit or retraining to review the report and make predictions.

1. Put the notebook in a fresh writable folder.
2. Install Python 3.12 if needed, then create a separate virtual environment. The Windows commands below use the Python launcher to select 3.12 explicitly; an absolute path to a Python 3.12 executable also works.
3. Install CPU PyTorch and the notebook dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install numpy==2.2.6 pandas==2.2.3 matplotlib==3.10.3 scikit-learn==1.6.1 Pillow==11.2.1 jupyterlab==4.4.3 ipywidgets==8.1.7 ipykernel==6.29.5 nbformat==5.10.4 nbclient==0.10.2
.\.venv\Scripts\python.exe -m jupyterlab Homework_5_DualVision.ipynb
```

4. Run all cells with the defaults: `ALLOW_TRAINING=False`, `DOWNLOAD_DATA=False` and `EXPERIMENT_ID="baseline"`.
5. The notebook checks and restores its embedded files, uses a verified NVIDIA CUDA device when available, and otherwise automatically uses CPU. Failed CUDA verification displays its reason and falls back to CPU. The selected device is clearly printed. It displays the saved measured report without retraining. Uploading an image runs the actual trained model locally. Path-based `predict_file(...)` also works if widget rendering is unavailable.

For troubleshooting, set `DEVICE_PREFERENCE = "cpu"` in the controls cell and rerun the notebook. The default is `"auto"`. Keep `ALLOW_TRAINING = False` and `DOWNLOAD_DATA = False`. Prediction functions also accept `device="cpu"`; no model conversion or retraining is necessary.

The embedded-artifacts cell is collapsed to keep the report readable. Expand it if desired; it contains encoded model/report data, not hidden training code. Restored files appear under `checkpoints/baseline/` and `results/baseline/`. If an existing file has different contents, use a fresh folder instead of overwriting a previous experiment.

Reported test metrics are the original held-out experiment results, bound to the selected checkpoints by SHA-256. Running the default notebook displays those results without re-evaluating all 20,000 test images. The included fixed-index examples and uploaded files do execute fresh inference on the selected device.

For a new experiment, explicitly enable training and downloads and use a new experiment ID. Such a run is optional and is not needed to grade the saved submission. Fill in the student/course placeholders before final submission.

Verification on the development workstation executes the entire notebook with CUDA hidden, in an empty directory, using only its embedded assets. The installed development wheel contains CUDA libraries, so this verifies the CPU execution path rather than a separate physical teacher laptop or fresh CPU-wheel installation. The documented CPU wheel pair provides the same model APIs.
