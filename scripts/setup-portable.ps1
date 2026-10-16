param([switch]$SkipPython)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.downloads/uv-cache'
$env:PIP_CACHE_DIR = Join-Path (Get-Location) '.downloads/pip-cache'
if (-not (Test-Path .runtime/uv/uv.exe)) {
    New-Item -ItemType Directory -Force .downloads | Out-Null
    Invoke-WebRequest 'https://github.com/astral-sh/uv/releases/download/0.8.15/uv-x86_64-pc-windows-msvc.zip' -OutFile .downloads/uv.zip
    Expand-Archive -LiteralPath .downloads/uv.zip -DestinationPath .runtime/uv -Force
}
if (-not $SkipPython) {
    & ./.runtime/uv/uv.exe python install 3.12.11 --install-dir .runtime/python --no-bin --no-registry
    if ($LASTEXITCODE -ne 0) { throw 'Project-local Python install failed.' }
}
$runtimePython = Join-Path (Get-Location) '.runtime/python/cpython-3.12.11-windows-x86_64-none/python.exe'
& $runtimePython -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'venv creation failed.' }
$projectPython = Join-Path (Get-Location) '.venv/Scripts/python.exe'
& $projectPython -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126 --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch install failed.' }
& $projectPython -m pip install -r requirements.txt --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw 'Dependency install failed.' }
& $projectPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
