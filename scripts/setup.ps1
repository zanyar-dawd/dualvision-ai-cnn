param([string]$Python312 = '', [switch]$InstallPython)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($InstallPython) {
    New-Item -ItemType Directory -Force .downloads | Out-Null
    $installer = Join-Path (Get-Location) '.downloads/python-3.12.10-amd64.exe'
    if (-not (Test-Path $installer)) {
        Invoke-WebRequest 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile $installer
    }
    $signature = Get-AuthenticodeSignature -LiteralPath $installer
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Python Software Foundation') {
        throw 'Python installer signature verification failed.'
    }
    $target = Join-Path (Get-Location) '.runtime/Python312'
    $process = Start-Process -FilePath $installer -ArgumentList @('/quiet', 'InstallAllUsers=0', "TargetDir=`"$target`"", 'PrependPath=0', 'Include_launcher=0', 'Include_test=0', 'AssociateFiles=0', 'Shortcuts=0') -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -ne 0) { throw "Python installation failed: $($process.ExitCode)" }
    $Python312 = Join-Path $target 'python.exe'
}
if (-not $Python312) { throw 'Supply -Python312 with an existing Python 3.12 executable or explicitly use -InstallPython.' }
& $Python312 -c "import sys; assert sys.version_info[:2] == (3,12), sys.version"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required.' }
& $Python312 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
$projectPython = Join-Path (Get-Location) '.venv/Scripts/python.exe'
& $projectPython -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $projectPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $projectPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
