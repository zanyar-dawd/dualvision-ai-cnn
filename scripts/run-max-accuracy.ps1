param(
    [string]$ExperimentId = "accuracy-max-20261017",
    [string]$Config = "configs\accuracy_max.json"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$stateDirectory = Join-Path $projectRoot "results\$ExperimentId\runner"
New-Item -ItemType Directory -Path $stateDirectory -Force | Out-Null

function Invoke-CandidateTraining([string]$Dataset) {
    $checkpointDirectory = Join-Path $projectRoot "checkpoints\$ExperimentId\$Dataset"
    $arguments = @(
        "-u", "-m", "experiments.train_v2", "--train",
        "--dataset", $Dataset,
        "--device", "cuda",
        "--experiment-id", $ExperimentId,
        "--config", $Config
    )
    if (Test-Path -LiteralPath (Join-Path $checkpointDirectory "resume.pt")) {
        $arguments += "--resume"
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Dataset training failed with exit code $LASTEXITCODE."
    }
}

try {
    "running" | Set-Content -LiteralPath (Join-Path $stateDirectory "status.txt") -Encoding UTF8
    Invoke-CandidateTraining "cifar10"
    Invoke-CandidateTraining "mnist"
    & $python -u -m scripts.finalize_accuracy_max `
        --experiment-id $ExperimentId --config $Config
    if ($LASTEXITCODE -ne 0) {
        throw "Final evaluation failed with exit code $LASTEXITCODE."
    }
    "complete" | Set-Content -LiteralPath (Join-Path $stateDirectory "status.txt") -Encoding UTF8
}
catch {
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $stateDirectory "failure.txt") -Encoding UTF8
    "failed" | Set-Content -LiteralPath (Join-Path $stateDirectory "status.txt") -Encoding UTF8
    throw
}
