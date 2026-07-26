param(
    [switch]$KeepDatasetArchive
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "The project environment is missing. Run scripts\setup_windows.ps1 first."
}

Push-Location $ProjectRoot
try {
    Invoke-Checked -Description "CUDA environment check" -Command {
        & $PythonExe ".\scripts\check_environment.py" --require-cuda
    }

    $DownloadArguments = @(
        ".\scripts\download_cifar10c.py",
        "--data-root",
        ".\data"
    )
    if (-not $KeepDatasetArchive) {
        $DownloadArguments += "--remove-archive"
    }
    Invoke-Checked -Description "CIFAR-10-C download and verification" -Command {
        & $PythonExe @DownloadArguments
    }

    $CleanDownloadArguments = @(
        ".\scripts\download_cifar10.py",
        "--data-root",
        ".\data"
    )
    if (-not $KeepDatasetArchive) {
        $CleanDownloadArguments += "--remove-archive"
    }
    Invoke-Checked -Description "Clean CIFAR-10 download and verification" -Command {
        & $PythonExe @CleanDownloadArguments
    }

    # Isolate pytest from stale or restricted Windows temp/cache directories.
    # The GUID makes the directory user-created and unique for every run.
    $PytestTemp = Join-Path `
        $ProjectRoot `
        ("outputs\pytest-temp-" + [Guid]::NewGuid().ToString("N"))
    Invoke-Checked -Description "Unit tests" -Command {
        & $PythonExe -m pytest `
            -q `
            -p no:cacheprovider `
            --basetemp $PytestTemp
    }

    # Avoid Hugging Face Xet/CAS reconstruction failures on unstable networks.
    # The standard HTTPS path preserves the same official model files and cache.
    $env:HF_HUB_DISABLE_XET = "1"
    $env:HF_HUB_DOWNLOAD_TIMEOUT = "120"
    $env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
    $PilotArguments = @(
        "-m",
        "promptfragc.runner",
        "--config",
        ".\configs\pilot.json"
    )
    $PilotMetrics = Join-Path $ProjectRoot "outputs\pilot\raw_metrics.csv"
    if (
        (-not (Test-Path -LiteralPath $PilotMetrics)) -or
        ((Get-Item -LiteralPath $PilotMetrics).Length -eq 0)
    ) {
        # A failed model download writes a manifest before producing any result
        # rows. It is safe to replace that empty attempt on the next launch.
        $PilotArguments += "--overwrite"
    }
    Invoke-Checked -Description "Pilot inference experiment" -Command {
        & $PythonExe @PilotArguments
    }
    Invoke-Checked -Description "Pilot statistical analysis" -Command {
        & $PythonExe -m promptfragc.analyze `
            --input ".\outputs\pilot\raw_metrics.csv" `
            --output-dir ".\outputs\pilot\analysis"
    }
}
finally {
    Pop-Location
}

Write-Output "Pilot complete. Open outputs\pilot\analysis for tables and figures."
