param(
    [string]$PythonExe = "",
    [string]$TorchWheelUrl = "https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.9.1%2Bcu128-cp312-cp312-win_amd64.whl",
    [string]$TorchVisionWheelUrl = "https://mirrors.aliyun.com/pytorch-wheels/cu128/torchvision-0.24.1%2Bcu128-cp312-cp312-win_amd64.whl",
    [string]$PyPIIndexUrl = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
    [int]$PipTimeoutSeconds = 120,
    [int]$PipRetries = 10
)

$ErrorActionPreference = "Stop"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvRoot = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"
$PipNetworkArguments = @(
    "--timeout", $PipTimeoutSeconds,
    "--retries", $PipRetries,
    "--prefer-binary",
    "--no-input"
)

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

if (-not $PythonExe) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCommand) {
        $PythonExe = $PythonCommand.Source
    }
}
if (-not $PythonExe) {
    $Candidate = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
    if (Test-Path -LiteralPath $Candidate) {
        $PythonExe = $Candidate
    }
}
if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python 3.10+ was not found. Pass its full path with -PythonExe."
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Invoke-Checked -Description "Virtual environment creation" -Command {
        & $PythonExe -m venv $VenvRoot
    }
}

$PythonWheelTag = & $VenvPython -c "import platform,sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}-{platform.system().lower()}-{platform.machine().lower()}')"
if ($LASTEXITCODE -ne 0) {
    throw "Python compatibility check failed with exit code $LASTEXITCODE"
}
if ($PythonWheelTag.Trim() -ne "cp312-windows-amd64") {
    throw "This setup is pinned to 64-bit Python 3.12; detected $PythonWheelTag"
}

Invoke-Checked -Description "Packaging-tool installation" -Command {
    & $VenvPython -m pip install `
        --upgrade "setuptools>=75" wheel `
        --index-url $PyPIIndexUrl `
        @PipNetworkArguments
}
Invoke-Checked -Description "CUDA PyTorch installation" -Command {
    & $VenvPython -m pip install `
        $TorchWheelUrl $TorchVisionWheelUrl `
        --index-url $PyPIIndexUrl `
        --no-cache-dir `
        @PipNetworkArguments
}
Invoke-Checked -Description "Experiment dependency installation" -Command {
    & $VenvPython -m pip install `
        -r (Join-Path $ProjectRoot "requirements.txt") `
        --index-url $PyPIIndexUrl `
        @PipNetworkArguments
}
Invoke-Checked -Description "Editable project installation" -Command {
    & $VenvPython -m pip install --editable $ProjectRoot --no-build-isolation
}

$EnvironmentRoot = Join-Path $ProjectRoot "outputs\environment"
New-Item -ItemType Directory -Force -Path $EnvironmentRoot | Out-Null
$FreezeOutput = & $VenvPython -m pip freeze
if ($LASTEXITCODE -ne 0) {
    throw "Environment freeze failed with exit code $LASTEXITCODE"
}
$FreezeOutput |
    Set-Content -LiteralPath (Join-Path $EnvironmentRoot "requirements-lock.txt") -Encoding utf8
$NvidiaOutput = & nvidia-smi
if ($LASTEXITCODE -ne 0) {
    throw "nvidia-smi failed with exit code $LASTEXITCODE"
}
$NvidiaOutput |
    Set-Content -LiteralPath (Join-Path $EnvironmentRoot "nvidia-smi.txt") -Encoding utf8
$CheckOutput = & $VenvPython (Join-Path $PSScriptRoot "check_environment.py") --require-cuda
if ($LASTEXITCODE -ne 0) {
    throw "Environment validation failed with exit code $LASTEXITCODE"
}
$CheckOutput |
    Tee-Object -FilePath (Join-Path $EnvironmentRoot "environment.json")

Write-Output "Environment ready: $VenvPython"
