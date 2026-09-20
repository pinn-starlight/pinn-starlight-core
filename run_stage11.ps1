param(
    [ValidateSet("a","b","cv5","e4","all")]
    [string]$Command = "all",

    [ValidateSet("auto","default","remote","remote-blackwell")]
    [string]$Environment = "auto",

    [string]$Gpus = "auto",
    [int]$MaxParallel = 0,

    [switch]$Force,
    [switch]$ForceData,
    [string]$StageADataModule = "",
    [string]$StageBDataModule = "",
    [switch]$PlanOnly,
    [switch]$SkipGpuProbe
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ProjectRoot

function Get-Stage11NvidiaInfo {
    $cmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        throw "nvidia-smi was not found. Stage11 formal runs require an NVIDIA driver/GPU."
    }

    $lines = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
    if ($LASTEXITCODE -ne 0 -or $null -eq $lines) {
        throw "nvidia-smi failed. Verify the NVIDIA driver before starting Stage11."
    }

    $rows = @()
    foreach ($line in $lines) {
        $parts = "$line".Split(",", 2)
        if ($parts.Count -ne 2) {
            throw "Unexpected nvidia-smi output: $line"
        }
        $rows += [PSCustomObject]@{
            Name = $parts[0].Trim()
            Driver = $parts[1].Trim()
        }
    }
    if ($rows.Count -eq 0) {
        throw "nvidia-smi reported zero GPUs."
    }
    return $rows
}

function Test-BlackwellGpuName {
    param([string]$Name)
    return ($Name -match '(?i)\bRTX\s*(?:PRO\s*)?50\d{2}\b|\bBlackwell\b|\bB100\b|\bB200\b|\bGB200\b|\bGB300\b')
}

function Resolve-Stage11Environment {
    param(
        [string]$Requested,
        [object[]]$GpuInfo
    )

    $HasBlackwell = $false
    foreach ($gpu in $GpuInfo) {
        if (Test-BlackwellGpuName $gpu.Name) {
            $HasBlackwell = $true
            break
        }
    }

    if ($Requested -ne "auto") {
        if ($HasBlackwell -and $Requested -ne "remote-blackwell") {
            throw "Blackwell/RTX-50 GPU detected. Use -Environment remote-blackwell (or auto)."
        }
        return $Requested
    }

    if ($HasBlackwell) {
        return "remote-blackwell"
    }

    if (Test-Path (Join-Path $ProjectRoot ".stage11_local")) {
        return "default"
    }

    # Preserve the known Windows + RTX 2080 Ti local profile without requiring
    # a marker file; other non-Blackwell hosts use the general remote profile.
    if ($env:OS -eq "Windows_NT") {
        $all2080 = $true
        foreach ($gpu in $GpuInfo) {
            if ($gpu.Name -notmatch '(?i)\b2080\b') {
                $all2080 = $false
                break
            }
        }
        if ($all2080) {
            return "default"
        }
    }

    return "remote"
}

function Assert-Stage11DriverCompatibility {
    param(
        [string]$SelectedEnvironment,
        [object[]]$GpuInfo
    )

    $majors = @()
    foreach ($gpu in $GpuInfo) {
        try {
            $majors += [int](($gpu.Driver -split '\.')[0])
        }
        catch {
            throw "Could not parse NVIDIA driver version '$($gpu.Driver)'."
        }
    }
    $minMajor = ($majors | Measure-Object -Minimum).Minimum

    if ($SelectedEnvironment -eq "remote-blackwell") {
        if ($minMajor -lt 580) {
            throw "remote-blackwell uses cu130 and requires an R580+ NVIDIA driver. Detected driver major: $minMajor."
        }
        return
    }

    if ($env:OS -eq "Windows_NT") {
        if ($minMajor -lt 528) {
            throw "cu126 profile requires an R528+ CUDA-12-family driver on Windows. Detected: $minMajor."
        }
    }
    elseif ($minMajor -lt 525) {
        throw "cu126 profile requires an R525+ CUDA-12-family driver on Linux. Detected: $minMajor."
    }
}

try {
    if (-not (Get-Command pixi -ErrorAction SilentlyContinue)) {
        throw "Pixi was not found in PATH. Install Pixi first, then reopen the terminal."
    }
    if (-not (Test-Path (Join-Path $ProjectRoot "pyproject.toml"))) {
        throw "pyproject.toml was not found in $ProjectRoot"
    }
    if (-not (Test-Path (Join-Path $ProjectRoot "pixi.lock"))) {
        throw "pixi.lock is missing."
    }
    if (-not $env:CUDA_DEVICE_ORDER) {
        $env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
    }

    $GpuInfo = @(Get-Stage11NvidiaInfo)
    $SelectedEnvironment = Resolve-Stage11Environment $Environment $GpuInfo
    Assert-Stage11DriverCompatibility $SelectedEnvironment $GpuInfo

    Write-Host ""
    Write-Host "=== Stage11 launcher ===" -ForegroundColor Cyan
    Write-Host "Host platform : Windows"
    foreach ($gpu in $GpuInfo) {
        Write-Host ("GPU           : " + $gpu.Name + " | driver " + $gpu.Driver)
    }
    Write-Host "Pixi env      : $SelectedEnvironment"
    if ($SelectedEnvironment -eq "remote-blackwell") {
        Write-Host "CUDA profile  : torch 2.14.0 + cu130"
    }
    else {
        Write-Host "CUDA profile  : torch 2.14.0 + cu126"
    }
    Write-Host "GPU scheduler : $Gpus (CV5 auto task parallelism)"
    Write-Host "Data policy   : missing A/B data are generated; existing formal data are reused"
    Write-Host "                -Force retrains only; -ForceData explicitly regenerates datasets"
    Write-Host ""

    $argsList = @(
        "run",
        "-e", $SelectedEnvironment,
        "--locked",
        "python", "-u", "run_stage11.py",
        $Command,
        "--gpus", $Gpus
    )

    if ($MaxParallel -gt 0) { $argsList += @("--max-parallel", "$MaxParallel") }
    if ($Force) { $argsList += "--force" }
    if ($ForceData) { $argsList += "--force-data" }
    if ($StageADataModule) { $argsList += @("--stage-a-data-module", $StageADataModule) }
    if ($StageBDataModule) { $argsList += @("--stage-b-data-module", $StageBDataModule) }
    if ($PlanOnly) { $argsList += "--plan-only" }
    if ($SkipGpuProbe) { $argsList += "--skip-gpu-probe" }

    Write-Host ("Running       : pixi " + ($argsList -join " ")) -ForegroundColor DarkGray
    Write-Host ""

    & pixi @argsList
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
