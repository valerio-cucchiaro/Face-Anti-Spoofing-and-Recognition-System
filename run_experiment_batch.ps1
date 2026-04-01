param(
    [switch]$ShutdownWhenDone
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $repoRoot "venv\Scripts\python.exe"
$logDir = Join-Path $repoRoot "logs"
$recognitionLog = Join-Path $logDir "recognition_runs.txt"
$padLog = Join-Path $logDir "pad_runs.txt"
$shutdownDelaySeconds = 60 *3 # 3 minutes

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found at: $pythonExe"
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Reset logs once at batch startup so each batch execution starts clean,
# while preserving append behavior across multiple runs inside the same batch.
"" | Out-File -FilePath $recognitionLog -Encoding utf8
"" | Out-File -FilePath $padLog -Encoding utf8

function Write-LogLines {
    param(
        [string]$LogPath,
        [string[]]$Lines
    )

    foreach ($line in $Lines) {
        $line | Out-File -FilePath $LogPath -Append -Encoding utf8
    }
}

function Add-LogHeader {
    param(
        [string]$LogPath,
        [string]$Title
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-LogLines -LogPath $LogPath -Lines @(
        "",
        "============================================================",
        "$Title | $timestamp",
        "============================================================"
    )
}

function Invoke-LoggedPythonRun {
    param(
        [string]$LogPath,
        [string]$Label,
        [string[]]$Arguments
    )

    $startTimestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-LogLines -LogPath $LogPath -Lines @(
        "",
        ">>> $Label",
        ">>> STATUS: STARTED at $startTimestamp",
        ">>> COMMAND: $pythonExe $($Arguments -join ' ')",
        ""
    )

    & $pythonExe @Arguments 2>&1 | ForEach-Object {
        $line = $_.ToString()
        Write-Output $line
        $line | Out-File -FilePath $LogPath -Append -Encoding utf8
    }

    $endTimestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-LogLines -LogPath $LogPath -Lines @(
        "",
        ">>> STATUS: FINISHED at $endTimestamp with exit code $LASTEXITCODE"
    )

    if ($LASTEXITCODE -ne 0) {
        throw "Command failed for run: $Label"
    }
}

Add-LogHeader -LogPath $recognitionLog -Title "RECOGNITION BATCH"
Add-LogHeader -LogPath $padLog -Title "PAD BATCH"

$recognitionRuns = @()

foreach ($run in $recognitionRuns) {
    Invoke-LoggedPythonRun -LogPath $recognitionLog -Label $run.Label -Arguments $run.Args
}

$padRuns = @(
    @{
        Label = "pad | full | baseline+ggd"
        Args = @(
            "-m", "src.antispoofing.training.pad_experiment_runner",
            "--experiment-speed", "full",
            "--pad-debug-feature-mode", "baseline",
            "--pad-lft-mode", "ggd_light"
        )
    },
    @{
        Label = "pad | full | baseline+tgsd"
        Args = @(
            "-m", "src.antispoofing.training.pad_experiment_runner",
            "--experiment-speed", "full",
            "--pad-debug-feature-mode", "tgsd",
            "--pad-lft-mode", "summary"
        )
    },
    @{
        Label = "pad | full | baseline+tgsd+ggd"
        Args = @(
            "-m", "src.antispoofing.training.pad_experiment_runner",
            "--experiment-speed", "full",
            "--pad-debug-feature-mode", "tgsd",
            "--pad-lft-mode", "ggd_light"
        )
    }
)

foreach ($run in $padRuns) {
    Invoke-LoggedPythonRun -LogPath $padLog -Label $run.Label -Arguments $run.Args
}

if ($ShutdownWhenDone) {
    $shutdownDelayMinutes = [int]($shutdownDelaySeconds / 60)
    $shutdownMessage = "Batch completed. Windows will close running apps and shut down the PC in $shutdownDelayMinutes minutes."

    Write-Output $shutdownMessage
    Write-LogLines -LogPath $recognitionLog -Lines @("", ">>> SHUTDOWN: $shutdownMessage")
    Write-LogLines -LogPath $padLog -Lines @("", ">>> SHUTDOWN: $shutdownMessage")

    shutdown /s /f /t $shutdownDelaySeconds /c $shutdownMessage
}

# powershell -ExecutionPolicy Bypass -File .\run_experiment_batch.ps1
# powershell -ExecutionPolicy Bypass -File .\run_experiment_batch.ps1 -ShutdownWhenDone
