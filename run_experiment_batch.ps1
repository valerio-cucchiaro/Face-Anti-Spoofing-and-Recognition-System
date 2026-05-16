param(
    [switch]$ShutdownWhenDone,
    [switch]$EnableReliabilityDiagnostics
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $repoRoot "venv\Scripts\python.exe"
$logDir = Join-Path $repoRoot "logs"
$recognitionLog = Join-Path $logDir "recognition_runs.txt"
$padLog = Join-Path $logDir "pad_runs.txt"
$shutdownDelaySeconds = 60 *3 # 3 minutes
$maxRunAttempts = 2
$retryDelaySeconds = 10

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

    for ($attempt = 1; $attempt -le $maxRunAttempts; $attempt++) {
        $startTimestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Write-LogLines -LogPath $LogPath -Lines @(
            "",
            ">>> $Label",
            ">>> STATUS: STARTED at $startTimestamp (attempt $attempt/$maxRunAttempts)",
            ">>> COMMAND: $pythonExe $($Arguments -join ' ')",
            ""
        )

        $previousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"

        & $pythonExe @Arguments 2>&1 | ForEach-Object {
            $line = $_.ToString()
            Write-Output $line
            $line | Out-File -FilePath $LogPath -Append -Encoding utf8
        }

        $commandExitCode = $LASTEXITCODE
        $ErrorActionPreference = $previousErrorActionPreference

        $endTimestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Write-LogLines -LogPath $LogPath -Lines @(
            "",
            ">>> STATUS: FINISHED at $endTimestamp with exit code $commandExitCode (attempt $attempt/$maxRunAttempts)"
        )

        if ($commandExitCode -eq 0) {
            return
        }

        if ($attempt -lt $maxRunAttempts) {
            $retryMessage = ">>> RETRY: waiting $retryDelaySeconds seconds before retrying $Label"
            Write-Output $retryMessage
            Write-LogLines -LogPath $LogPath -Lines @("", $retryMessage)
            Start-Sleep -Seconds $retryDelaySeconds
        }
    }

    throw "Command failed for run after $maxRunAttempts attempts: $Label"
}

Add-LogHeader -LogPath $recognitionLog -Title "RECOGNITION BATCH"

$recognitionDiagnosticArgs = @()
if ($EnableReliabilityDiagnostics) {
    $recognitionDiagnosticArgs += "--enable-reliability-diagnostics"
}

$recognitionRuns = @(
    @{
        Label = "recognition | custom -> custom"
        Args = @(
            "-m", "src.recognition.recognition_experiment_runner",
            "--train-mode", "custom",
            "--eval-mode", "custom",
            "--radius", "1",
            "--quality-filter-mode", "report_only",
            "--save-system-threshold"
        )
    }
    #@{
    #    Label = "recognition | orl -> orl"
    #    Args = @(
    #        "-m", "src.recognition.recognition_experiment_runner",
    #        "--train-mode", "orl",
    #        "--eval-mode", "orl",
    #        "--radius", "1",
    #        "--quality-filter-mode", "report_only"
    #    )
    #}
    #@{
    #    Label = "recognition | multipie -> multipie"
    #    Args = @(
    #     "-m", "src.recognition.recognition_experiment_runner",
    #        "--train-mode", "multipie",
    #        "--eval-mode", "multipie",
    #       "--radius", "1",
    #        "--quality-filter-mode", "report_only"
    #    )
    #}
)

foreach ($run in $recognitionRuns) {
    Invoke-LoggedPythonRun `
        -LogPath $recognitionLog `
        -Label $run.Label `
        -Arguments ($run.Args + $recognitionDiagnosticArgs)
}

Add-LogHeader -LogPath $padLog -Title "PAD BATCH"

$padRuns = @(
    @{
        Label = "pad | combined_custom -> all"
        Args = @(
            "-m", "src.antispoofing.training.pad_experiment_runner",
            "--train-mode", "combined_custom",
            "--eval-mode", "all"
        )
    }
    #@{
    #     Label = "pad | custom -> custom"
    #     Args = @(
    #         "-m", "src.antispoofing.training.pad_experiment_runner",
    #         "--train-mode", "custom",
    #         "--eval-mode", "custom"
    #     )
    #}
    # @{
    #     Label = "pad | casia -> casia"
    #     Args = @(
    #         "-m", "src.antispoofing.training.pad_experiment_runner",
    #         "--train-mode", "casia",
    #         "--eval-mode", "casia"
    #     )
    # }
    # @{
    #     Label = "pad | replay -> replay"
    #     Args = @(
    #         "-m", "src.antispoofing.training.pad_experiment_runner",
    #         "--train-mode", "replay",
    #         "--eval-mode", "replay"
    #     )
    # }
)

foreach ($run in $padRuns) {
    Invoke-LoggedPythonRun -LogPath $padLog -Label $run.Label -Arguments $run.Args
}

if ($ShutdownWhenDone) {
    $shutdownDelayMinutes = [int]($shutdownDelaySeconds / 60)
    $shutdownMessage = "Batch completed. Windows will close running apps and shut down the PC in $shutdownDelayMinutes minutes."

    Write-Output $shutdownMessage
    Write-LogLines -LogPath $recognitionLog -Lines @("", ">>> SHUTDOWN: $shutdownMessage")

    shutdown /s /f /t $shutdownDelaySeconds /c $shutdownMessage
}

# powershell -ExecutionPolicy Bypass -File .\run_experiment_batch.ps1
# powershell -ExecutionPolicy Bypass -File .\run_experiment_batch.ps1 -EnableReliabilityDiagnostics
# powershell -ExecutionPolicy Bypass -File .\run_experiment_batch.ps1 -ShutdownWhenDone
