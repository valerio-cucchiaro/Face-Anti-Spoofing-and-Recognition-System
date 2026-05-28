# Face Anti-Spoofing and Recognition System

## Project Overview

This project implements a complete face-analysis pipeline that combines:

- face detection
- face anti-spoofing / presentation attack detection (PAD)
- face recognition
- integrated runtime decision logic
- offline training and evaluation workflows

The system supports both real-time webcam execution and offline experiments.
At runtime, a face is detected and validated first, PAD is used to decide
whether the face is likely live or spoofed, and recognition is applied only
when the current pipeline variant allows it.

The repository also includes dataset preparation utilities, experiment
runners, evaluation scripts, saved-model handling, and an automated test
suite for the main contracts used by the project.

## Main Features

- Full webcam runtime that combines detection, PAD, and recognition
- PAD-only webcam runtime for liveness-focused checks
- Recognition-only webcam runtime for identity-focused checks
- Offline PAD training and evaluation with custom and public datasets
- Offline recognition training and evaluation with custom and public datasets
- System-level evaluation with PAD-gated recognition decisions
- Dataset preparation and collection utilities
- Automated tests for loaders, evaluation contracts, runtime helpers, and
  export behavior

## Repository Structure

```text
.
|-- data/                      # datasets and collected samples
|-- logs/                      # exported metrics and run logs
|-- models/                    # saved PAD and recognition artifacts
|-- src/
|   |-- antispoofing/          # PAD runtime features, models, utilities
|   |-- dataset/               # dataset splitters, collectors, extractors
|   |-- detection/             # face detector and landmark helpers
|   |-- evaluation/            # offline PAD / recognition / system evaluation
|   |-- pipeline/              # webcam runtime pipelines and shared runtime state
|   |-- preprocessing/         # face crop, validation, pose, and preprocessing helpers
|   |-- recognition/           # recognition runtime, loaders, training, evaluation helpers
|   |-- main.py                # full runtime entry point
|   |-- main_pad.py            # PAD-only runtime entry point
|   `-- main_recognition.py    # recognition-only runtime entry point
|-- tests/                     # automated test suite
|-- devlog.md                  # development history
|-- pytest.ini
|-- requirements.txt
`-- run_experiment_batch.ps1   # batch runner for offline experiments
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/valerio-cucchiaro/Face-Anti-Spoofing-and-Recognition-System.git
cd Face-Anti-Spoofing-and-Recognition-System
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
venv\Scripts\activate.bat
```

Linux / macOS:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## How To Run

The project uses a `src` layout. Run modules with `python -m ...` so Python
resolves imports correctly.

### Full Runtime

Runs the integrated webcam pipeline with detection, PAD, and recognition.

```bash
python -m src.main
```

The full runtime also supports a logging-level switch:

```bash
python -m src.main --log-level verbose
```

If no flag is passed, the full runtime uses `normal` logging by default.

### PAD-Only Runtime

Runs the webcam pipeline dedicated to anti-spoofing / liveness checks.

```bash
python -m src.main_pad
```

The PAD runtime also supports a logging-level switch:

```bash
python -m src.main_pad --log-level verbose
```

If no flag is passed, the PAD runtime uses `normal` logging by default.

### Recognition-Only Runtime

Runs the webcam pipeline dedicated to face recognition without the PAD gate.

```bash
python -m src.main_recognition
```

The recognition runtime also supports a logging-level switch:

```bash
python -m src.main_recognition --log-level verbose
```

If no flag is passed, the recognition runtime uses `normal` logging by default.

### Command-Line Help

Some entrypoints expose a small command-line helper via `--help`.

Runtime entrypoints:

```bash
python -m src.main --help
python -m src.main_pad --help
python -m src.main_recognition --help
```

Offline scripts:

```bash
python -m src.recognition.recognition_experiment_runner --help
python -m src.antispoofing.training.pad_experiment_runner --help
python -m src.evaluation.system_evaluator --help
```

Use these helpers to inspect the supported flags before launching a run.

## Offline Experiments

### Recognition

Example custom recognition train + evaluation run:

```bash
python -m src.recognition.recognition_experiment_runner --train-mode custom --eval-mode custom --save-system-threshold
```

### PAD

Example custom PAD train + evaluation run:

```bash
python -m src.antispoofing.training.pad_experiment_runner --train-mode custom --eval-mode custom --experiment-speed full
```

The repository also includes a batch runner:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_experiment_batch.ps1
```

### System Evaluation

Runs the integrated offline evaluation using saved PAD and recognition
artifacts and thresholds.

```bash
python -m src.evaluation.system_evaluator
```

## Datasets

The project supports both custom and public datasets.

### Recognition Datasets

- `custom`
- `orl`
- `lfw`
- `multipie`

Recognition data may be used for:

- gallery / enrollment loading
- validation split loading
- probe split loading
- custom internal open-set validation / probe derivation

For the custom recognition dataset, some evaluation splits such as
`validation_known_internal` and `probe_unknown_internal` are logical internal
subsets derived from the on-disk probe folders rather than separate physical
directories.

### PAD Datasets

- `custom`
- `casia`
- `replay`
- combined / all-dataset experiment modes

The repository also contains utilities to:

- collect custom recognition images
- collect custom PAD runtime crops
- prepare ORL / LFW / Multi-PIE recognition splits
- prepare CASIA PAD splits
- extract Replay-Attack PAD frames

## Models And Outputs

### Recognition Artifacts

Typical recognition artifacts are stored under:

- `models/recognition/lbph_model.xml`
- `models/recognition/label_map.pkl`
- `models/recognition/recognizer_config.json`
- `models/recognition/system_recognition_threshold_config.json`

### PAD Artifacts

Typical PAD artifacts are stored under:

- `models/antispoofing/`

This includes the trained PAD classifier, scaler, config, and runtime/offline
artifact variants used by the project.

### Metrics And Reports

Offline runs export metrics and reports under:

- `logs/metrics/recognition/`
- `logs/metrics/pad/`
- `logs/metrics/system/`

These outputs include JSON reports, CSV summaries, prediction exports, and
threshold-related metadata.

## Testing

Run the full automated test suite with:

```bash
python -m pytest
```

The test suite covers:

- dataset loader and split contracts
- PAD and recognition helper logic
- runtime state helpers
- export path and threshold-config behavior
- dataset utility behavior

## Notes

- Always run modules with `python -m ...` rather than executing files
  directly.
- The repository contains both runtime code and offline experiment code; not
  every module is meant to be used as a user-facing entry point.
- `devlog.md` is the historical development log for the project and is useful
  when you need implementation history or refactor context.
