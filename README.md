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

## Required Runtime Artifacts

The webcam runtimes are not self-contained from source code alone. A
working runtime setup requires the final trained artifacts under
`models/`.

Required PAD runtime artifacts:

- `models/pad/pad_svm.joblib`
- `models/pad/pad_scaler.joblib`
- `models/pad/pad_config.json`

Required recognition runtime artifacts:

- `models/recognition/lbph_model.xml`
- `models/recognition/label_map.pkl`
- `models/recognition/recognizer_config.json`

If these files are missing, `python -m src.main_pad`,
`python -m src.main_recognition`, and `python -m src.main` are not
guaranteed to start correctly on a fresh clone.

## Required Offline Evaluation Artifacts

The integrated offline evaluator also requires the saved system-level
threshold references:

- `models/pad/system_pad_threshold_config.json`
- `models/recognition/system_recognition_threshold_config.json`

These files are required by `python -m src.evaluation.system_evaluator`
and related offline evaluation flows. They are not required by the
three webcam runtimes.

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

Canonical PAD train + evaluation run used to refresh the final runtime
artifact set and save the offline system threshold:

```bash
python -m src.antispoofing.training.pad_experiment_runner --train-mode all --eval-mode all --experiment-speed full --save-system-threshold
```

This is the only untagged PAD run that promotes the webcam runtime PAD
artifacts. Other PAD runs remain experiment-specific.

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

### Expected Dataset Layout

The repository expects datasets under `data/`. The exact collection
contents depend on the experiment being executed, but the structure is
split by task.

Recognition datasets are expected under:

- `data/recognition/custom_raw/`
- `data/recognition/orl/`
- `data/recognition/lfw/`
- `data/recognition/multipie/`

PAD datasets are expected under:

- `data/pad/pad_custom_raw/`
- `data/pad/casia_pad/`
- `data/pad/replay_attack_pad/`

These datasets are needed for offline training and evaluation. They are
not required to run the final webcam runtimes if the final artifacts are
already present under `models/`.

## Threshold Types

The project uses different threshold concepts in different contexts.

- Webcam threshold: threshold used by the live runtime during webcam
  inference.
- Benchmark threshold: threshold selected during an offline evaluation
  protocol on validation data and then applied to benchmark splits.
- System threshold: final offline threshold saved to
  `models/pad/system_pad_threshold_config.json` or
  `models/recognition/system_recognition_threshold_config.json` for the
  integrated system configuration.

These threshold roles should not be treated as interchangeable. Offline
benchmark calibration and final runtime behavior are related, but they
are not the same step in the pipeline.

## Final Delivered Configuration

The project delivery is based on one declared final configuration.

### Recognition

- Model: `LBPH`
- LBPH parameters:
  - `radius=1`
  - `neighbors=8`
  - `grid_x=6`
  - `grid_y=6`
- Preprocessing:
  - `normalization_mode=equalize`
  - `crop_padding_ratio=0.1`
- Webcam runtime threshold:
  - `40.0`
- Canonical runtime artifacts:
  - `models/recognition/lbph_model.xml`
  - `models/recognition/label_map.pkl`
  - `models/recognition/recognizer_config.json`
- Offline evaluation artifact:
  - `models/recognition/system_recognition_threshold_config.json`

### PAD

- Feature stack: `tgsd + ggd_light`
- TGSD configuration:
  - `tau=170`
  - `top_percent=0.05`
  - `gradient_operator=sobel`
- Canonical protocol:
  - `train-mode=all`
  - `eval-mode=all`
  - `experiment-speed=full`
- Final sampling setup for the canonical `all -> all | full` protocol:
  - `CASIA sample ratio=0.25`
  - `Replay sample ratio=1.0`
  - `custom_repeat=20`
- Validation operating point:
  - `target APCER=2%`
- Canonical runtime artifacts:
  - `models/pad/pad_svm.joblib`
  - `models/pad/pad_scaler.joblib`
  - `models/pad/pad_config.json`
- Offline evaluation artifact:
  - `models/pad/system_pad_threshold_config.json`

## Models And Outputs

### Recognition Artifacts

Typical recognition artifacts are stored under:

- `models/recognition/lbph_model.xml`
- `models/recognition/label_map.pkl`
- `models/recognition/recognizer_config.json`
- `models/recognition/system_recognition_threshold_config.json`

### PAD Artifacts

Typical PAD artifacts are stored under:

- `models/pad/pad_svm.joblib`
- `models/pad/pad_scaler.joblib`
- `models/pad/pad_config.json`
- `models/pad/system_pad_threshold_config.json`

Additional experiment-only PAD artifact snapshots may also exist under
`models/pad/`, but the files above are the canonical runtime artifacts.

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

- The final webcam runtime depends on pre-generated artifacts under
  `models/`; a source-only clone is not sufficient unless those
  artifacts are included or recreated.
- PAD performance is sensitive to capture distance, lighting, and domain
  mismatch between collected custom data and public datasets.
- Public and custom datasets used for offline experiments are not
  included in the repository by default.

- Always run modules with `python -m ...` rather than executing files
  directly.
- The repository contains both runtime code and offline experiment code; not
  every module is meant to be used as a user-facing entry point.
