# Face Anti-Spoofing and Recognition System

This project implements a simple **face authentication system** that combines:

* **face detection**
* **anti-spoofing (fake face detection)**
* **face recognition**

The system detects a face from a webcam or image, determines whether the face is **real or spoofed**, and only if the face is genuine proceeds with **identity recognition**.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/valerio-cucchiaro/Face-Anti-Spoofing-and-Recognition-System.git
cd Face-Anti-Spoofing-and-Recognition-System
```

### 2. Create Python virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows (Command Prompt)**:

```bash
venv\Scripts\activate
```

**Linux / macOS**:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the project

Example:

```bash
python main.py
```

---

## Planned Structure

```text
project/
│
├── dataset/
├── detection/
├── preprocessing/
├── antispoofing/
├── feature_extraction/
├── recognition/
├── evaluation/
└── demo/
```

Each module corresponds to a stage of the biometric processing pipeline.

---

## Requirements

Main libraries used:

* OpenCV
* NumPy
* Scikit-learn
* Matplotlib

## Running the Project

The project uses a **src layout**, therefore modules must be executed using the `-m` flag so that Python correctly resolves imports.

All internal imports follow this format:

```text
from src.<module> import ...
```

For this reason, scripts must be executed **as modules**, not as standalone files.

### Full Pipeline

Runs the complete system (used for development and GitHub version).

```bash
python -m src.main
```

### Computer Vision Pipeline

Runs the pipeline prepared for the **Computer Vision project**.

```bash
python -m src.main_cv
```

### Biometric Systems Pipeline

Runs the pipeline prepared for the **Biometric Systems project**.

```bash
python -m src.main_bs
```

### Dataset Collection

Collects face images from the webcam and stores them in the dataset directory.

```bash
python -m src.dataset.dataset_collector
```

### Dataset Loading (for training / experiments)

```bash
python -m src.dataset.dataset_loader
```

### Important

Do **not** run files directly like this:

```bash
python src/main.py
python src/dataset/dataset_collector.py
```

because Python will not correctly resolve the `src` package.

Always run modules using:

```bash
python -m src.<module>
```
