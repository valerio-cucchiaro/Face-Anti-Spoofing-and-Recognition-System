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
