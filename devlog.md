# Development Log

## 2026-03-08

**Project Initialization**:

- created GitHub repository
- defined project idea and architecture
- connected local project to GitHub repository
- created general `.gitignore`

## 2026-03-09

**Environment Setup**:

- simplified and reorganized project ```README```
- created Python virtual environment (```venv```)
- installed initial project dependencies
- generated ```requirements.txt```
- created development log (```devlog.md```)
- downloaded `haarcascade_frontalface_default.xml` face detection model to `models/`

**Face Detection Pipeline (main.py)**:
Implemented the first functional pipeline for real-time face detection using OpenCV.
The script performs the following steps:

1. Access the system webcam using OpenCV VideoCapture.
2. Convert each captured frame to grayscale for faster and more reliable face detection.
3. Use a Haar Cascade classifier (haarcascade_frontalface_default.xml) to detect faces.
4. Draw bounding boxes around detected faces in the original frame.
5. Extract the detected face region (ROI) from the grayscale frame.
6. Normalize the face by resizing it to a fixed resolution (128x128).

This establishes the base pipeline required for the later stages of the project:
webcam → face detection → face crop → face normalization

**Code Modularization**:

- refactored the initial `main.py` script into separate modules to improve code structure
- separated face detection logic into `detection/face_detector.py`
- separated image preprocessing functions into `preprocessing/face_preprocessing.py`
- kept `main.py` responsible only for pipeline orchestration and webcam handling
- added `__init__.py` files to define Python packages

Project structure after refactoring:

```text
project/
│
├── main.py
├── models/
│   └── haarcascade_frontalface_default.xml
├── detection/
│   ├── __init__.py
│   └── face_detector.py
└── preprocessing/
    ├── __init__.py
    └── face_preprocessing.py
```

## 2026-03-10

**Face Detection Improvements**:

Refined the face detection stage and improved runtime behavior of the webcam pipeline.

- implemented detection handling when no faces are found
- added on-screen feedback message ("No face detected")
- modified detection logic to select only the largest detected face when multiple faces are present
- added distance feedback to the user based on detected face size ("Move closer to the camera")
- introduced configurable detection parameters (`scaleFactor`, `minNeighbors`, `minSize`)
- exposed detection parameters as constants in `main.py` to simplify tuning

These changes improve detection stability and provide clearer feedback during webcam interaction.

**Face Preprocessing Improvements**:

Extended the preprocessing pipeline with illumination normalization.

- implemented histogram equalization (`cv2.equalizeHist`) on the detected face region
- added new preprocessing function `face_histogram_equalization()` to `preprocessing.py`

Updated preprocessing pipeline:

webcam frame → grayscale → face detection → face crop → resize → histogram equalization

This normalization step improves robustness for future face recognition and anti-spoofing stages by reducing sensitivity to lighting variations.

**Stage 1 completed**:
Implemented and stabilized the real-time webcam face detection and preprocessing pipeline, including largest-face selection, distance feedback, crop, resize, and histogram equalization. The codebase was modularized into main orchestration, detection, and preprocessing modules. The project is now ready to move to Stage 2: dataset collection.

**Stage 2: Dataset Collection Implementation**:

- created dataset module `src/dataset/dataset_collector.py`
- created dataset directory structure `data/dataset/<person>`
- implemented automatic dataset directory creation if missing
- implemented dataset indexing to continue numbering existing images
- integrated face detection and preprocessing pipeline inside dataset collector
- implemented face validation using minimum face area threshold
- implemented keyboard-controlled dataset capture (`s` key)
- implemented sequential image saving (`001.png`, `002.png`, ...)
- implemented maximum sample limit per person
- collected first dataset samples for initial testing (30 images)

The dataset collector reuses the same detection and preprocessing pipeline used in the main application to ensure consistency between training data and runtime recognition.

**Dataset Loading Implementation**:

Implemented the dataset loading module required for training the face recognition model.

- created dataset loader module `src/dataset/dataset_loader.py`
- implemented automatic scanning of all person directories inside `data/dataset/`
- implemented dynamic assignment of numeric labels to each person
- implemented label mapping (`label → person_name`)
- implemented loading of dataset images in grayscale using OpenCV
- implemented construction of `faces` and `labels` arrays required for training
- added basic validation to skip unreadable images

The dataset loader converts the dataset folder structure into the format required by OpenCV face recognizers (list of face images and corresponding numeric labels).

## 2026-03-11 — Project Refactor: Modular Pipeline Architecture

### Summary

Major refactor of the project architecture to properly separate modules and support three different pipelines:

- Full system (GitHub project)
- Computer Vision project
- Biometric Systems project

The goal of this refactor was to cleanly separate **detection**, **preprocessing**, and **pipeline logic**, making the system modular and easier to maintain and extend.

---

### Implemented Changes

#### 1. Modular Pipeline Architecture

Created a new `pipeline` module to centralize orchestration logic.

```text
src/pipeline/
```

Two core files were introduced:

```text
frame_processing.py
run_pipeline_full.py
```

- `frame_processing.py` contains logic for validating detected faces.
- `run_pipeline_full.py` contains the full runtime pipeline of the system.

---

#### 2. Face Validation Module

Implemented the function:

```text
get_valid_face()
```

This function performs the following steps:

1. Convert frame to grayscale
2. Detect faces
3. Select the largest detected face
4. Verify minimum face area threshold

This module acts as the bridge between:

```text
detection → preprocessing
```

and keeps validation logic out of the main pipeline.

---

#### 3. Preprocessing Pipeline Function

Added a new function in:

```text
src/preprocessing/face_preprocessing.py
```

```text
preprocess_face()
```

This function performs the complete preprocessing pipeline:

```text
crop → resize → histogram equalization
```

This centralizes preprocessing operations and avoids duplicated logic.

---

#### 4. Full Pipeline Implementation

Created:

```text
src/pipeline/run_pipeline_full.py
```

Implemented the function:

```text
run_pipeline_full()
```

This pipeline currently performs:

```text
webcam acquisition
↓
face detection
↓
face validation
↓
face preprocessing
↓
display
```

This represents the **current operational system**.

---

#### 5. Entry Point Refactor

Refactored the main scripts to act only as entry points.

```text
src/main.py
src/main_cv.py
src/main_bs.py
```

Each file now simply launches its corresponding pipeline:

```text
main
```

## IN FASE DI IMPLEMENTAZIONE

1. Train face recognition model (LBPH)

2. Save trained model to disk

3. Implement face recognition in main pipeline

4. Integrate recognition with webcam detection

5. Display predicted identity in real time

6. Prepare anti-spoofing module structure

7. Implement liveness detection approach

8. Evaluate recognition and anti-spoofing performance
