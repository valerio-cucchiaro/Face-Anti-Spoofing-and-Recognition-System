# Development Log

## 2026-03-08

**Project Initialization**:

- created GitHub repository
- defined project idea and architecture
- connected local project to GitHub repository
- created general `.gitignore`

---

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

---

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

---

## 2026-03-11 — Project Refactor: Modular Pipeline Architecture

**Pipeline Architecture Refactor**:

- created new module `src/pipeline/` to centralize system orchestration
- created `frame_processing.py` for frame-level validation logic
- created `run_pipeline_full.py` implementing the full runtime pipeline
- implemented function `get_valid_face()` to validate detected faces
- added face validation steps: grayscale conversion, face detection, largest face selection, minimum face area check
- implemented `preprocess_face()` function in `src/preprocessing/face_preprocessing.py`
- centralized preprocessing pipeline: crop → resize → histogram equalization
- refactored the runtime pipeline to use modular detection, validation, and preprocessing stages
- created clean entry points:

  - `src/main.py`
  - `src/main_cv.py`
  - `src/main_bs.py`
- separated orchestration logic from processing modules

Runtime pipeline after refactor:

webcam acquisition
↓
face detection
↓
face validation
↓
face preprocessing
↓
display

This refactor establishes a clean modular architecture that will support the future integration of face recognition and anti‑spoofing modules.

---

## 2026-03-12

**Face Recognition Training Implementation**:

- created recognition module `src/recognition/`
- implemented training script `src/recognition/train_model.py`
- integrated dataset loader with recognition training pipeline
- initialized OpenCV LBPH face recognizer
- trained recognition model using dataset images
- implemented automatic creation of `models/` directory if missing
- saved trained model to `models/lbph_model.xml`
- saved label mapping to `models/label_map.pkl`
- updated project dependencies to use `opencv-contrib-python` (required for LBPH recognizer)

**Offline Recognition Testing**:

- implemented offline evaluation script `src/recognition/test_model.py`
- added functionality to load trained LBPH model from disk
- implemented loading of label mapping file
- implemented prediction of identity using test dataset images
- implemented confidence score reporting for each prediction
- implemented automatic accuracy calculation on test dataset
- validated recognition performance using reserved test images

**Recognition Evaluation Preparation**:

- collected confidence values for genuine matches
- prepared infrastructure for computing genuine and impostor score distributions
- prepared threshold-based recognition logic for handling unknown identities

Recognition decision rule prepared for runtime pipeline:

if confidence < threshold
→ recognized identity
else
→ Unknown

This stage completes the offline face recognition workflow and prepares the system for integration of real‑time recognition into the webcam pipeline.

**Real-Time Face Recognition Integration**:

- created recognition runtime module `src/recognition/face_recognizer.py`
- implemented function `load_recognizer()` to load the trained LBPH model and label map
- implemented function `predict_identity()` to perform identity prediction
- implemented threshold-based classification to handle unknown identities
- integrated recognition stage into the full webcam pipeline (`run_pipeline_full.py`)
- implemented real-time identity prediction on detected faces
- added visualization of predicted identity on the webcam stream
- added visualization of LBPH confidence score
- implemented runtime classification rule:

confidence < threshold → recognized identity  
confidence ≥ threshold → Unknown

- validated recognition behavior using partial occlusions and distance variations
- confirmed that high confidence values correctly trigger "Unknown" classification
- added module and function docstring

---

## IN FASE DI IMPLEMENTAZIONE

1. Improve recognition robustness with multiple identities in the dataset

2. Collect dataset samples for additional users

3. Evaluate recognition performance and tune UNKNOWN_THRESHOLD

4. Design anti-spoofing module architecture

5. Implement liveness detection approach

6. Integrate anti-spoofing into the runtime pipeline

7. Combine recognition and anti-spoofing decisions

8. Evaluate recognition and anti-spoofing performance
