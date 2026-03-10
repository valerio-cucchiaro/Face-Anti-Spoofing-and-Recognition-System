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
