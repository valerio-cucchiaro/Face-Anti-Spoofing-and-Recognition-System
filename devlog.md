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
