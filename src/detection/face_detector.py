"""
Module: face_detector

Description:
This module provides utilities for loading and using the Haar Cascade
face detector from OpenCV.

It encapsulates the initialization of the classifier and exposes a
function for detecting faces in grayscale frames. The detected face
regions are returned as bounding boxes used by later stages of the
pipeline.
"""

import cv2  
import mediapipe as mp


def load_mediapipe_detector():
    """Load and initialize the MediaPipe face detector."""

    # Use the short-range model, which is more suitable for webcam usage.
    mp_face_detection = mp.solutions.face_detection
    detector = mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )
    return detector


def detect_faces_mediapipe(detector, frame):
    """Detect faces in a BGR frame using the MediaPipe detector."""

    # MediaPipe expects RGB input, so convert the OpenCV BGR frame first.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb_frame)

    boxes = []

    if results.detections:
        h, w, _ = frame.shape

        for detection in results.detections:
            # Convert the relative MediaPipe bounding box into pixel coordinates.
            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            boxes.append((x, y, bw, bh))

    return boxes


def load_face_detector(model_path: str):
    """Load and initialize the Haar Cascade face detector."""

    # Load the Haar Cascade classifier from disk.
    detector = cv2.CascadeClassifier(model_path)

    # Stop early if OpenCV fails to initialize the detector correctly.
    if detector.empty():
        raise RuntimeError("Failed to load Haar Cascade classifier")
    
    return detector


def detect_faces(detector, frame, method="mediapipe", **kwargs):
    """Detect faces using the selected backend."""

    if method == "mediapipe":
        return detect_faces_mediapipe(detector, frame)

    elif method == "haar":
        # Haar Cascade detection is applied on the grayscale frame.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return detector.detectMultiScale(
            gray,
            scaleFactor=kwargs.get("scaleFactor", 1.1),
            minNeighbors=kwargs.get("minNeighbors", 5),
            minSize=kwargs.get("minSize", (30, 30))
        )

    else:
        raise ValueError("Unknown detection method")
