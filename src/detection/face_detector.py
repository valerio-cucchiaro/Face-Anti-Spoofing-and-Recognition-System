"""
Module: face_detector

Description:
This module provides the MediaPipe face detection utilities used
throughout the project.

It exposes a detector factory and a detection helper that return face
bounding boxes in pixel coordinates. Those boxes are reused by the
runtime pipeline, dataset collectors, and PAD dataset loading tools.
"""

import cv2
import mediapipe as mp


def load_mediapipe_detector():
    """
    Load and initialize the MediaPipe face detector used by the
    project runtime and dataset utilities.
    """

    # Use the short-range model, which is more suitable for webcam usage.
    mp_face_detection = mp.solutions.face_detection
    detector = mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )
    return detector


def detect_faces_mediapipe(detector, frame):
    """
    Detect faces in a BGR frame using MediaPipe and return bounding
    boxes in ``(x, y, w, h)`` format.
    """

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


def detect_faces(detector, frame):
    """
    Detect faces using the shared MediaPipe backend configured for the
    project.
    """

    return detect_faces_mediapipe(detector, frame)
