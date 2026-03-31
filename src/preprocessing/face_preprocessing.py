"""
Module: face_preprocessing

Description:
This module implements preprocessing operations applied to detected
faces before they are used for recognition or other analysis tasks.

The preprocessing pipeline standardizes the face images by converting
them to grayscale, extracting the face region, resizing it to a fixed
resolution, and applying histogram equalization to normalize lighting
conditions.
"""

import cv2  

def to_grayscale(frame):
    """Convert a BGR frame to grayscale."""

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def crop_face(gray_frame, x, y, w, h):
    """
    Extract the face region from the grayscale frame.

    The bounding box is clamped to the image limits so that partially
    out-of-frame detections do not produce invalid slices.
    """

    frame_height, frame_width = gray_frame.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_width, x + w)
    y2 = min(frame_height, y + h)

    if x1 >= x2 or y1 >= y2:
        return None

    return gray_frame[y1:y2, x1:x2]

def resize_face(face, size=(128, 128)):
    """
    Resize a face image to a fixed resolution.

    Returns ``None`` when the provided face crop is empty or invalid.
    """

    if face is None or face.size == 0:
        return None

    return cv2.resize(face, size)

def face_histogram_equalization(face):
    """Normalize face illumination using histogram equalization."""

    return cv2.equalizeHist(face)

def preprocess_face_for_recognition(gray_frame, x, y, w, h):
    """
    Apply face preprocessing for recognition.

    This pipeline crops the face, resizes it to a fixed resolution,
    and applies histogram equalization to reduce illumination changes.

    If the crop is invalid or empty, the function returns ``None``.
    """

    face = crop_face(gray_frame, x, y, w, h)
    face = resize_face(face)
    if face is None:
        return None
    face = face_histogram_equalization(face)
    return face


def preprocess_face_for_pad(gray_frame, x, y, w, h):
    """
    Apply face preprocessing for presentation attack detection (PAD).

    This pipeline crops the face and resizes it to a fixed resolution
    without applying histogram equalization, in order to preserve
    image quality artifacts useful for spoof detection.

    If the crop is invalid or empty, the function returns ``None``.
    """

    face = crop_face(gray_frame, x, y, w, h)
    face = resize_face(face)
    return face
