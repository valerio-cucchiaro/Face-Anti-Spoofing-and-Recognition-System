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
    """Face Crop: extract the face region from the grayscale frame."""

    return gray_frame[y:y+h, x:x+w]

def resize_face(face, size=(128, 128)):
    """Resize a face image to a fixed resolution."""

    return cv2.resize(face, size)

def face_histogram_equalization(face):
    """Normalize face illumination using histogram equalization."""

    return cv2.equalizeHist(face)

def preprocess_face(gray_frame, x, y, w, h):
    """
    Apply the complete preprocessing pipeline to a detected face
    (crop, resize, and illumination normalization).
    """

    face = crop_face(gray_frame, x, y, w, h)
    face = resize_face(face)
    face = face_histogram_equalization(face)
    return face