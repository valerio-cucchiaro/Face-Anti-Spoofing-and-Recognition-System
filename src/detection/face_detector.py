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

def load_face_detector(model_path: str):
    """Load and initialize the Haar Cascade face detector."""

    detector = cv2.CascadeClassifier(model_path)

    if detector.empty():
        raise RuntimeError("Failed to load Haar Cascade classifier")
    
    return detector

def detect_faces(detector, gray_frame, scaleFactor, minNeighbors, minSize):
    """Detect faces in a grayscale frame using the Haar Cascade classifier."""

    return detector.detectMultiScale(
        gray_frame, 
        scaleFactor=scaleFactor, 
        minNeighbors=minNeighbors,
        minSize=minSize
    )