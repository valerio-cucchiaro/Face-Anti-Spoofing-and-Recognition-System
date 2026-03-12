"""
Module: face_recognizer

Description:
This module provides the runtime face recognition functionality of the
system. It loads the trained LBPH model and the label mapping used to
associate numeric labels with person identities.

The module exposes utilities to initialize the recognizer and to
predict the identity of a preprocessed face image during the runtime
pipeline.
"""

import cv2
import pickle
from pathlib import Path


# ----- CONSTANTS ----- #
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "lbph_model.xml"
LABEL_MAP_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "label_map.pkl"
UNKNOWN_THRESHOLD = 75.0


def load_recognizer():
    """Load the trained LBPH recognition model and its label mapping."""

    # check if trained model exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Recognition model not found: {MODEL_PATH}")

    # check if label map exists
    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(f"Label map not found: {LABEL_MAP_PATH}")

    # load trained LBPH recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))

    # load label map
    with open(LABEL_MAP_PATH, "rb") as f:
        label_map = pickle.load(f)

    return recognizer, label_map


def predict_identity(face, recognizer, label_map, threshold=UNKNOWN_THRESHOLD):
    """
    Predict the identity of a preprocessed face using the trained model.

    The function returns the predicted identity and the confidence score,
    applying a threshold to classify low-confidence matches as "Unknown".
    """

    # predict label and confidence from preprocessed face
    label, confidence = recognizer.predict(face)

    # apply threshold for unknown identity
    if confidence < threshold:
        predicted_name = label_map[label]
    else:
        predicted_name = "Unknown"

    return predicted_name, confidence