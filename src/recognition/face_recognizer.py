"""
Module: face_recognizer

Description:
This module provides the runtime face recognition functionality of the
system. It loads the trained LBPH model and the label mapping used to
associate numeric labels with person identities, together with the
runtime threshold configuration saved by the evaluator.

The module exposes utilities to initialize the recognizer and to
predict the identity of a preprocessed face image during the runtime
pipeline.
"""

import cv2
import json
import pickle
from pathlib import Path


# ----- CONSTANTS ----- #
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "lbph_model.xml"
LABEL_MAP_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "label_map.pkl"
RECOGNITION_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "recognition_config.json"
UNKNOWN_THRESHOLD = 75.0


def load_recognition_config():
    """
    Load the recognition runtime configuration if available.

    The evaluator stores the latest selected runtime threshold in a JSON
    file. If that file is missing or incomplete, the runtime falls back
    to the default static threshold.
    """
    if not RECOGNITION_CONFIG_PATH.exists():
        return {
            "unknown_threshold": UNKNOWN_THRESHOLD,
            "threshold_source": "default_fallback",
        }

    with open(RECOGNITION_CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    if "unknown_threshold" not in config:
        config["unknown_threshold"] = UNKNOWN_THRESHOLD
        config["threshold_source"] = "default_fallback_missing_key"

    return config


def load_recognizer():
    """Load the trained LBPH recognition model, label map, and runtime config."""

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

    recognition_config = load_recognition_config()

    return recognizer, label_map, recognition_config


def predict_identity(face, recognizer, label_map, threshold=None):
    """
    Predict the identity of a preprocessed face using the trained model.

    The function returns the predicted identity and the confidence score,
    applying a threshold to classify low-confidence matches as "Unknown".
    """
    if threshold is None:
        threshold = UNKNOWN_THRESHOLD

    # predict label and confidence from preprocessed face
    label, confidence = recognizer.predict(face)

    # apply threshold for unknown identity
    if confidence < threshold:
        predicted_name = label_map[label]
    else:
        predicted_name = "Unknown"

    return predicted_name, confidence
