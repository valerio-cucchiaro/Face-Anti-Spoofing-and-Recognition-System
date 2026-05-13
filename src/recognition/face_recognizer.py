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
RECOGNITION_MODELS_ROOT = Path(__file__).resolve().parent.parent.parent / "models" / "recognition"
MODEL_PATH = RECOGNITION_MODELS_ROOT / "lbph_model.xml"
LABEL_MAP_PATH = RECOGNITION_MODELS_ROOT / "label_map.pkl"
# Fallback for direct runtime calls that do not pass an explicit
# threshold. Offline evaluator/system calibration writes a separate
# system_recognition_threshold_config.json and does not overwrite this.
DEFAULT_PREDICTION_THRESHOLD = 44.0


def load_recognizer():
    """
    Load the trained LBPH recognition model and label map used by the
    runtime pipelines.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Recognition model not found: {MODEL_PATH}")

    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(f"Label map not found: {LABEL_MAP_PATH}")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))

    with open(LABEL_MAP_PATH, "rb") as f:
        label_map = pickle.load(f)

    return recognizer, label_map


def predict_identity(face, recognizer, label_map, threshold=None):
    """
    Predict the identity of a preprocessed face using the trained model.

    The function returns the predicted identity and the LBPH distance.
    OpenCV names this value "confidence", but lower values represent
    better matches. If no explicit threshold is provided, the project
    runtime fallback threshold is used to classify weak matches as
    ``"Unknown"``.
    """
    if threshold is None:
        threshold = DEFAULT_PREDICTION_THRESHOLD

    label, confidence = recognizer.predict(face)

    if confidence < threshold:
        predicted_name = label_map[label]
    else:
        predicted_name = "Unknown"

    return predicted_name, confidence


def predict_identity_safe(face, recognizer, label_map, threshold=None):
    """
    Runtime-safe identity prediction.

    The plain ``predict_identity`` helper intentionally keeps a strict
    contract and raises when the model output is inconsistent. Webcam
    sessions should instead keep running: this wrapper converts
    recognizer failures and missing label-map entries into an Unknown
    decision plus a failure status that callers can log.
    """
    if threshold is None:
        threshold = DEFAULT_PREDICTION_THRESHOLD

    try:
        label, confidence = recognizer.predict(face)
    except Exception as exc:
        return {
            "predicted_name": "Unknown",
            "confidence": None,
            "failure_status": "predict_failed",
            "error": str(exc),
        }

    if confidence >= threshold:
        return {
            "predicted_name": "Unknown",
            "confidence": confidence,
            "failure_status": None,
            "error": None,
        }

    if label not in label_map:
        return {
            "predicted_name": "Unknown",
            "confidence": confidence,
            "failure_status": "missing_label",
            "error": f"Predicted label not found in label map: {label}",
        }

    return {
        "predicted_name": label_map[label],
        "confidence": confidence,
        "failure_status": None,
        "error": None,
    }
