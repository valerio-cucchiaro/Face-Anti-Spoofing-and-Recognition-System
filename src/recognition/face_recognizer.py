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
import json
import pickle
from pathlib import Path


# ----- CONSTANTS ----- #
RECOGNITION_MODELS_ROOT = Path(__file__).resolve().parent.parent.parent / "models" / "recognition"
MODEL_PATH = RECOGNITION_MODELS_ROOT / "lbph_model.xml"
LABEL_MAP_PATH = RECOGNITION_MODELS_ROOT / "label_map.pkl"
RECOGNIZER_CONFIG_PATH = RECOGNITION_MODELS_ROOT / "recognizer_config.json"
# Fallback for direct runtime calls that do not pass an explicit
# threshold. Offline evaluator/system calibration writes a separate
# system_recognition_threshold_config.json and does not overwrite this.
DEFAULT_PREDICTION_THRESHOLD = 44.0


def get_recognition_artifact_paths(train_mode="custom"):
    """
    Resolve the recognition artifact paths associated with one train mode.

    The custom mode owns the canonical runtime artifacts. Other training
    modes write isolated snapshots so they cannot overwrite the runtime
    custom recognizer by accident.
    """
    selected_mode = train_mode or "custom"

    if selected_mode == "custom":
        return {
            "model_path": MODEL_PATH,
            "label_map_path": LABEL_MAP_PATH,
            "config_path": RECOGNIZER_CONFIG_PATH,
        }

    suffix = selected_mode
    return {
        "model_path": RECOGNITION_MODELS_ROOT / f"lbph_model_{suffix}.xml",
        "label_map_path": RECOGNITION_MODELS_ROOT / f"label_map_{suffix}.pkl",
        "config_path": RECOGNITION_MODELS_ROOT / f"recognizer_config_{suffix}.json",
    }


def load_recognizer_config(train_mode="custom"):
    """
    Load the recognizer metadata associated with one artifact set.
    """
    artifact_paths = get_recognition_artifact_paths(train_mode)
    config_path = artifact_paths["config_path"]

    if not config_path.exists():
        raise FileNotFoundError(f"Recognizer config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_recognizer():
    """
    Load the trained LBPH recognition model and label map used by the
    runtime pipelines.
    """
    artifact_paths = get_recognition_artifact_paths("custom")
    model_path = artifact_paths["model_path"]
    label_map_path = artifact_paths["label_map_path"]
    recognizer_config = load_recognizer_config("custom")

    if recognizer_config.get("recognizer_type", "lbph_single") != "lbph_single":
        raise ValueError(
            "Unsupported runtime recognition model type: "
            f"{recognizer_config.get('recognizer_type')}"
        )

    if recognizer_config.get("train_mode") != "custom":
        raise ValueError(
            "Runtime recognition artifacts must come from custom training. "
            f"Found train_mode={recognizer_config.get('train_mode')} in "
            f"{artifact_paths['config_path']}."
        )

    if not model_path.exists():
        raise FileNotFoundError(f"Recognition model not found: {model_path}")

    if not label_map_path.exists():
        raise FileNotFoundError(f"Label map not found: {label_map_path}")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(model_path))

    with open(label_map_path, "rb") as f:
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
