"""
Module: recognition_model_trainer

Description:
This module trains the face recognition model using one selected
gallery dataset or one of the supported mixed gallery configurations.

The trained LBPH model and the corresponding label map are saved to the
shared recognition model paths used by the rest of the project.

The trainer also supports configurable LBPH hyperparameters so the
project can run small tuning campaigns without changing the training
logic itself.
"""

import cv2
import json
import pickle
from pathlib import Path
import time
from datetime import datetime, timezone

from src.preprocessing.face_preprocessing import get_recognition_preprocessing_config
from src.recognition.recognition_dataset_loader import load_gallery_dataset
from src.preprocessing.pose_statistics import format_pose_summary


# ----- CONSTANTS ----- #
RECOGNITION_MODELS_ROOT = Path(__file__).resolve().parent.parent.parent / "models" / "recognition"
MODEL_PATH = RECOGNITION_MODELS_ROOT / "lbph_model.xml"
LABEL_MAP_PATH = RECOGNITION_MODELS_ROOT / "label_map.pkl"
RECOGNIZER_CONFIG_FILENAME = "recognizer_config.json"
RECOGNIZER_CONFIG_PATH = RECOGNITION_MODELS_ROOT / RECOGNIZER_CONFIG_FILENAME
TRAIN_MODE = "custom"
DEFAULT_LBPH_PARAMS = {
    "radius": 1,
    "neighbors": 8,
    "grid_x": 6,
    "grid_y": 6,
}


def resolve_lbph_params(lbph_params=None):
    """
    Resolve the effective LBPH hyperparameters for the current run.

    Parameters not provided by the caller fall back to the project
    defaults so experiments can override only the values they need.
    """
    resolved_params = DEFAULT_LBPH_PARAMS.copy()

    if lbph_params is not None:
        resolved_params.update(lbph_params)

    return resolved_params


def save_recognizer_config(config_data):
    """
    Save the recognition model type and LBPH parameters used for training.
    """

    RECOGNIZER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECOGNIZER_CONFIG_PATH.write_text(
        json.dumps(config_data, indent=4) + "\n",
        encoding="utf-8",
    )


def train_recognition_model(
    train_mode=None,
    lbph_params=None,
    lfw_sample_ratio=None,
    quality_filter_mode="report_only",
    debug_quality=False,
):
    """
    Train the recognition model using the selected gallery configuration.

    Training modes are defined by the shared dataset loader. The
    default project setup uses `custom`, while experiments may still
    select `orl`, `lfw`, `multipie`, or `all`.

    LBPH hyperparameters can be overridden through ``lbph_params``.
    """
    selected_mode = train_mode or TRAIN_MODE
    selected_lbph_params = resolve_lbph_params(lbph_params)
    selected_lfw_sample_ratio = 1.0 if lfw_sample_ratio is None else lfw_sample_ratio
    total_start_time = time.perf_counter()
    preprocessing_config = get_recognition_preprocessing_config()

    print(f"=== RECOGNITION TRAINING MODE: {selected_mode.upper()} ===")
    print(
        "Recognition preprocessing:",
        f"normalization={preprocessing_config['normalization_mode']},",
        f"padding_ratio={preprocessing_config['crop_padding_ratio']:.3f}",
    )
    print("Recognition protocol:")
    print("- enrollment split: gallery")
    print("- gallery images are used to train LBPH templates")
    print("- validation/test images are not used during training")
    print(f"- quality filter mode: {quality_filter_mode}")
    print("Loading gallery dataset...")

    load_start_time = time.perf_counter()
    faces, labels, label_map, dataset_counts, pose_summary = load_gallery_dataset(
        selected_mode,
        collect_pose_metrics=True,
        lfw_sample_ratio=selected_lfw_sample_ratio,
        quality_filter_mode=quality_filter_mode,
        debug_quality=debug_quality,
    )
    load_elapsed = time.perf_counter() - load_start_time

    if len(faces) == 0:
        raise ValueError("No valid gallery images were found for recognition training.")

    print("Loaded faces:", len(faces))
    print("Loaded identities:", len(label_map))
    print(f"Gallery loading completed in {load_elapsed:.2f}s")

    for dataset_name, sample_count in dataset_counts.items():
        print(f"{dataset_name} gallery samples: {sample_count}")
    if pose_summary is not None:
        print(format_pose_summary(pose_summary, title="Gallery pose summary:"))

    print("Training LBPH model...")
    print(
        "LBPH params:",
        ", ".join(f"{key}={value}" for key, value in selected_lbph_params.items()),
    )
    if "lfw" in selected_mode:
        print(f"LFW sample ratio: {selected_lfw_sample_ratio:.3f}")

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=selected_lbph_params["radius"],
        neighbors=selected_lbph_params["neighbors"],
        grid_x=selected_lbph_params["grid_x"],
        grid_y=selected_lbph_params["grid_y"],
    )

    training_start_time = time.perf_counter()
    recognizer.train(faces, labels)
    training_elapsed = time.perf_counter() - training_start_time

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    recognizer.save(str(MODEL_PATH))
    save_recognizer_config(
        {
            "config_version": 2,
            "recognizer_type": "lbph_single",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "train_mode": selected_mode,
            "lbph_params": {
                "radius": selected_lbph_params["radius"],
                "neighbors": selected_lbph_params["neighbors"],
                "grid_x": selected_lbph_params["grid_x"],
                "grid_y": selected_lbph_params["grid_y"],
            },
            "preprocessing": preprocessing_config,
            "quality_filter_mode": quality_filter_mode,
            "lfw_sample_ratio": selected_lfw_sample_ratio,
            "dataset_counts": dataset_counts,
            "loaded_faces": len(faces),
            "loaded_identities": len(label_map),
        }
    )

    with open(LABEL_MAP_PATH, "wb") as file:
        pickle.dump(label_map, file)

    print("Recognition model trained and saved successfully")
    print("Model path:", MODEL_PATH)
    print("Label map path:", LABEL_MAP_PATH)
    print("Recognizer config path:", RECOGNIZER_CONFIG_PATH)
    print(f"LBPH training completed in {training_elapsed:.2f}s")
    print(f"Total training time: {time.perf_counter() - total_start_time:.2f}s")


if __name__ == "__main__":
    train_recognition_model()
