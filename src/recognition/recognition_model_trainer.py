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
import pickle
from pathlib import Path
import time

from src.preprocessing.face_preprocessing import get_recognition_preprocessing_config
from src.recognition.recognition_dataset_loader import load_gallery_dataset


# ----- CONSTANTS ----- #
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "lbph_model.xml"
LABEL_MAP_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "label_map.pkl"
TRAIN_MODE = "custom_orl"
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


def train_recognition_model(train_mode=None, lbph_params=None):
    """
    Train the recognition model using the selected gallery configuration.

    Training modes are defined by the shared dataset loader. The
    default project setup uses `custom_orl`, while experiments may still
    select `custom`, `orl`, `lfw`, or `all`.

    LBPH hyperparameters can be overridden through ``lbph_params``.
    """
    selected_mode = train_mode or TRAIN_MODE
    selected_lbph_params = resolve_lbph_params(lbph_params)
    total_start_time = time.perf_counter()
    preprocessing_config = get_recognition_preprocessing_config()

    print(f"=== RECOGNITION TRAINING MODE: {selected_mode.upper()} ===")
    print(
        "Recognition preprocessing:",
        f"normalization={preprocessing_config['normalization_mode']},",
        f"padding_ratio={preprocessing_config['crop_padding_ratio']:.3f}",
    )
    print("Loading gallery dataset...")

    load_start_time = time.perf_counter()
    faces, labels, label_map, dataset_counts = load_gallery_dataset(selected_mode)
    load_elapsed = time.perf_counter() - load_start_time

    if len(faces) == 0:
        raise ValueError("No valid gallery images were found for recognition training.")

    print("Loaded faces:", len(faces))
    print("Loaded identities:", len(label_map))
    print(f"Gallery loading completed in {load_elapsed:.2f}s")

    for dataset_name, sample_count in dataset_counts.items():
        print(f"{dataset_name} gallery samples: {sample_count}")

    print("Training LBPH model...")
    print(
        "LBPH params:",
        ", ".join(f"{key}={value}" for key, value in selected_lbph_params.items()),
    )

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

    with open(LABEL_MAP_PATH, "wb") as file:
        pickle.dump(label_map, file)

    print("Recognition model trained and saved successfully")
    print("Model path:", MODEL_PATH)
    print("Label map path:", LABEL_MAP_PATH)
    print(f"LBPH training completed in {training_elapsed:.2f}s")
    print(f"Total training time: {time.perf_counter() - total_start_time:.2f}s")


if __name__ == "__main__":
    train_recognition_model()
