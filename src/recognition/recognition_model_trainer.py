"""
Module: recognition_model_trainer

Description:
This module trains the face recognition model using one selected
gallery dataset or the combination of all supported gallery datasets.

The trained LBPH model and the corresponding label map are saved to the
shared recognition model paths used by the rest of the project.
"""

import cv2
import pickle
from pathlib import Path
import time

from src.recognition.recognition_dataset_loader import load_gallery_dataset


# ----- CONSTANTS ----- #
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "lbph_model.xml"
LABEL_MAP_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "label_map.pkl"
TRAIN_MODE = "custom"


def train_recognition_model(train_mode=None):
    """
    Train the recognition model using the selected gallery configuration.
    """
    selected_mode = train_mode or TRAIN_MODE
    total_start_time = time.perf_counter()

    print(f"=== RECOGNITION TRAINING MODE: {selected_mode.upper()} ===")
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

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8,
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
