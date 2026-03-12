"""
Module: train_model

Description:
This module trains the face recognition model using the collected
dataset. It loads the dataset of preprocessed face images, trains an
LBPH (Local Binary Patterns Histograms) recognizer, and saves the
trained model to disk.

The module also stores the mapping between numeric labels and
person identities, which is required during the recognition stage.
"""

import cv2
import pickle
from pathlib import Path

from src.dataset.dataset_loader import load_dataset


def train_model():
    """
    Train the LBPH face recognition model using the collected dataset.

    The function loads the dataset, trains the recognizer, and saves both
    the trained model and the label map used to associate numeric labels
    with person identities.
    """

    print("Loading dataset...")

    # load dataset
    faces, labels, label_map = load_dataset()

    print(f"Loaded {len(faces)} faces")
    print("Label map:", label_map)

    # create LBPH recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )

    print("Training LBPH model...")

    # train the model
    recognizer.train(faces, labels)

    # create models directory if it doesn't exist
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # save trained model
    model_path = model_dir / "lbph_model.xml"
    recognizer.save(str(model_path))

    # save label map
    label_map_path = model_dir / "label_map.pkl"

    # save the label map as a pickle file
    with open(label_map_path, "wb") as f:
        pickle.dump(label_map, f)

    print("Model saved at:", model_path)
    print("Label map saved at:", label_map_path)


if __name__ == "__main__":
    train_model()