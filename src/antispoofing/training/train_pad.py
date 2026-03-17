"""
Module: train_pad

Description:
This module implements the training pipeline for the face anti-spoofing
presentation attack detection (PAD) model.

The script loads the CASIA PAD training dataset, extracts the PAD
feature vectors through the dataset loader, trains the SVM classifier,
and saves both the trained model and the fitted scaler to disk.

The saved artifacts will later be used during runtime inference to
classify detected faces as live or spoof before face recognition.
"""

import joblib
from pathlib import Path

from src.antispoofing.training.pad_dataset_loader import load_pad_dataset
from src.antispoofing.training.svm_classifier import train_pad_classifier


# ----- CONSTANTS ----- #
MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "pad_svm.joblib"
SCALER_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "pad_scaler.joblib"

def train_pad_model():
    """
    Train the PAD model using the CASIA PAD training dataset.

    This function loads the training data, trains the PAD classifier,
    and saves both the trained model and the feature scaler to disk
    for later use during inference.
    """

    # Load the training dataset
    X_train, y_train = load_pad_dataset()

    # print control information about the loaded dataset
    live_count = (y_train == 1).sum()
    spoof_count = (y_train == 0).sum()

    print("Total samples:", len(y_train))
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("Live samples:", live_count)
    print("Spoof samples:", spoof_count)

    # Train the PAD classifier
    scaler, classifier = train_pad_classifier(X_train, y_train)

    # create models directory if it does not exist
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save the trained model and scaler to disk
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(classifier, MODEL_PATH)
 
    print("PAD model trained and saved successfully")
    print("Model path:", MODEL_PATH)
    print("Scaler path:", SCALER_PATH)




if __name__ == "__main__":
    train_pad_model()