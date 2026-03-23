"""
Module: pad_model_trainer

Description:
This module implements the training pipeline for the face anti-spoofing
presentation attack detection (PAD) model.

The script loads the selected PAD training dataset, trains the SVM
classifier, and saves both the trained model and the fitted scaler
to disk.

The saved artifacts will later be used during runtime inference to
classify detected faces as live or spoof before face recognition.
"""

import joblib
from pathlib import Path

from src.antispoofing.training.pad_dataset_loader import (
    TRAIN_LIVE_PATH,
    TRAIN_SPOOF_PATH,
    load_combined_pad_training_dataset,
    load_combined_with_runtime_pad_training_dataset,
    load_replay_pad_training_dataset,
    load_runtime_pad_training_dataset,
    load_pad_split_dataset,
)
from src.antispoofing.training.pad_svm_trainer import train_pad_classifier


# ----- CONSTANTS ----- #
MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "pad_svm.joblib"
SCALER_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "pad_scaler.joblib"
TRAIN_MODE = "replay"
RUNTIME_REPEAT = 5


def load_configured_training_dataset(train_mode=None):
    """
    Load the PAD training split selected by TRAIN_MODE.

    Supported modes:
    - casia
    - replay
    - combined
    - runtime
    - combined_runtime
    """

    selected_mode = train_mode or TRAIN_MODE

    if selected_mode == "casia":
        print("=== PAD TRAINING MODE: CASIA TRAIN ===")
        return "CASIA train", load_pad_split_dataset(TRAIN_LIVE_PATH, TRAIN_SPOOF_PATH)

    if selected_mode == "replay":
        print("=== PAD TRAINING MODE: REPLAY-ATTACK TRAIN ===")
        return "Replay-Attack train", load_replay_pad_training_dataset()

    if selected_mode == "combined":
        print("=== PAD TRAINING MODE: COMBINED TRAIN ===")
        return "Combined train", load_combined_pad_training_dataset()

    if selected_mode == "runtime":
        print("=== PAD TRAINING MODE: RUNTIME TRAIN ===")
        return "Runtime train", load_runtime_pad_training_dataset()

    if selected_mode == "combined_runtime":
        print("=== PAD TRAINING MODE: COMBINED + RUNTIME TRAIN ===")
        dataset_name = f"Combined + runtime train (runtime_repeat={RUNTIME_REPEAT})"
        return dataset_name, load_combined_with_runtime_pad_training_dataset(
            runtime_repeat=RUNTIME_REPEAT
        )

    raise ValueError(
        "Unsupported TRAIN_MODE: "
        f"{selected_mode}. Use 'casia', 'replay', 'combined', 'runtime', or 'combined_runtime'."
    )


def train_pad_model(train_mode=None):
    """
    Train the PAD model using the selected PAD training dataset.
    """

    # Load the training dataset
    dataset_name, (X_train, y_train) = load_configured_training_dataset(train_mode)

    # print control information about the loaded dataset
    live_count = (y_train == 1).sum()
    spoof_count = (y_train == 0).sum()

    print("Dataset:", dataset_name)
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
