"""
Module: pad_model_evaluator

Description:
This module evaluates the performance of the trained presentation
attack detection (PAD) classifier on one or more selected evaluation
datasets.

The script can evaluate the trained model on:
- CASIA test
- Replay-Attack test
- custom runtime PAD test

It can also run the three evaluations sequentially while keeping the
output of each dataset separated, so the behavior remains equivalent to
running each evaluation mode individually.

Evaluation metrics such as accuracy, precision, recall, F1 score,
and the confusion matrix are computed to assess the ability of the
model to distinguish between live faces and spoof attacks.
"""

import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from src.antispoofing.training.pad_dataset_loader import (
    load_pad_test_dataset_casia,
    load_pad_test_dataset_replay,
    load_pad_split_dataset,
    RUNTIME_TEST_LIVE_PATH,
    RUNTIME_TEST_SPOOF_PATH,
)


# ----- CONSTANTS ----- #
MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "pad_svm.joblib"
SCALER_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "pad_scaler.joblib"
EVAL_MODE = "runtime"     # casia, replay, runtime


def load_runtime_pad_test_dataset():
    """
    Load the custom runtime PAD test dataset.
    """

    return load_pad_split_dataset(
        RUNTIME_TEST_LIVE_PATH,
        RUNTIME_TEST_SPOOF_PATH,
        balance=False,
        dataset_type="final_pad_crop",
    )


def load_configured_test_dataset(eval_mode):
    """
    Load the PAD evaluation dataset selected by ``eval_mode``.

    Supported modes:
    - casia
    - replay
    - runtime
    """

    if eval_mode == "casia":
        print("=== PAD EVALUATION MODE: CASIA TEST ===")
        return "CASIA test", load_pad_test_dataset_casia()

    if eval_mode == "replay":
        print("=== PAD EVALUATION MODE: REPLAY-ATTACK TEST ===")
        return "Replay-Attack test", load_pad_test_dataset_replay()

    if eval_mode == "runtime":
        print("=== PAD EVALUATION MODE: RUNTIME TEST ===")
        return "Runtime test", load_runtime_pad_test_dataset()

    raise ValueError(
        f"Unsupported EVAL_MODE: {eval_mode}. Use 'casia', 'replay', 'runtime', or 'all'."
    )


def evaluate_single_pad_dataset(eval_mode, scaler, classifier):
    """
    Evaluate the trained PAD model on one selected PAD test dataset.
    """

    dataset_name, (X_test, y_test) = load_configured_test_dataset(eval_mode)

    if len(y_test) == 0:
        print("Dataset:", dataset_name)
        print("No valid samples found for evaluation.")
        return

    X_test_scaled = scaler.transform(X_test)
    y_pred = classifier.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Dataset:", dataset_name)
    print("Predicted live:", (y_pred == 1).sum())
    print("Predicted spoof:", (y_pred == 0).sum())
    print("Total test samples:", len(y_test))
    print("Live test samples:", (y_test == 1).sum())
    print("Spoof test samples:", (y_test == 0).sum())
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)

    print("Live mean:", X_test[y_test == 1].mean(axis=0))
    print("Spoof mean:", X_test[y_test == 0].mean(axis=0))
    print()


def evaluate_pad_model(eval_mode=None):
    """
    Evaluate the trained PAD model on one dataset or on all supported
    datasets sequentially.
    """

    selected_mode = eval_mode or EVAL_MODE

    scaler = joblib.load(SCALER_PATH)
    classifier = joblib.load(MODEL_PATH)

    if selected_mode == "all":
        for mode in ("casia", "replay", "runtime"):
            evaluate_single_pad_dataset(mode, scaler, classifier)
        return

    evaluate_single_pad_dataset(selected_mode, scaler, classifier)


if __name__ == "__main__":
    evaluate_pad_model()
