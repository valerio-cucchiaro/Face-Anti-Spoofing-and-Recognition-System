"""
Module: evaluate_pad

Description:
This module evaluates the performance of the trained presentation
attack detection (PAD) classifier using the CASIA face anti-spoofing
test dataset.

The script loads the test dataset, extracts the PAD feature vectors
for each image, applies the same feature normalization used during
training, and performs predictions using the trained SVM classifier.

Evaluation metrics such as accuracy, precision, recall, F1 score,
and the confusion matrix are computed to assess the ability of the
model to distinguish between live faces and spoof attacks.
"""

import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from src.antispoofing.training.pad_dataset_loader import load_pad_test_dataset


# ----- CONSTANTS ----- #
MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "pad_svm.joblib"
SCALER_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "pad_scaler.joblib"


def evaluate_pad_model():
    """
    Evaluate the trained PAD model on the CASIA PAD test dataset.

    This function loads the test dataset, applies the same normalization
    used during training, and evaluates the trained SVM classifier.
    """

    # Load the test dataset
    X_test, y_test = load_pad_test_dataset()

    # Load the trained model and scaler from disk
    scaler = joblib.load(SCALER_PATH)
    classifier = joblib.load(MODEL_PATH)

    # Normalize the test features using the loaded scaler
    X_test_scaled = scaler.transform(X_test)

    # Predict using the loaded classifier
    y_pred = classifier.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print evaluation results
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


if __name__ == "__main__":
    evaluate_pad_model()