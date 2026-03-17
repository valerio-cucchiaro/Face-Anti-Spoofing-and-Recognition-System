"""
Module: test_pad_runtime_dataset

Description:
This script evaluates the trained PAD model on a custom runtime test
dataset collected from the real webcam scenario.

The test dataset must be organized as:

data/pad_runtime_test/
    live/
    spoof/

For each image, the script reuses the same preprocessing pipeline used
during PAD training:

image
→ face detection
→ face cropping
→ face normalization
→ PAD feature extraction
→ scaler
→ SVM prediction

The script prints detailed per-image results and final summary metrics
to help diagnose whether the PAD model generalizes to real webcam and
phone-spoof samples.
"""

import cv2
import joblib
from pathlib import Path

from src.detection.face_detector import load_face_detector
from src.antispoofing.training.pad_dataset_loader import extract_pad_features_from_img


# ----- CONSTANTS ----- #
MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "pad_svm.joblib"
SCALER_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "pad_scaler.joblib"
DETECTOR_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "haarcascade_frontalface_default.xml"

TEST_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data" / "pad_runtime_test"
LIVE_PATH = TEST_ROOT / "live"
SPOOF_PATH = TEST_ROOT / "spoof"

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def evaluate_folder(folder_path, true_label, detector, scaler, classifier):
    """
    Evaluate all images inside a folder and print detailed PAD results.

    Args:
        folder_path: directory containing test images
        true_label: ground-truth class (1 for live, 0 for spoof)
        detector: loaded face detector
        scaler: fitted PAD scaler
        classifier: trained PAD SVM classifier

    Returns:
        tuple:
            - total_images (int)
            - valid_images (int)
            - correct_predictions (int)
    """

    total_images = 0
    valid_images = 0
    correct_predictions = 0

    for img_path in sorted(folder_path.iterdir()):
        if img_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        total_images += 1

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[SKIP] {img_path.name} -> unreadable image")
            continue

        # Reuse the same feature extraction pipeline used during training
        features = extract_pad_features_from_img(img, detector)
        # features = extract_pad_features(img)

        if features is None:
            print(f"[SKIP] {img_path.name} -> no valid face detected")
            continue

        valid_images += 1

        # Reshape because scaler and classifier expect 2D input
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)

        prediction = int(classifier.predict(features_scaled)[0])
        probabilities = classifier.predict_proba(features_scaled)[0]

        prob_spoof = float(probabilities[0])
        prob_live = float(probabilities[1])

        if prediction == 1:
            predicted_label = "live"
            predicted_score = prob_live
        else:
            predicted_label = "spoof"
            predicted_score = prob_spoof

        true_label_name = "live" if true_label == 1 else "spoof"

        if prediction == true_label:
            correct_predictions += 1
            status = "OK"
        else:
            status = "WRONG"

        print(f"[{status}] {img_path.name}")
        print(f"  true label      : {true_label_name}")
        print(f"  predicted label : {predicted_label}")
        print(f"  prediction      : {prediction}")
        print(f"  prob spoof      : {prob_spoof:.4f}")
        print(f"  prob live       : {prob_live:.4f}")
        print(f"  predicted score : {predicted_score:.4f}")
        print(f"  feature shape   : {features.shape}")
        print(f"  feature mean    : {features.mean():.4f}")
        print(f"  feature std     : {features.std():.4f}")
        print()

    return total_images, valid_images, correct_predictions


def main():
    """Run PAD evaluation on the custom runtime test dataset."""

    if not LIVE_PATH.exists():
        raise FileNotFoundError(f"Missing folder: {LIVE_PATH}")

    if not SPOOF_PATH.exists():
        raise FileNotFoundError(f"Missing folder: {SPOOF_PATH}")

    # Load detector, scaler and classifier
    detector = load_face_detector(str(DETECTOR_PATH))
    scaler = joblib.load(SCALER_PATH)
    classifier = joblib.load(MODEL_PATH)

    print("Classifier classes:", classifier.classes_)
    print()

    print("=== PAD RUNTIME TEST DATASET EVALUATION ===")
    print(f"Live folder : {LIVE_PATH}")
    print(f"Spoof folder: {SPOOF_PATH}")

    print("=== EVALUATING LIVE IMAGES ===")
    live_total, live_valid, live_correct = evaluate_folder(
        LIVE_PATH,
        true_label=1,
        detector=detector,
        scaler=scaler,
        classifier=classifier
    )

    print("=== EVALUATING SPOOF IMAGES ===")
    spoof_total, spoof_valid, spoof_correct = evaluate_folder(
        SPOOF_PATH,
        true_label=0,
        detector=detector,
        scaler=scaler,
        classifier=classifier
    )

    total_images = live_total + spoof_total
    valid_images = live_valid + spoof_valid
    correct_predictions = live_correct + spoof_correct

    accuracy = 0.0
    if valid_images > 0:
        accuracy = correct_predictions / valid_images

    print("=== SUMMARY ===")
    print(f"Total images           : {total_images}")
    print(f"Valid processed images : {valid_images}")
    print(f"Correct predictions    : {correct_predictions}")
    print(f"Accuracy               : {accuracy:.4f}")
    print()
    print("Per-class summary:")
    print(f"  LIVE  -> total: {live_total}, valid: {live_valid}, correct: {live_correct}")
    print(f"  SPOOF -> total: {spoof_total}, valid: {spoof_valid}, correct: {spoof_correct}")


if __name__ == "__main__":
    main()
