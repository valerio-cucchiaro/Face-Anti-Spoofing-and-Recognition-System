"""
Module: pad_runtime_evaluator

Description:
This script evaluates the trained PAD model on the custom runtime test
dataset collected from the real webcam scenario.

The test dataset must be organized as:

data/pad_runtime_dataset/
    test/
        live/
        spoof/

Each image in this dataset is already a PAD face crop collected from
the runtime pipeline. For this reason, the evaluator reuses the same
inference path used online:

image
-> PAD face normalization
-> PAD feature extraction
-> scaler
-> SVM prediction

No second face detection or face recropping is performed here.
"""

import cv2
import joblib
from pathlib import Path

from src.antispoofing.pad_inference_preprocessing import extract_pad_features_from_crop
from src.antispoofing.pad_model_config import (
    configure_pad_feature_extractor_from_config,
    get_expected_feature_dim_from_scaler,
    load_pad_model_config,
)


# ----- CONSTANTS ----- #
MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "pad_svm.joblib"
SCALER_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "pad_scaler.joblib"

RUNTIME_DATASET_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data" / "pad_runtime_dataset"
TEST_SPLIT_ROOT = RUNTIME_DATASET_ROOT / "test"
LIVE_PATH = TEST_SPLIT_ROOT / "live"
SPOOF_PATH = TEST_SPLIT_ROOT / "spoof"

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def evaluate_folder(folder_path, true_label, scaler, classifier):
    """
    Evaluate all images inside a folder and print detailed PAD results.
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

        # Runtime dataset samples are already PAD crops, so we must
        # reuse the same preprocessing path used by runtime inference.
        features = extract_pad_features_from_crop(img)

        if features is None:
            print(f"[SKIP] {img_path.name} -> invalid PAD crop")
            continue

        valid_images += 1

        # Reshape because scaler and classifier expect 2D input.
        features = features.reshape(1, -1)
        extracted_feature_dim = int(features.shape[1])
        expected_feature_dim = get_expected_feature_dim_from_scaler(scaler)
        if extracted_feature_dim != expected_feature_dim:
            raise ValueError(
                "PAD runtime evaluator feature mismatch: extracted "
                f"{extracted_feature_dim} features, but the loaded model expects "
                f"{expected_feature_dim}."
            )
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

        if status == "WRONG":
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

    # Load the same scaler and classifier used by the runtime detector.
    scaler = joblib.load(SCALER_PATH)
    classifier = joblib.load(MODEL_PATH)
    expected_feature_dim = get_expected_feature_dim_from_scaler(scaler)
    pad_config = load_pad_model_config(expected_feature_dim=expected_feature_dim)
    configure_pad_feature_extractor_from_config(pad_config)

    print("=== PAD RUNTIME TEST DATASET EVALUATION ===")
    print(f"Live folder : {LIVE_PATH}")
    print(f"Spoof folder: {SPOOF_PATH}")
    print("PAD debug feature mode:", pad_config["pad_debug_feature_mode"])
    print("PAD LFT mode:", pad_config["pad_lft_mode"])
    print("Expected PAD feature dim:", pad_config["feature_dim"])

    print("=== EVALUATING LIVE IMAGES ===")
    live_total, live_valid, live_correct = evaluate_folder(
        LIVE_PATH,
        true_label=1,
        scaler=scaler,
        classifier=classifier
    )

    print("=== EVALUATING SPOOF IMAGES ===")
    spoof_total, spoof_valid, spoof_correct = evaluate_folder(
        SPOOF_PATH,
        true_label=0,
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
