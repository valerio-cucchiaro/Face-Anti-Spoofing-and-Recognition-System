"""
Module: pad_custom_evaluator

Description:
This script evaluates the trained PAD model on the custom PAD test
dataset collected from the real webcam scenario.

The test dataset must be organized as:

data/pad/pad_custom_raw/
    test/
        live/
        spoof/

Each image in this dataset is already a PAD face crop collected from
the live pipeline. For this reason, the evaluator reuses the same
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
    get_pad_artifact_paths,
    load_pad_model_config,
)
from src.antispoofing.pad_probability_utils import (
    get_pad_class_probability_indices,
)
from src.antispoofing.training.pad_dataset_loader import load_custom_pad_test_dataset
from src.preprocessing.pose_statistics import format_pose_summary


# ----- CONSTANTS ----- #
PAD_RUNTIME_ARTIFACT_PATHS = get_pad_artifact_paths("runtime")
MODEL_PATH = PAD_RUNTIME_ARTIFACT_PATHS["model_path"]
SCALER_PATH = PAD_RUNTIME_ARTIFACT_PATHS["scaler_path"]

PAD_CUSTOM_RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "pad" / "pad_custom_raw"
TEST_SPLIT_ROOT = PAD_CUSTOM_RAW_ROOT / "test"
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

        # Custom PAD dataset samples are already PAD crops, so we must
        # reuse the same preprocessing path used by live inference.
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
                "PAD custom evaluator feature mismatch: extracted "
                f"{extracted_feature_dim} features, but the loaded model expects "
                f"{expected_feature_dim}."
            )
        features_scaled = scaler.transform(features)

        prediction = int(classifier.predict(features_scaled)[0])
        probabilities = classifier.predict_proba(features_scaled)[0]
        pad_class_indices = get_pad_class_probability_indices(classifier)

        prob_spoof = float(probabilities[pad_class_indices["spoof"]])
        prob_live = float(probabilities[pad_class_indices["live"]])

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
            print(
                f"[WRONG] {img_path.name} | true={true_label_name} | pred={predicted_label} "
                f"| prob_spoof={prob_spoof:.4f} | prob_live={prob_live:.4f} "
                f"| feature_mean={features.mean():.4f} | feature_std={features.std():.4f}"
            )

    return total_images, valid_images, correct_predictions


def evaluate_loaded_dataset(X_test, y_test, sample_paths, pose_summary, scaler, classifier):
    """Evaluate the custom PAD test dataset already loaded as features."""

    if len(y_test) == 0:
        print("PAD custom evaluation dataset: unavailable")
        print("No valid samples found for evaluation.")
        return

    X_test_scaled = scaler.transform(X_test)
    y_pred = classifier.predict(X_test_scaled)

    accuracy = (y_pred == y_test).sum() / len(y_test)
    live_total = int((y_test == 1).sum())
    spoof_total = int((y_test == 0).sum())
    live_correct = int(((y_test == 1) & (y_pred == 1)).sum())
    spoof_correct = int(((y_test == 0) & (y_pred == 0)).sum())

    print("PAD custom evaluation dataset:", "Custom PAD test")
    if pose_summary is not None:
        print(format_pose_summary(pose_summary, title="PAD custom pose summary:"))
    print(
        "PAD custom summary:",
        f"total={len(y_test)} | valid={len(y_test)} | correct={(y_pred == y_test).sum()} | accuracy={accuracy:.4f}",
    )
    print(
        "PAD custom per-class:",
        (
            f"live(total={live_total}, valid={live_total}, correct={live_correct}) | "
            f"spoof(total={spoof_total}, valid={spoof_total}, correct={spoof_correct})"
        ),
    )


def main():
    """Run PAD evaluation on the custom PAD test dataset."""

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

    X_test, y_test, sample_paths, pose_summary = load_custom_pad_test_dataset(
        return_image_paths=True,
        collect_pose_metrics=True,
        return_pose_summary=True,
    )

    print("PAD custom evaluation dataset:", "Custom PAD test")
    print(
        "PAD custom feature setup:",
        (
            f"debug={pad_config['pad_debug_feature_mode']} | "
            f"lft={pad_config['pad_lft_mode']} | "
            f"tgsd_tau={pad_config.get('tgsd_tau')} | "
            f"tgsd_top_percent={pad_config.get('tgsd_top_percent')} | "
            f"tgsd_grad={pad_config.get('tgsd_gradient_operator')} | "
            f"feature_dim={pad_config['feature_dim']}"
        ),
    )

    evaluate_loaded_dataset(
        X_test,
        y_test,
        sample_paths,
        pose_summary,
        scaler=scaler,
        classifier=classifier,
    )


if __name__ == "__main__":
    main()
