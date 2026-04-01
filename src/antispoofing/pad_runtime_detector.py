"""
Module: pad_runtime_detector

Description:
This module implements the runtime inference logic for face
anti-spoofing detection (Presentation Attack Detection, PAD).

The module loads the trained PAD scaler and SVM classifier from disk
and applies anti-spoofing feature extraction on a face crop provided
by the main runtime pipeline.

Unlike the offline training pipeline, no face detection is performed
here. The detector expects an already localized and cropped face
region and returns a PAD prediction indicating whether the input
appears to be a live face or a spoof attack.
"""

import joblib
import time
from pathlib import Path

from src.antispoofing.pad_inference_preprocessing import (
    extract_pad_features_from_crop_with_timing,
)
from src.antispoofing.pad_model_config import (
    configure_pad_feature_extractor_from_config,
    get_expected_feature_dim_from_scaler,
    load_pad_model_config,
)


# ----- CONSTANTS ----- #
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "pad_svm.joblib"
SCALER_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "pad_scaler.joblib"


class AntiSpoofingDetector:
    """
    Runtime anti-spoofing detector based on a trained PAD classifier.

    The detector receives a cropped face image from the runtime pipeline,
    extracts PAD features, applies the trained scaler, and performs
    classification using the SVM model.
    """

    def __init__(self):
        """
        Load the trained scaler and classifier used for PAD inference.
        """

        self.scaler = joblib.load(SCALER_PATH)
        self.classifier = joblib.load(MODEL_PATH)
        self.expected_feature_dim = get_expected_feature_dim_from_scaler(self.scaler)
        self.pad_config = load_pad_model_config(
            expected_feature_dim=self.expected_feature_dim
        )
        configure_pad_feature_extractor_from_config(self.pad_config)

    def predict(self, face_image):
        """
        Predict whether the input face crop corresponds to a live face
        or a spoof attack.
        """

        extraction_start_time = time.perf_counter()
        features, feature_timings = extract_pad_features_from_crop_with_timing(
            face_image
        )
        extraction_elapsed_ms = (time.perf_counter() - extraction_start_time) * 1000.0

        if features is None or len(features) == 0:
            return {
                "status": "invalid_input",
                "prediction": None,
                "label": "unknown",
                "score": 0.0,
                "timings_ms": {
                    **feature_timings,
                    "total_predict": extraction_elapsed_ms,
                },
            }

        # Reshape to match classifier input format (n_samples, n_features)
        features = features.reshape(1, -1)

        extracted_feature_dim = int(features.shape[1])
        if extracted_feature_dim != self.expected_feature_dim:
            raise ValueError(
                "PAD runtime feature mismatch: extracted "
                f"{extracted_feature_dim} features, but the loaded model expects "
                f"{self.expected_feature_dim}. Active PAD config: "
                f"{self.pad_config['pad_debug_feature_mode']} + "
                f"{self.pad_config['pad_lft_mode']}."
            )

        # Apply feature normalization
        features_scaled = self.scaler.transform(features)

        # Run PAD classifier
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]

        if prediction == 1:
            label = "live"
            score = probabilities[1]
        else:
            label = "spoof"
            score = probabilities[0]

        # Dictionary containing the PAD inference result
        return {
            "status": "ok",
            "prediction": int(prediction),
            "label": label,
            "score": float(score),
            "timings_ms": {
                **feature_timings,
                "total_predict": extraction_elapsed_ms,
            },
        }
