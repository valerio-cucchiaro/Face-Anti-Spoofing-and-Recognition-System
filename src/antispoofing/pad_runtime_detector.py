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

from src.antispoofing.pad_inference_preprocessing import (
    extract_pad_features_from_crop_with_timing,
)
from src.antispoofing.pad_model_config import (
    configure_pad_feature_extractor_from_config,
    get_expected_feature_dim_from_scaler,
    get_pad_artifact_paths,
    load_pad_model_config,
)
from src.antispoofing.pad_probability_utils import (
    get_pad_class_probability_indices,
)


class AntiSpoofingDetector:
    """
    Runtime anti-spoofing detector based on a trained PAD classifier.

    The detector receives a cropped face image from the runtime pipeline,
    extracts PAD features, applies the trained scaler, and performs
    classification using the SVM model.
    """

    def __init__(self, artifact_profile="runtime", artifact_tag=None):
        """
        Load the trained scaler and classifier used for PAD inference.
        """

        self.artifact_profile = artifact_profile
        self.artifact_tag = artifact_tag
        self.artifact_paths = get_pad_artifact_paths(
            artifact_profile,
            artifact_tag=artifact_tag,
        )
        self.scaler = joblib.load(self.artifact_paths["scaler_path"])
        self.classifier = joblib.load(self.artifact_paths["model_path"])
        if not hasattr(self.classifier, "predict_proba"):
            raise ValueError(
                "Loaded PAD runtime classifier does not expose predict_proba(). "
                "The runtime PAD pipeline requires probability estimates to "
                "compute live_score and drive the smoothed decision logic."
            )
        self.expected_feature_dim = get_expected_feature_dim_from_scaler(self.scaler)
        self.pad_config = load_pad_model_config(
            expected_feature_dim=self.expected_feature_dim,
            config_path=self.artifact_paths["config_path"],
        )
        configure_pad_feature_extractor_from_config(self.pad_config)
        self.pad_class_indices = get_pad_class_probability_indices(self.classifier)

    def _build_result_payload(
        self,
        *,
        status,
        prediction=None,
        label="unknown",
        score=None,
        live_score=None,
        prediction_label=None,
        probability_label=None,
        prediction_probability_mismatch=False,
        error=None,
        timings_ms=None,
    ):
        """
        Build a uniform PAD runtime result payload.

        Runtime callers should be able to inspect the same keys in both
        success and failure cases without adding special-case guards for
        missing probability fields.
        """

        return {
            "status": status,
            "prediction": prediction,
            "label": label,
            "score": score,
            "live_score": live_score,
            "prediction_label": prediction_label,
            "probability_label": probability_label,
            "prediction_probability_mismatch": prediction_probability_mismatch,
            "error": error,
            "timings_ms": timings_ms if timings_ms is not None else {},
        }

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
            return self._build_result_payload(
                status="invalid_input",
                score=0.0,
                timings_ms={
                    **feature_timings,
                    "total_predict": extraction_elapsed_ms,
                },
            )

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
        pad_class_indices = getattr(
            self,
            "pad_class_indices",
            get_pad_class_probability_indices(self.classifier),
        )
        live_score = float(probabilities[pad_class_indices["live"]])
        probability_label = "live" if live_score >= 0.5 else "spoof"
        prediction_label = "live" if int(prediction) == 1 else "spoof"
        label = probability_label
        if label == "live":
            score = float(probabilities[pad_class_indices["live"]])
        else:
            score = float(probabilities[pad_class_indices["spoof"]])

        # Dictionary containing the PAD inference result
        return self._build_result_payload(
            status="ok",
            prediction=int(prediction),
            label=label,
            score=score,
            live_score=live_score,
            prediction_label=prediction_label,
            probability_label=probability_label,
            prediction_probability_mismatch=prediction_label != probability_label,
            timings_ms={
                **feature_timings,
                "total_predict": extraction_elapsed_ms,
            },
        )

    def predict_safe(self, face_image):
        """
        Runtime-safe PAD prediction.

        The strict ``predict`` method remains useful for tests and
        offline debugging because it exposes inconsistent model or
        feature states immediately. Webcam runtimes should use this
        wrapper instead: technical failures become an explicit unknown
        PAD result that can be logged without closing the session.
        """

        try:
            return self.predict(face_image)
        except Exception as exc:
            return self._build_result_payload(
                status="predict_failed",
                error=str(exc),
            )
