import numpy as np
import pytest

import src.antispoofing.pad_runtime_detector as detector_module
from src.antispoofing.pad_runtime_detector import AntiSpoofingDetector


class DummyScaler:
    def transform(self, features):
        return features


class DummyClassifier:
    def __init__(self, prediction, probabilities):
        self._prediction = prediction
        self._probabilities = probabilities
        self.classes_ = np.array([0, 1])

    def predict(self, features):
        return [self._prediction]

    def predict_proba(self, features):
        return [self._probabilities]


def build_detector(prediction=1, probabilities=(0.1, 0.9), expected_dim=124):
    detector = AntiSpoofingDetector.__new__(AntiSpoofingDetector)
    detector.scaler = DummyScaler()
    detector.classifier = DummyClassifier(prediction, probabilities)
    detector.expected_feature_dim = expected_dim
    detector.pad_config = {
        "pad_debug_feature_mode": "baseline",
        "pad_lft_mode": "ggd_light",
    }
    return detector


def assert_prediction_result_contract(result):
    required_keys = {
        "status",
        "prediction",
        "label",
        "score",
        "live_score",
        "prediction_label",
        "probability_label",
        "prediction_probability_mismatch",
        "error",
        "timings_ms",
    }
    assert required_keys.issubset(result.keys())


def test_predict_returns_invalid_input_for_empty_features(monkeypatch):
    detector = build_detector()

    monkeypatch.setattr(
        detector_module,
        "extract_pad_features_from_crop_with_timing",
        lambda face_image: (None, {"feature_total": 0.0}),
    )

    result = detector.predict(face_image=None)

    assert result["status"] == "invalid_input"
    assert result["label"] == "unknown"
    assert_prediction_result_contract(result)
    assert result["prediction"] is None
    assert result["score"] == 0.0
    assert result["live_score"] is None
    assert result["prediction_label"] is None
    assert result["probability_label"] is None
    assert result["prediction_probability_mismatch"] is False
    assert result["error"] is None


def test_predict_raises_on_feature_dim_mismatch(monkeypatch):
    detector = build_detector(expected_dim=126)

    monkeypatch.setattr(
        detector_module,
        "extract_pad_features_from_crop_with_timing",
        lambda face_image: (np.array([1.0] * 124, dtype=np.float32), {"feature_total": 1.0}),
    )

    with pytest.raises(ValueError):
        detector.predict(face_image=None)


def test_predict_returns_live_label(monkeypatch):
    detector = build_detector(prediction=1, probabilities=(0.2, 0.8))

    monkeypatch.setattr(
        detector_module,
        "extract_pad_features_from_crop_with_timing",
        lambda face_image: (np.array([1.0] * 124, dtype=np.float32), {"feature_total": 1.0}),
    )

    result = detector.predict(face_image=None)

    assert result["status"] == "ok"
    assert result["label"] == "live"
    assert result["prediction"] == 1
    assert result["score"] == pytest.approx(0.8)
    assert_prediction_result_contract(result)
    assert result["live_score"] == pytest.approx(0.8)
    assert result["prediction_label"] == "live"
    assert result["probability_label"] == "live"
    assert result["prediction_probability_mismatch"] is False


def test_predict_returns_spoof_label(monkeypatch):
    detector = build_detector(prediction=0, probabilities=(0.9, 0.1))

    monkeypatch.setattr(
        detector_module,
        "extract_pad_features_from_crop_with_timing",
        lambda face_image: (np.array([1.0] * 124, dtype=np.float32), {"feature_total": 1.0}),
    )

    result = detector.predict(face_image=None)

    assert result["status"] == "ok"
    assert result["label"] == "spoof"
    assert result["prediction"] == 0
    assert result["score"] == pytest.approx(0.9)
    assert_prediction_result_contract(result)
    assert result["live_score"] == pytest.approx(0.1)
    assert result["prediction_label"] == "spoof"
    assert result["probability_label"] == "spoof"
    assert result["prediction_probability_mismatch"] is False


def test_predict_prefers_probability_label_when_predict_mismatches(monkeypatch):
    detector = build_detector(prediction=1, probabilities=(0.8, 0.2))

    monkeypatch.setattr(
        detector_module,
        "extract_pad_features_from_crop_with_timing",
        lambda face_image: (np.array([1.0] * 124, dtype=np.float32), {"feature_total": 1.0}),
    )

    result = detector.predict(face_image=None)

    assert result["status"] == "ok"
    assert result["prediction"] == 1
    assert result["label"] == "spoof"
    assert result["score"] == pytest.approx(0.8)
    assert result["live_score"] == pytest.approx(0.2)
    assert result["prediction_label"] == "live"
    assert result["probability_label"] == "spoof"
    assert result["prediction_probability_mismatch"] is True


def test_predict_propagates_feature_extractor_error(monkeypatch):
    detector = build_detector()

    def raising_extractor(face_image):
        raise RuntimeError("feature extraction failed")

    monkeypatch.setattr(
        detector_module,
        "extract_pad_features_from_crop_with_timing",
        raising_extractor,
    )

    with pytest.raises(RuntimeError, match="feature extraction failed"):
        detector.predict(face_image=np.zeros((128, 128), dtype=np.uint8))


def test_predict_safe_returns_unknown_on_feature_extractor_error(monkeypatch):
    detector = build_detector()

    def raising_extractor(face_image):
        raise RuntimeError("feature extraction failed")

    monkeypatch.setattr(
        detector_module,
        "extract_pad_features_from_crop_with_timing",
        raising_extractor,
    )

    result = detector.predict_safe(face_image=np.zeros((128, 128), dtype=np.uint8))

    assert result["status"] == "predict_failed"
    assert result["prediction"] is None
    assert result["label"] == "unknown"
    assert result["score"] is None
    assert result["live_score"] is None
    assert result["prediction_label"] is None
    assert result["probability_label"] is None
    assert result["prediction_probability_mismatch"] is False
    assert result["error"] == "feature extraction failed"
    assert result["timings_ms"] == {}


def test_predict_safe_returns_unknown_on_feature_dim_mismatch(monkeypatch):
    detector = build_detector(expected_dim=126)

    monkeypatch.setattr(
        detector_module,
        "extract_pad_features_from_crop_with_timing",
        lambda face_image: (np.array([1.0] * 124, dtype=np.float32), {"feature_total": 1.0}),
    )

    result = detector.predict_safe(face_image=None)

    assert result["status"] == "predict_failed"
    assert result["prediction"] is None
    assert result["label"] == "unknown"
    assert result["score"] is None
    assert result["live_score"] is None
    assert result["prediction_label"] is None
    assert result["probability_label"] is None
    assert result["prediction_probability_mismatch"] is False
    assert "feature mismatch" in result["error"]


def test_predict_with_real_feature_extractor_and_dummy_model(sharp_gray_face):
    detector = build_detector(prediction=1, probabilities=(0.2, 0.8))
    detector_module.configure_pad_feature_extractor_from_config(
        {
            "pad_debug_feature_mode": "baseline",
            "pad_lft_mode": "ggd_light",
            "feature_dim": 124,
        }
    )

    result = detector.predict(face_image=sharp_gray_face)

    assert result["status"] == "ok"
    assert result["label"] == "live"
    assert result["prediction"] == 1
    assert result["score"] == pytest.approx(0.8)
    assert_prediction_result_contract(result)
    assert result["live_score"] == pytest.approx(0.8)
    assert result["prediction_label"] == "live"
    assert result["probability_label"] == "live"
    assert result["prediction_probability_mismatch"] is False
