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
    assert set(result.keys()) == {"status", "prediction", "label", "score", "timings_ms"}


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
    assert set(result.keys()) == {"status", "prediction", "label", "score", "timings_ms"}


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
    assert set(result.keys()) == {"status", "prediction", "label", "score", "timings_ms"}


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


def test_predict_with_real_feature_extractor_and_dummy_model(sharp_gray_face):
    detector = build_detector(prediction=1, probabilities=(0.2, 0.8))

    result = detector.predict(face_image=sharp_gray_face)

    assert result["status"] == "ok"
    assert result["label"] == "live"
    assert result["prediction"] == 1
    assert result["score"] == pytest.approx(0.8)
    assert set(result.keys()) == {"status", "prediction", "label", "score", "timings_ms"}
