from pathlib import Path

import pytest

import src.recognition.face_recognizer as face_recognizer_module
from src.recognition.face_recognizer import (
    load_recognizer,
    predict_identity,
    predict_identity_safe,
)


class DummyRecognizer:
    def __init__(self, label=0, confidence=20.0):
        self._label = label
        self._confidence = confidence

    def predict(self, face):
        return self._label, self._confidence


def test_load_recognizer_raises_if_model_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(face_recognizer_module, "MODEL_PATH", tmp_path / "missing_model.xml")
    monkeypatch.setattr(face_recognizer_module, "LABEL_MAP_PATH", tmp_path / "label_map.pkl")

    with pytest.raises(FileNotFoundError):
        load_recognizer()


def test_load_recognizer_raises_if_label_map_is_missing(monkeypatch, tmp_path):
    model_path = tmp_path / "lbph_model.xml"
    model_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(face_recognizer_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(face_recognizer_module, "LABEL_MAP_PATH", tmp_path / "missing_label_map.pkl")

    with pytest.raises(FileNotFoundError):
        load_recognizer()


def test_predict_identity_accepts_match_under_threshold():
    recognizer = DummyRecognizer(label=0, confidence=20.0)
    label_map = {0: "custom:valerio"}

    predicted_name, confidence = predict_identity(
        face="dummy_face",
        recognizer=recognizer,
        label_map=label_map,
        threshold=45.0,
    )

    assert predicted_name == "custom:valerio"
    assert confidence == 20.0


def test_predict_identity_rejects_match_above_threshold():
    recognizer = DummyRecognizer(label=0, confidence=60.0)
    label_map = {0: "custom:valerio"}

    predicted_name, confidence = predict_identity(
        face="dummy_face",
        recognizer=recognizer,
        label_map=label_map,
        threshold=45.0,
    )

    assert predicted_name == "Unknown"
    assert confidence == 60.0


def test_predict_identity_rejects_match_at_exact_threshold():
    recognizer = DummyRecognizer(label=0, confidence=45.0)
    label_map = {0: "custom:valerio"}

    predicted_name, confidence = predict_identity(
        face="dummy_face",
        recognizer=recognizer,
        label_map=label_map,
        threshold=45.0,
    )

    assert predicted_name == "Unknown"
    assert confidence == 45.0


def test_predict_identity_raises_if_predicted_label_is_missing():
    recognizer = DummyRecognizer(label=3, confidence=20.0)
    label_map = {0: "custom:valerio"}

    with pytest.raises(KeyError):
        predict_identity(
            face="dummy_face",
            recognizer=recognizer,
            label_map=label_map,
            threshold=45.0,
        )


def test_predict_identity_propagates_recognizer_error_on_invalid_face():
    class RaisingRecognizer:
        def predict(self, face):
            raise ValueError("invalid face")

    label_map = {0: "custom:valerio"}

    with pytest.raises(ValueError, match="invalid face"):
        predict_identity(
            face=None,
            recognizer=RaisingRecognizer(),
            label_map=label_map,
            threshold=45.0,
        )


def test_predict_identity_safe_returns_unknown_if_predicted_label_is_missing():
    recognizer = DummyRecognizer(label=3, confidence=20.0)
    label_map = {0: "custom:valerio"}

    result = predict_identity_safe(
        face="dummy_face",
        recognizer=recognizer,
        label_map=label_map,
        threshold=45.0,
    )

    assert result["predicted_name"] == "Unknown"
    assert result["confidence"] == 20.0
    assert result["failure_status"] == "missing_label"
    assert "3" in result["error"]


def test_predict_identity_safe_returns_unknown_if_recognizer_fails():
    class RaisingRecognizer:
        def predict(self, face):
            raise ValueError("invalid face")

    label_map = {0: "custom:valerio"}

    result = predict_identity_safe(
        face=None,
        recognizer=RaisingRecognizer(),
        label_map=label_map,
        threshold=45.0,
    )

    assert result["predicted_name"] == "Unknown"
    assert result["confidence"] is None
    assert result["failure_status"] == "predict_failed"
    assert result["error"] == "invalid face"
