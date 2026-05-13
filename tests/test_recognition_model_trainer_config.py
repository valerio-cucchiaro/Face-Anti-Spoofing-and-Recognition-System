import json

import numpy as np

from src.recognition import recognition_model_trainer


class DummyLBPHRecognizer:
    def train(self, faces, labels):
        self.faces = faces
        self.labels = labels

    def save(self, path):
        with open(path, "w", encoding="utf-8") as file:
            file.write("dummy lbph model")


def test_train_recognition_model_writes_reproducibility_metadata(tmp_path, monkeypatch):
    model_path = tmp_path / "lbph_model.xml"
    label_map_path = tmp_path / "label_map.pkl"
    config_path = tmp_path / "recognizer_config.json"

    monkeypatch.setattr(recognition_model_trainer, "MODEL_PATH", model_path)
    monkeypatch.setattr(recognition_model_trainer, "LABEL_MAP_PATH", label_map_path)
    monkeypatch.setattr(recognition_model_trainer, "RECOGNIZER_CONFIG_PATH", config_path)
    monkeypatch.setattr(
        recognition_model_trainer.cv2.face,
        "LBPHFaceRecognizer_create",
        lambda **kwargs: DummyLBPHRecognizer(),
    )
    monkeypatch.setattr(
        recognition_model_trainer,
        "load_gallery_dataset",
        lambda *args, **kwargs: (
            [np.zeros((32, 32), dtype=np.uint8)],
            np.array([0], dtype=np.int32),
            {0: "alice"},
            {"custom": 1},
            None,
        ),
    )

    lbph_params = {"radius": 2, "neighbors": 16, "grid_x": 8, "grid_y": 8}
    recognition_model_trainer.train_recognition_model(
        train_mode="custom",
        lbph_params=lbph_params,
        lfw_sample_ratio=0.5,
        quality_filter_mode="report_only",
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["config_version"] == 2
    assert config["recognizer_type"] == "lbph_single"
    assert config["train_mode"] == "custom"
    assert config["lbph_params"] == lbph_params
    assert config["preprocessing"]["normalization_mode"] == "equalize"
    assert config["preprocessing"]["crop_padding_ratio"] == 0.10
    assert config["quality_filter_mode"] == "report_only"
    assert config["lfw_sample_ratio"] == 0.5
    assert config["dataset_counts"] == {"custom": 1}
    assert config["loaded_faces"] == 1
    assert config["loaded_identities"] == 1
    assert "created_at_utc" in config
