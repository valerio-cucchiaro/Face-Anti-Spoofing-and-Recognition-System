from pathlib import Path

import src.recognition.recognition_dataset_loader as recognition_dataset_loader


def _write_recognition_image(path: Path):
    path.write_bytes(b"img")


def test_split_custom_open_set_identity_images_is_deterministic_and_complementary():
    identity_images = [(Path(f"img_{index}.png"), object()) for index in range(10)]

    validation = recognition_dataset_loader.split_custom_open_set_identity_images(
        identity_images,
        "validation_known_internal",
        "alice",
    )
    probe = recognition_dataset_loader.split_custom_open_set_identity_images(
        identity_images,
        "probe_known_internal",
        "alice",
    )

    assert len(validation) == 4
    assert len(probe) == 6
    assert set(validation).isdisjoint(probe)
    assert set(validation) | set(probe) == set(identity_images)


def test_load_probe_dataset_uses_internal_custom_probe_splits(monkeypatch, tmp_path):
    probe_known_dir = tmp_path / "probe_known"
    probe_unknown_dir = tmp_path / "probe_unknown"
    (probe_known_dir / "alice").mkdir(parents=True)
    (probe_unknown_dir / "mallory").mkdir(parents=True)

    for index in range(5):
        _write_recognition_image(probe_known_dir / "alice" / f"{index}.png")
    for index in range(5):
        _write_recognition_image(probe_unknown_dir / "mallory" / f"{index}.png")

    dataset_dirs = recognition_dataset_loader.DATASET_SPLIT_DIRS["custom"].copy()
    dataset_dirs["probe_known"] = probe_known_dir
    dataset_dirs["probe_unknown"] = probe_unknown_dir
    monkeypatch.setitem(recognition_dataset_loader.DATASET_SPLIT_DIRS, "custom", dataset_dirs)
    monkeypatch.setattr(recognition_dataset_loader, "load_mediapipe_detector", lambda: object())
    monkeypatch.setattr(
        recognition_dataset_loader,
        "load_images_from_identity_dir",
        lambda person_dir, **kwargs: [(path, person_dir.name) for path in sorted(person_dir.iterdir())],
    )

    validation_records, _, _ = recognition_dataset_loader.load_probe_dataset(
        "custom",
        split="validation_known_internal",
        collect_pose_metrics=False,
    )
    probe_records, _, _ = recognition_dataset_loader.load_probe_dataset(
        "custom",
        split="probe_known_internal",
        collect_pose_metrics=False,
    )

    assert len(validation_records) == 2
    assert len(probe_records) == 3
    assert {record["path"] for record in validation_records}.isdisjoint(
        {record["path"] for record in probe_records}
    )
