from pathlib import Path

import src.antispoofing.training.pad_dataset_loader as pad_dataset_loader


def _write_pad_image(path: Path):
    path.write_bytes(b"img")


def test_resolve_custom_pad_train_validation_paths_is_deterministic_and_disjoint(tmp_path, monkeypatch):
    live_dir = tmp_path / "train" / "live"
    spoof_dir = tmp_path / "train" / "spoof"
    live_dir.mkdir(parents=True)
    spoof_dir.mkdir(parents=True)

    for index in range(10):
        _write_pad_image(live_dir / f"live_{index}.png")
    for index in range(8):
        _write_pad_image(spoof_dir / f"spoof_{index}.png")

    monkeypatch.setattr(pad_dataset_loader, "CUSTOM_TRAIN_LIVE_PATH", live_dir)
    monkeypatch.setattr(pad_dataset_loader, "CUSTOM_TRAIN_SPOOF_PATH", spoof_dir)
    monkeypatch.setattr(pad_dataset_loader, "_CUSTOM_PAD_VALID_SPLIT_CACHE", {})

    valid_live_paths = [
        live_dir / f"live_{index}.png"
        for index in range(8)
    ]
    valid_spoof_paths = [
        spoof_dir / f"spoof_{index}.png"
        for index in range(6)
    ]

    def fake_load_pad_split_dataset(*args, **kwargs):
        labels = [1] * len(valid_live_paths) + [0] * len(valid_spoof_paths)
        sample_paths = [str(path) for path in valid_live_paths + valid_spoof_paths]
        return None, labels, sample_paths

    monkeypatch.setattr(
        pad_dataset_loader,
        "load_pad_split_dataset",
        fake_load_pad_split_dataset,
    )

    split_a = pad_dataset_loader.resolve_custom_pad_train_validation_paths(
        validation_ratio=0.2,
        validation_seed=42,
    )
    split_b = pad_dataset_loader.resolve_custom_pad_train_validation_paths(
        validation_ratio=0.2,
        validation_seed=42,
    )

    assert split_a == split_b
    assert len(split_a["validation_live_paths"]) == 2
    assert len(split_a["validation_spoof_paths"]) == 1
    assert set(split_a["train_live_paths"]).isdisjoint(split_a["validation_live_paths"])
    assert set(split_a["train_spoof_paths"]).isdisjoint(split_a["validation_spoof_paths"])
    assert set(split_a["train_live_paths"]) | set(split_a["validation_live_paths"]) == set(
        valid_live_paths
    )
    assert set(split_a["train_spoof_paths"]) | set(split_a["validation_spoof_paths"]) == set(
        valid_spoof_paths
    )
