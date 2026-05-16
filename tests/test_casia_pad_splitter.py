from pathlib import Path

import src.dataset.casia_pad_splitter as casia_pad_splitter


def _write_image(path: Path):
    path.write_bytes(b"img")


def test_casia_pad_splitter_rebuilds_output_from_raw_and_preserves_test(tmp_path, monkeypatch):
    raw_root = tmp_path / "casia_pad_raw"
    output_root = tmp_path / "casia_pad"

    for split in ("train", "test"):
        for label in ("live", "spoof"):
            (raw_root / split / label).mkdir(parents=True, exist_ok=True)

    raw_train_live = raw_root / "train" / "live"
    raw_train_spoof = raw_root / "train" / "spoof"
    raw_test_live = raw_root / "test" / "live"
    raw_test_spoof = raw_root / "test" / "spoof"

    for filename in ("s1v1f1.png", "s1v1f2.png", "s2v1f1.png", "s2v1f2.png"):
        _write_image(raw_train_live / filename)
    for filename in ("s1v2f1.png", "s1v2f2.png", "s2v2f1.png", "s2v2f2.png"):
        _write_image(raw_train_spoof / filename)
    _write_image(raw_test_live / "test_live.png")
    _write_image(raw_test_spoof / "test_spoof.png")

    (output_root / "stale.txt").parent.mkdir(parents=True, exist_ok=True)
    (output_root / "stale.txt").write_text("stale", encoding="utf-8")

    monkeypatch.setattr(casia_pad_splitter, "CASIA_RAW_ROOT", raw_root)
    monkeypatch.setattr(casia_pad_splitter, "CASIA_OUTPUT_ROOT", output_root)

    casia_pad_splitter.main(validation_ratio=0.5, seed=42)

    assert not (output_root / "stale.txt").exists()
    assert sorted(path.name for path in (output_root / "test" / "live").iterdir()) == ["test_live.png"]
    assert sorted(path.name for path in (output_root / "test" / "spoof").iterdir()) == ["test_spoof.png"]
    total_train_validation_live = casia_pad_splitter.count_image_files(output_root / "train" / "live") + casia_pad_splitter.count_image_files(output_root / "validation" / "live")
    total_train_validation_spoof = casia_pad_splitter.count_image_files(output_root / "train" / "spoof") + casia_pad_splitter.count_image_files(output_root / "validation" / "spoof")
    assert total_train_validation_live == 4
    assert total_train_validation_spoof == 4
