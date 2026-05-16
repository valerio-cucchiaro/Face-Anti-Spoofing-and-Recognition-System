from pathlib import Path

import src.dataset.lfw_dataset_splitter as lfw_dataset_splitter


def _write_image(path: Path):
    path.write_bytes(b"img")


def test_lfw_splitter_summary_reports_selected_copied_and_skip_reasons(tmp_path, monkeypatch, capsys):
    raw_dir = tmp_path / "raw"
    output_root = tmp_path / "out"
    raw_dir.mkdir()

    for output_dir_name in (
        "gallery",
        "validation_known",
        "probe_known",
        "validation_unknown",
        "probe_unknown",
    ):
        monkeypatch.setattr(
            lfw_dataset_splitter,
            f"LFW_{output_dir_name.upper()}_DIR" if output_dir_name != "gallery" else "LFW_GALLERY_DIR",
            output_root / output_dir_name,
        )

    monkeypatch.setattr(
        lfw_dataset_splitter,
        "OUTPUT_DIRS",
        tuple(output_root / name for name in ("gallery", "validation_known", "probe_known", "validation_unknown", "probe_unknown")),
    )
    monkeypatch.setattr(lfw_dataset_splitter, "LFW_RAW_DIR", raw_dir)

    alice_dir = raw_dir / "alice"
    bob_dir = raw_dir / "bob"
    carol_dir = raw_dir / "carol"
    empty_dir = raw_dir / "empty"
    for directory in (alice_dir, bob_dir, carol_dir, empty_dir):
        directory.mkdir()

    for index in range(3):
        _write_image(alice_dir / f"a_{index}.jpg")
    for index in range(2):
        _write_image(bob_dir / f"b_{index}.jpg")
    for index in range(3):
        _write_image(carol_dir / f"c_{index}.jpg")

    lfw_dataset_splitter.prepare_lfw_gallery_probe()
    captured = capsys.readouterr().out

    assert "Known identities selected:" in captured
    assert "Known identities copied:" in captured
    assert "Unknown identities selected:" in captured
    assert "Skipped identities:" in captured
    assert "too_few_images=1" in captured
    assert "no_valid_images=1" in captured
