from pathlib import Path

import src.dataset.orl_dataset_splitter as orl_dataset_splitter


def _write_image(path: Path):
    path.write_bytes(b"img")


def test_orl_splitter_summary_avoids_theoretical_totals_for_incomplete_identities(tmp_path, monkeypatch, capsys):
    raw_dir = tmp_path / "raw"
    output_root = tmp_path / "out"
    raw_dir.mkdir()

    monkeypatch.setattr(orl_dataset_splitter, "ORL_RAW_DIR", raw_dir)
    monkeypatch.setattr(orl_dataset_splitter, "ORL_GALLERY_DIR", output_root / "gallery")
    monkeypatch.setattr(orl_dataset_splitter, "ORL_VALIDATION_KNOWN_DIR", output_root / "validation_known")
    monkeypatch.setattr(orl_dataset_splitter, "ORL_PROBE_KNOWN_DIR", output_root / "probe_known")
    monkeypatch.setattr(orl_dataset_splitter, "ORL_VALIDATION_UNKNOWN_DIR", output_root / "validation_unknown")
    monkeypatch.setattr(orl_dataset_splitter, "ORL_PROBE_UNKNOWN_DIR", output_root / "probe_unknown")
    monkeypatch.setattr(
        orl_dataset_splitter,
        "OUTPUT_DIRS",
        (
            output_root / "gallery",
            output_root / "validation_known",
            output_root / "probe_known",
            output_root / "validation_unknown",
            output_root / "probe_unknown",
        ),
    )
    monkeypatch.setattr(orl_dataset_splitter, "KNOWN_SUBJECTS", 1)
    monkeypatch.setattr(orl_dataset_splitter, "UNKNOWN_SUBJECTS", 1)
    monkeypatch.setattr(orl_dataset_splitter, "VALIDATION_UNKNOWN_SUBJECTS", 1)

    known_dir = raw_dir / "s1"
    unknown_dir = raw_dir / "s2"
    known_dir.mkdir()
    unknown_dir.mkdir()

    for index in range(1, 10):
        _write_image(known_dir / f"{index}.pgm")
    for index in range(1, 11):
        _write_image(unknown_dir / f"{index}.pgm")

    monkeypatch.setattr(
        orl_dataset_splitter,
        "select_known_unknown_identities",
        lambda identity_dirs: ([known_dir], [unknown_dir]),
    )

    orl_dataset_splitter.prepare_orl_gallery_probe()
    captured = capsys.readouterr().out

    assert "Known identities selected: 1" in captured
    assert "Known identities copied: 0" in captured
    assert "Known identities skipped: 1" in captured
    assert "Validation unknown identities copied: 1" in captured
    assert "Gallery images: 0" in captured
    assert "Validation unknown images: 10" in captured
    assert "unexpected_image_count=1" in captured
