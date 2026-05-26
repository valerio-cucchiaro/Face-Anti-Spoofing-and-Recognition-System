from pathlib import Path

import pytest

from src.dataset import multipie_dataset_splitter
from src.dataset import pad_runtime_dataset_collector
from src.dataset import recognition_dataset_collector
from src.dataset import replayattack_frame_extractor


def test_replayattack_split_policy_label_and_output_stem(monkeypatch):
    monkeypatch.setattr(replayattack_frame_extractor, "REPLAY_PATH", Path("dataset"))
    video_path = Path("dataset") / "devel" / "attack" / "mobile cam" / "clip-01.mov"

    assert replayattack_frame_extractor.get_output_split(video_path, "legacy") == "train"
    assert (
        replayattack_frame_extractor.get_output_split(video_path, "validation")
        == "validation"
    )
    assert replayattack_frame_extractor.get_output_label(video_path) == "spoof"
    assert (
        replayattack_frame_extractor.build_output_stem(video_path)
        == "devel__attack__mobile_cam__clip_01"
    )


def test_replayattack_rejects_unknown_split_policy():
    with pytest.raises(ValueError, match="Unsupported split policy"):
        replayattack_frame_extractor.get_output_split(Path("test.mov"), "unsupported")


def test_pad_runtime_collector_paths_and_next_index(tmp_path, monkeypatch):
    monkeypatch.setattr(pad_runtime_dataset_collector, "PAD_CUSTOM_RAW_ROOT", tmp_path)

    split_root, live_path, spoof_path = (
        pad_runtime_dataset_collector.get_custom_split_paths("train")
    )

    live_path.mkdir(parents=True)
    (live_path / "001.png").write_bytes(b"1")
    (live_path / "002.png").write_bytes(b"2")

    assert split_root == tmp_path / "train"
    assert live_path == tmp_path / "train" / "live"
    assert spoof_path == tmp_path / "train" / "spoof"
    assert pad_runtime_dataset_collector.get_next_image_index(live_path) == 3

    with pytest.raises(ValueError, match="Unsupported split_name"):
        pad_runtime_dataset_collector.get_custom_split_paths("validation")


def test_recognition_collector_paths_and_identity_contamination_guards(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(recognition_dataset_collector, "CUSTOM_RAW_ROOT", tmp_path)

    resolved_path = recognition_dataset_collector.resolve_dataset_dir("gallery", "alice")
    assert resolved_path == tmp_path / "gallery" / "alice"

    (tmp_path / "gallery" / "alice").mkdir(parents=True)
    with pytest.raises(ValueError, match="cannot be collected as probe_unknown"):
        recognition_dataset_collector.validate_custom_identity_split(
            "alice",
            "probe_unknown",
        )

    (tmp_path / "probe_unknown" / "bob").mkdir(parents=True)
    with pytest.raises(ValueError, match="cannot be collected as an enrolled gallery"):
        recognition_dataset_collector.validate_custom_identity_split(
            "bob",
            "gallery",
        )


def test_multipie_filename_parsing_and_subject_selection_are_stable(monkeypatch):
    parsed = multipie_dataset_splitter.parse_multipie_filename(
        Path("001_01_01_010_00_crop_128.png")
    )

    assert parsed["subject_id"] == "001"
    assert parsed["session"] == "01"
    assert parsed["expression"] == "01"
    assert parsed["pose"] == "010"
    assert parsed["illumination"] == "00"

    monkeypatch.setattr(multipie_dataset_splitter, "KNOWN_SUBJECTS", 2)
    monkeypatch.setattr(multipie_dataset_splitter, "UNKNOWN_SUBJECTS", 1)
    monkeypatch.setattr(multipie_dataset_splitter, "RANDOM_SEED", 7)

    known_subjects, unknown_subjects = (
        multipie_dataset_splitter.select_known_unknown_subjects(
            ["001", "002", "003", "004"]
        )
    )
    known_subjects_repeat, unknown_subjects_repeat = (
        multipie_dataset_splitter.select_known_unknown_subjects(
            ["001", "002", "003", "004"]
        )
    )

    assert len(known_subjects) == 2
    assert len(unknown_subjects) == 1
    assert set(known_subjects).isdisjoint(set(unknown_subjects))
    assert known_subjects == known_subjects_repeat
    assert unknown_subjects == unknown_subjects_repeat


def test_multipie_invalid_filename_raises():
    with pytest.raises(ValueError, match="Invalid Multi-PIE filename"):
        multipie_dataset_splitter.parse_multipie_filename(Path("bad_name.png"))
