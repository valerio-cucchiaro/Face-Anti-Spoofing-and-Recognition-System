import numpy as np
import pytest

from src.antispoofing.pad_crop_utils import extract_pad_face_crop
from src.antispoofing.pad_feature_extractor import (
    configure_pad_feature_modes,
    get_pad_feature_modes,
)
from src.antispoofing.pad_inference_preprocessing import extract_pad_features_from_crop
import src.antispoofing.training.pad_dataset_preprocessing as pad_dataset_preprocessing
from src.antispoofing.training.pad_dataset_preprocessing import (
    extract_pad_features_from_dataset_image,
    extract_pad_features_from_final_crop,
    extract_pad_features_from_raw_frame,
)


@pytest.fixture(autouse=True)
def restore_pad_feature_modes():
    configure_pad_feature_modes(
        debug_mode="baseline",
        lft_mode="ggd_light",
        tgsd_tau=170,
        tgsd_top_percent=0.05,
        tgsd_gradient_operator="sobel",
    )
    yield
    configure_pad_feature_modes(
        debug_mode="baseline",
        lft_mode="ggd_light",
        tgsd_tau=170,
        tgsd_top_percent=0.05,
        tgsd_gradient_operator="sobel",
    )


def test_runtime_and_dataset_final_crop_paths_match(sharp_gray_face):
    dataset_features = extract_pad_features_from_final_crop(
        sharp_gray_face,
        apply_quality_gate=False,
    )
    runtime_features = extract_pad_features_from_crop(
        sharp_gray_face,
        apply_quality_gate=False,
    )

    assert dataset_features.dtype == np.float32
    assert runtime_features.dtype == np.float32
    assert dataset_features.shape == runtime_features.shape
    assert np.allclose(dataset_features, runtime_features)


def test_runtime_and_dataset_final_crop_paths_keep_blurry_crop_when_geometry_is_valid(
    blurry_gray_face,
):
    dataset_features = extract_pad_features_from_final_crop(
        blurry_gray_face,
        apply_quality_gate=True,
    )
    runtime_features = extract_pad_features_from_crop(
        blurry_gray_face,
        apply_quality_gate=True,
    )

    assert dataset_features is not None
    assert runtime_features is not None
    assert np.allclose(dataset_features, runtime_features)


def test_raw_frame_and_final_crop_paths_match(monkeypatch, sharp_gray_face):
    bgr_frame = np.stack([sharp_gray_face] * 3, axis=-1)
    face_rect = (16, 16, 96, 96)
    gray_frame = sharp_gray_face.copy()

    monkeypatch.setattr(
        pad_dataset_preprocessing,
        "get_valid_face",
        lambda img, detector, threshold_area: (face_rect, gray_frame, None),
    )

    raw_features = extract_pad_features_from_raw_frame(
        bgr_frame,
        detector=object(),
        apply_quality_gate=False,
    )
    final_crop = extract_pad_face_crop(gray_frame, *face_rect)
    final_features = extract_pad_features_from_final_crop(
        final_crop,
        apply_quality_gate=False,
    )

    assert raw_features.dtype == np.float32
    assert final_features.dtype == np.float32
    assert raw_features.shape == final_features.shape
    assert np.allclose(raw_features, final_features)


def test_extract_pad_features_from_raw_frame_returns_none_when_no_face_found(
    monkeypatch,
    sharp_gray_face,
):
    bgr_frame = np.stack([sharp_gray_face] * 3, axis=-1)

    monkeypatch.setattr(
        pad_dataset_preprocessing,
        "get_valid_face",
        lambda img, detector, threshold_area: (None, None, None),
    )

    result = extract_pad_features_from_raw_frame(
        bgr_frame,
        detector=object(),
        apply_quality_gate=False,
    )

    assert result is None


def test_raw_frame_path_with_quality_gate_keeps_blurry_crop_when_geometry_is_valid(
    monkeypatch,
    blurry_gray_face,
):
    bgr_frame = np.stack([blurry_gray_face] * 3, axis=-1)
    face_rect = (16, 16, 96, 96)

    monkeypatch.setattr(
        pad_dataset_preprocessing,
        "get_valid_face",
        lambda img, detector, threshold_area: (face_rect, blurry_gray_face, None),
    )

    result = extract_pad_features_from_raw_frame(
        bgr_frame,
        detector=object(),
        apply_quality_gate=True,
    )

    assert result is not None


def test_extract_pad_features_from_dataset_image_requires_detector_for_raw_frame(
    sharp_gray_face,
):
    bgr_frame = np.stack([sharp_gray_face] * 3, axis=-1)

    with pytest.raises(ValueError, match="detector is required"):
        extract_pad_features_from_dataset_image(
            bgr_frame,
            detector=None,
            dataset_type="raw_frame",
            apply_quality_gate=False,
        )


def test_extract_pad_features_from_dataset_image_rejects_unknown_dataset_type(
    sharp_gray_face,
):
    with pytest.raises(ValueError, match="Unsupported dataset_type"):
        extract_pad_features_from_dataset_image(
            sharp_gray_face,
            dataset_type="unsupported",
            apply_quality_gate=False,
        )


def test_pad_feature_mode_defaults_are_restored_for_contract_tests():
    active_modes = get_pad_feature_modes()

    assert active_modes["pad_debug_feature_mode"] == "baseline"
    assert active_modes["pad_lft_mode"] == "ggd_light"
    assert active_modes["tgsd_tau"] == 170.0
    assert active_modes["tgsd_top_percent"] == 0.05
    assert active_modes["tgsd_gradient_operator"] == "sobel"
