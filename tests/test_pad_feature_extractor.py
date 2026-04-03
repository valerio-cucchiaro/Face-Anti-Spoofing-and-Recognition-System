import numpy as np
import pytest

from src.antispoofing.pad_feature_extractor import (
    configure_pad_feature_modes,
    extract_pad_features,
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


def test_extract_pad_features_baseline_has_fixed_shape(sharp_gray_face):
    features = extract_pad_features(sharp_gray_face)

    assert features.dtype == np.float32
    assert features.shape == (124,)


def test_extract_pad_features_baseline_accepts_bgr_input(sharp_gray_face):
    bgr_face = np.stack([sharp_gray_face] * 3, axis=-1)

    features = extract_pad_features(bgr_face)

    assert features.dtype == np.float32
    assert features.shape == (124,)


def test_extract_pad_features_tgsd_extends_baseline(sharp_gray_face):
    baseline = extract_pad_features(sharp_gray_face)

    configure_pad_feature_modes(debug_mode="tgsd")
    tgsd_features = extract_pad_features(sharp_gray_face)

    assert baseline.shape == (124,)
    assert tgsd_features.shape == (126,)
    assert tgsd_features.shape[0] > baseline.shape[0]
    assert np.allclose(tgsd_features[: baseline.shape[0]], baseline)


def test_extract_pad_features_is_deterministic(sharp_gray_face):
    features_a = extract_pad_features(sharp_gray_face)
    features_b = extract_pad_features(sharp_gray_face)

    assert np.allclose(features_a, features_b)


def test_extract_pad_features_tgsd_is_deterministic(sharp_gray_face):
    configure_pad_feature_modes(debug_mode="tgsd")

    features_a = extract_pad_features(sharp_gray_face)
    features_b = extract_pad_features(sharp_gray_face)

    assert np.allclose(features_a, features_b)


def test_extract_pad_features_constant_crop_has_no_nan_or_inf():
    constant_face = np.full((128, 128), 127, dtype=np.uint8)

    baseline_features = extract_pad_features(constant_face)

    configure_pad_feature_modes(debug_mode="tgsd")
    tgsd_features = extract_pad_features(constant_face)

    assert np.isfinite(baseline_features).all()
    assert np.isfinite(tgsd_features).all()


def test_extract_pad_features_tgsd_differs_from_baseline(sharp_gray_face):
    baseline_features = extract_pad_features(sharp_gray_face)

    configure_pad_feature_modes(debug_mode="tgsd")
    tgsd_features = extract_pad_features(sharp_gray_face)

    assert tgsd_features.shape[0] > baseline_features.shape[0]
    assert np.allclose(tgsd_features[: baseline_features.shape[0]], baseline_features)
    assert not np.allclose(tgsd_features[-2:], 0.0)


def test_configure_pad_feature_modes_rejects_invalid_debug_mode():
    with pytest.raises(ValueError):
        configure_pad_feature_modes(debug_mode="full")


def test_configure_pad_feature_modes_rejects_invalid_lft_mode():
    with pytest.raises(ValueError):
        configure_pad_feature_modes(lft_mode="summary")
