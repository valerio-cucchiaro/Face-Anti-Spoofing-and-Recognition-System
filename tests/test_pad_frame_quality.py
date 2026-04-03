import numpy as np

from src.antispoofing.pad_frame_quality import (
    PAD_MAX_BBOX_DELTA_H,
    PAD_MAX_BBOX_DELTA_W,
    PAD_MAX_BBOX_DELTA_X,
    PAD_MAX_BBOX_DELTA_Y,
    PAD_MAX_ASPECT_RATIO,
    PAD_MIN_ASPECT_RATIO,
    PAD_MIN_CROP_SIZE,
    evaluate_bbox_stability,
    evaluate_pad_crop_quality,
)


def test_pad_crop_quality_rejects_none():
    result = evaluate_pad_crop_quality(None)

    assert result["is_valid_crop"] is False
    assert result["is_sharp_enough"] is False
    assert result["is_frame_usable_for_history"] is False


def test_pad_crop_quality_rejects_empty_crop(empty_crop):
    result = evaluate_pad_crop_quality(empty_crop)

    assert result["is_valid_crop"] is False
    assert result["crop_width"] == 0
    assert result["crop_height"] == 0


def test_pad_crop_quality_rejects_small_crop():
    crop = np.full((50, 50), 127, dtype=np.uint8)

    result = evaluate_pad_crop_quality(crop)

    assert result["is_large_enough"] is False
    assert result["is_valid_crop"] is False


def test_pad_crop_quality_rejects_bad_aspect_ratio():
    crop = np.full((80, 160), 127, dtype=np.uint8)

    result = evaluate_pad_crop_quality(crop)

    assert result["has_plausible_aspect_ratio"] is False
    assert result["is_valid_crop"] is False


def test_pad_crop_quality_rejects_blurry_valid_crop(blurry_gray_face):
    result = evaluate_pad_crop_quality(blurry_gray_face)

    assert result["is_valid_crop"]
    assert not result["is_sharp_enough"]
    assert not result["is_frame_usable_for_history"]


def test_pad_crop_quality_accepts_sharp_valid_crop(sharp_gray_face):
    result = evaluate_pad_crop_quality(sharp_gray_face)

    assert result["is_valid_crop"]
    assert result["is_sharp_enough"]
    assert result["is_frame_usable_for_history"]


def test_pad_crop_quality_accepts_valid_bgr_crop(sharp_gray_face):
    crop_bgr = np.stack([sharp_gray_face] * 3, axis=-1)

    result = evaluate_pad_crop_quality(crop_bgr)

    assert result["is_valid_crop"]
    assert result["is_sharp_enough"]
    assert result["crop_width"] == 128
    assert result["crop_height"] == 128


def test_pad_crop_quality_accepts_crop_at_exact_minimum_size():
    grid_y, grid_x = np.indices((PAD_MIN_CROP_SIZE, PAD_MIN_CROP_SIZE))
    crop = (((grid_x // 4 + grid_y // 4) % 2) * 255).astype(np.uint8)

    result = evaluate_pad_crop_quality(crop)

    assert result["is_large_enough"]
    assert result["is_valid_crop"]


def test_pad_crop_quality_accepts_crop_at_min_aspect_ratio_boundary():
    height = 120
    width = int(height * PAD_MIN_ASPECT_RATIO)
    grid_y, grid_x = np.indices((height, width))
    crop = (((grid_x // 4 + grid_y // 4) % 2) * 255).astype(np.uint8)

    result = evaluate_pad_crop_quality(crop)

    assert result["has_plausible_aspect_ratio"]
    assert result["is_valid_crop"]


def test_pad_crop_quality_accepts_crop_at_max_aspect_ratio_boundary():
    height = PAD_MIN_CROP_SIZE
    width = int(height * PAD_MAX_ASPECT_RATIO)
    grid_y, grid_x = np.indices((height, width))
    crop = (((grid_x // 4 + grid_y // 4) % 2) * 255).astype(np.uint8)

    result = evaluate_pad_crop_quality(crop)

    assert result["has_plausible_aspect_ratio"]
    assert result["is_valid_crop"]


def test_bbox_stability_with_no_previous_face_is_stable():
    result = evaluate_bbox_stability((10, 20, 100, 100), None)

    assert result["is_stable"] is True
    assert result["dx"] == 0
    assert result["dy"] == 0
    assert result["dw"] == 0
    assert result["dh"] == 0


def test_bbox_stability_rejects_large_motion():
    result = evaluate_bbox_stability(
        (100, 120, 160, 160),
        (10, 20, 100, 100),
    )

    assert result["is_stable"] is False
    assert result["dx"] > 30


def test_bbox_stability_accepts_small_motion():
    result = evaluate_bbox_stability(
        (15, 24, 110, 108),
        (10, 20, 100, 100),
    )

    assert result["is_stable"] is True
    assert result["dx"] == 5
    assert result["dy"] == 4
    assert result["dw"] == 10
    assert result["dh"] == 8


def test_bbox_stability_accepts_exact_threshold_motion():
    result = evaluate_bbox_stability(
        (
            10 + PAD_MAX_BBOX_DELTA_X,
            20 + PAD_MAX_BBOX_DELTA_Y,
            100 + PAD_MAX_BBOX_DELTA_W,
            100 + PAD_MAX_BBOX_DELTA_H,
        ),
        (10, 20, 100, 100),
    )

    assert result["is_stable"] is True
    assert result["dx"] == PAD_MAX_BBOX_DELTA_X
    assert result["dy"] == PAD_MAX_BBOX_DELTA_Y
    assert result["dw"] == PAD_MAX_BBOX_DELTA_W
    assert result["dh"] == PAD_MAX_BBOX_DELTA_H
