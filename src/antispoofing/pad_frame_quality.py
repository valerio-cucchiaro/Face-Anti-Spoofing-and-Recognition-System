"""
Module: pad_frame_quality

Description:
This module implements shared quality checks for the PAD crop and the
runtime-only stability checks used during online acquisition.

The crop-quality evaluation verifies whether the crop is structurally
valid and visually reliable enough to be used by the PAD pipeline.

The checks include:
- crop existence
- minimum crop size
- plausible aspect ratio
- blur estimation through Laplacian variance

The module also exposes a separate bounding-box stability helper for
runtime collection and online temporal logic. That helper is not part
of the PAD feature path.
"""

import cv2

# ----- SHARED PAD CROP QUALITY CONSTANTS ----- #
PAD_MIN_VARIANCE_LAPLACIAN = 80.0
PAD_MIN_CROP_SIZE = 80
PAD_MIN_ASPECT_RATIO = 0.75
PAD_MAX_ASPECT_RATIO = 1.35

# ----- RUNTIME BBOX STABILITY CONSTANTS ----- #
PAD_MAX_BBOX_DELTA_X = 30
PAD_MAX_BBOX_DELTA_Y = 30
PAD_MAX_BBOX_DELTA_W = 35
PAD_MAX_BBOX_DELTA_H = 35


def evaluate_pad_crop_quality(
    pad_face_crop,
    min_variance_laplacian=PAD_MIN_VARIANCE_LAPLACIAN,
    min_crop_size=PAD_MIN_CROP_SIZE,
    min_aspect_ratio=PAD_MIN_ASPECT_RATIO,
    max_aspect_ratio=PAD_MAX_ASPECT_RATIO,
):
    """
    Evaluate whether the PAD crop is structurally valid and sharp enough
    to be used by the PAD pipeline.

    The returned ``is_frame_usable_for_history`` flag combines the
    geometric checks and the sharpness check; callers can use the
    individual fields if they need a weaker or stricter policy.
    """

    if pad_face_crop is None or pad_face_crop.size == 0:
        return {
            "is_valid_crop": False,
            "crop_width": 0,
            "crop_height": 0,
            "aspect_ratio": 0.0,
            "is_large_enough": False,
            "has_plausible_aspect_ratio": False,
            "laplacian_variance": 0.0,
            "is_sharp_enough": False,
            "is_frame_usable_for_history": False,
        }

    crop_height, crop_width = pad_face_crop.shape[:2]
    
    # The aspect ratio check is applied on the extracted PAD crop, not on
    # the original detector box, because this is the actual model input.
    aspect_ratio = crop_width / crop_height if crop_height > 0 else 0.0

    # Reject tiny crops before inference: after resizing they would look
    # formally valid, but the PAD features would come from too little detail.
    is_large_enough = (
        crop_width >= min_crop_size and crop_height >= min_crop_size
    )

    # PAD crops should remain roughly face-shaped after the shared padding.
    # Extremely wide or tall crops usually indicate a bad detection or a
    # crop clipped too aggressively by the frame borders.
    has_plausible_aspect_ratio = (
        min_aspect_ratio <= aspect_ratio <= max_aspect_ratio
    )

    is_valid_crop = is_large_enough and has_plausible_aspect_ratio

    if is_valid_crop:
        # Blur is evaluated only after the geometric checks pass, so we do
        # not treat obviously invalid crops as meaningful sharp/blurred data.
        laplacian_variance = cv2.Laplacian(pad_face_crop, cv2.CV_64F).var()
    else:
        laplacian_variance = 0.0

    is_sharp_enough = laplacian_variance >= min_variance_laplacian

    # Only frames that are both structurally valid and sharp enough are
    # allowed to influence the temporal PAD smoothing state.
    is_frame_usable_for_history = is_valid_crop and is_sharp_enough

    return {
        "is_valid_crop": is_valid_crop,
        "crop_width": crop_width,
        "crop_height": crop_height,
        "aspect_ratio": float(aspect_ratio),
        "is_large_enough": is_large_enough,
        "has_plausible_aspect_ratio": has_plausible_aspect_ratio,
        "laplacian_variance": float(laplacian_variance),
        "is_sharp_enough": is_sharp_enough,
        "is_frame_usable_for_history": is_frame_usable_for_history,
    }


def evaluate_bbox_stability(
    current_face_rect,
    previous_face_rect,
    max_delta_x=PAD_MAX_BBOX_DELTA_X,
    max_delta_y=PAD_MAX_BBOX_DELTA_Y,
    max_delta_w=PAD_MAX_BBOX_DELTA_W,
    max_delta_h=PAD_MAX_BBOX_DELTA_H,
):
    """
    Evaluate whether two consecutive face boxes are stable enough for
    runtime acquisition/history updates.

    This helper belongs to the runtime control logic, not to the PAD
    feature extraction path.
    """

    if current_face_rect is None:
        return {
            "is_stable": False,
            "dx": None,
            "dy": None,
            "dw": None,
            "dh": None,
        }

    if previous_face_rect is None:
        return {
            "is_stable": True,
            "dx": 0,
            "dy": 0,
            "dw": 0,
            "dh": 0,
        }

    x, y, w, h = current_face_rect
    prev_x, prev_y, prev_w, prev_h = previous_face_rect

    dx = abs(x - prev_x)
    dy = abs(y - prev_y)
    dw = abs(w - prev_w)
    dh = abs(h - prev_h)

    is_stable = (
        dx <= max_delta_x and
        dy <= max_delta_y and
        dw <= max_delta_w and
        dh <= max_delta_h
    )

    return {
        "is_stable": is_stable,
        "dx": dx,
        "dy": dy,
        "dw": dw,
        "dh": dh,
    }
