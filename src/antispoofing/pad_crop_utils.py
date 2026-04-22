"""
Module: pad_crop_utils

Description:
This module provides the shared face-cropping utilities used by the
PAD pipeline.

The goal is to keep the PAD face crop consistent across runtime
inference, runtime dataset collection, and offline PAD evaluation or
training helpers.
"""

# ----- CONSTANTS ----- #
PAD_CROP_SCALE = 1.28
PAD_CROP_TOP_BIAS_RATIO = 0.14


def build_pad_crop_bounds(
    frame,
    x,
    y,
    w,
    h,
    crop_scale=PAD_CROP_SCALE,
    top_bias_ratio=PAD_CROP_TOP_BIAS_RATIO,
):
    """
    Compute the canonical PAD crop bounds.

    The crop is a slightly enlarged square centered a bit above the
    detector box center so it keeps more forehead context without adding
    much extra space below the chin.
    """

    if frame is None or frame.size == 0:
        return None

    crop_side = int(max(w, h) * crop_scale)
    crop_side = max(crop_side, 1)

    center_x = x + (w // 2)
    center_y = y + int(h * (0.5 - top_bias_ratio))

    x1 = center_x - (crop_side // 2)
    y1 = center_y - (crop_side // 2)
    x2 = x1 + crop_side
    y2 = y1 + crop_side

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > frame.shape[1]:
        shift = x2 - frame.shape[1]
        x1 = max(0, x1 - shift)
        x2 = frame.shape[1]
    if y2 > frame.shape[0]:
        shift = y2 - frame.shape[0]
        y1 = max(0, y1 - shift)
        y2 = frame.shape[0]

    if x1 >= x2 or y1 >= y2:
        return None

    return x1, y1, x2, y2


def extract_pad_face_crop(gray_frame, x, y, w, h):
    """
    Extract the canonical grayscale PAD crop.
    """

    bounds = build_pad_crop_bounds(gray_frame, x, y, w, h)
    if bounds is None:
        return None

    x1, y1, x2, y2 = bounds
    return gray_frame[y1:y2, x1:x2]


def extract_pad_face_crop_bgr(frame, x, y, w, h):
    """
    Extract the canonical PAD crop from a BGR frame.
    """

    bounds = build_pad_crop_bounds(frame, x, y, w, h)
    if bounds is None:
        return None

    x1, y1, x2, y2 = bounds
    return frame[y1:y2, x1:x2]
