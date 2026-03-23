"""
Module: pad_crop_utils

Description:
This module provides the shared face-cropping utilities used by the
PAD pipeline.

The goal is to keep the PAD face crop consistent across runtime
inference, runtime dataset collection, and offline PAD evaluation or
training helpers.
"""


# ----- PAD CROP MARGINS ----- #
PAD_MARGIN_X_RATIO = 0.12
PAD_MARGIN_Y_RATIO = 0.18


def extract_pad_face_crop(gray_frame, x, y, w, h):
    """
    Extract the grayscale PAD face crop using the shared runtime
    margins.

    The function expands the detected face box with fixed relative
    margins and clips the result to the frame boundaries.
    """

    if gray_frame is None or gray_frame.size == 0:
        return None

    pad_margin_x = int(w * PAD_MARGIN_X_RATIO)
    pad_margin_y = int(h * PAD_MARGIN_Y_RATIO)

    x1 = max(0, x - pad_margin_x)
    y1 = max(0, y - pad_margin_y)
    x2 = min(gray_frame.shape[1], x + w + pad_margin_x)
    y2 = min(gray_frame.shape[0], y + h + pad_margin_y)

    return gray_frame[y1:y2, x1:x2]
