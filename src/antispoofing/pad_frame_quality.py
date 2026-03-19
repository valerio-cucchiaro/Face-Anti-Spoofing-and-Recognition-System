"""
Module: pad_frame_quality

Description:
This module implements runtime quality checks for the grayscale PAD
face crop used by the anti-spoofing pipeline.

The quality evaluation is based on the variance of the Laplacian,
used as a simple blur estimate. The returned information helps the
runtime pipeline decide whether the current frame is reliable enough
to update the temporal PAD history.
"""

import cv2


def evaluate_pad_crop_quality(pad_face_crop, min_variance_laplacian):
    """
    Evaluate whether the PAD crop is valid and sharp enough to be used
    for temporal PAD smoothing.
    """

    if pad_face_crop is None or pad_face_crop.size == 0:
        return {
            "is_valid_crop": False,
            "laplacian_variance": 0.0,
            "is_sharp_enough": False,
            "is_frame_usable_for_history": False,
        }

    laplacian_variance = cv2.Laplacian(pad_face_crop, cv2.CV_64F).var()
    is_sharp_enough = laplacian_variance >= min_variance_laplacian

    return {
        "is_valid_crop": True,
        "laplacian_variance": float(laplacian_variance),
        "is_sharp_enough": is_sharp_enough,
        "is_frame_usable_for_history": is_sharp_enough,
    }
