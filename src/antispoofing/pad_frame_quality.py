"""
Module: pad_frame_quality

Description:
This module implements runtime quality checks for the grayscale PAD
face crop used by the anti-spoofing pipeline.

The quality evaluation verifies whether the crop is structurally valid
and visually reliable enough to be used by the PAD runtime logic.

The checks include:
- crop existence
- minimum crop size
- plausible aspect ratio
- blur estimation through Laplacian variance

The returned information helps the runtime pipeline decide whether the
current frame is reliable enough to update the temporal PAD history.
"""

import cv2


def evaluate_pad_crop_quality(
    pad_face_crop,
    min_variance_laplacian,
    min_crop_size=80,
    min_aspect_ratio=0.75,
    max_aspect_ratio=1.35,
):
    """
    Evaluate whether the PAD crop is structurally valid and sharp enough
    to be used for temporal PAD smoothing.
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
