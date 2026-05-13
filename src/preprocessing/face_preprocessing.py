"""
Module: face_preprocessing

Description:
This module implements preprocessing operations applied to detected
faces before they are used for recognition or other analysis tasks.

The preprocessing pipeline standardizes the face images by converting
them to grayscale, extracting the face region, resizing it to a fixed
resolution, and applying histogram equalization for recognition.
"""

import cv2


RECOGNITION_CROP_PADDING_RATIO = 0.10

def to_grayscale(frame):
    """Convert a BGR frame to grayscale."""

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def configure_recognition_preprocessing(
    crop_padding_ratio=None,
):
    """
    Update the active recognition preprocessing configuration.

    This lets experiment runners compare recognition preprocessing
    variants without editing this module before each run.
    """
    global RECOGNITION_CROP_PADDING_RATIO
    if crop_padding_ratio is not None:
        if crop_padding_ratio < 0.0:
            raise ValueError("Recognition crop padding ratio must be >= 0.0.")
        RECOGNITION_CROP_PADDING_RATIO = crop_padding_ratio


def get_recognition_preprocessing_config():
    """
    Return the currently active recognition preprocessing settings.
    """
    return {
        "normalization_mode": "equalize",
        "crop_padding_ratio": RECOGNITION_CROP_PADDING_RATIO,
    }


def crop_face(gray_frame, x, y, w, h, padding_ratio=0.0):
    """
    Extract the face region from the grayscale frame.

    The bounding box is clamped to the image limits so that partially
    out-of-frame detections do not produce invalid slices.
    """

    frame_height, frame_width = gray_frame.shape[:2]
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame_width, x + w + pad_x)
    y2 = min(frame_height, y + h + pad_y)

    if x1 >= x2 or y1 >= y2:
        return None

    return gray_frame[y1:y2, x1:x2]

def resize_face(face, size=(128, 128)):
    """
    Resize a face image to a fixed resolution.

    Returns ``None`` when the provided face crop is empty or invalid.
    """

    if face is None or face.size == 0:
        return None

    return cv2.resize(face, size)

def face_histogram_equalization(face):
    """Normalize face illumination using histogram equalization."""

    return cv2.equalizeHist(face)


def normalize_face_for_recognition(face):
    """
    Apply the standard normalization used by recognition.
    """
    if face is None:
        return None

    return face_histogram_equalization(face)


def pad_loaded_face_image(gray_face, padding_ratio=0.0):
    """
    Add a light replicated border around an already cropped face image.
    """
    if gray_face is None or gray_face.size == 0:
        return None

    if padding_ratio <= 0.0:
        return gray_face

    height, width = gray_face.shape[:2]
    pad_x = int(width * padding_ratio)
    pad_y = int(height * padding_ratio)

    return cv2.copyMakeBorder(
        gray_face,
        pad_y,
        pad_y,
        pad_x,
        pad_x,
        cv2.BORDER_REPLICATE,
    )


def preprocess_loaded_face_for_recognition(
    gray_face,
    crop_padding_ratio=None,
):
    """
    Apply the recognition preprocessing pipeline to an already loaded face image.
    """
    if gray_face is None or gray_face.size == 0:
        return None

    selected_padding_ratio = (
        RECOGNITION_CROP_PADDING_RATIO
        if crop_padding_ratio is None
        else crop_padding_ratio
    )

    face = pad_loaded_face_image(gray_face, padding_ratio=selected_padding_ratio)
    face = resize_face(face)
    if face is None:
        return None

    return normalize_face_for_recognition(face)


def preprocess_face_for_recognition(
    gray_frame,
    x,
    y,
    w,
    h,
    crop_padding_ratio=None,
):
    """
    Apply face preprocessing for recognition.

    This pipeline crops the face, resizes it to a fixed resolution,
    and applies histogram equalization.

    If the crop is invalid or empty, the function returns ``None``.
    """

    selected_padding_ratio = (
        RECOGNITION_CROP_PADDING_RATIO
        if crop_padding_ratio is None
        else crop_padding_ratio
    )

    face = crop_face(gray_frame, x, y, w, h, padding_ratio=selected_padding_ratio)
    face = resize_face(face)
    if face is None:
        return None

    return normalize_face_for_recognition(face)


def preprocess_face_for_pad(gray_frame, x, y, w, h):
    """
    Apply face preprocessing for presentation attack detection (PAD).

    This pipeline crops the face and resizes it to a fixed resolution
    without applying histogram equalization, in order to preserve
    image quality artifacts useful for spoof detection.

    If the crop is invalid or empty, the function returns ``None``.
    """

    face = crop_face(gray_frame, x, y, w, h)
    face = resize_face(face)
    return face
