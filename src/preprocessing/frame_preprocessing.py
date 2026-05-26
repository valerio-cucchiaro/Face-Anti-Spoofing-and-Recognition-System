"""
Module: frame_preprocessing

Description:
This module implements shared face validation logic for webcam frames.
It is used by the runtime pipeline and by dataset collection/loading
utilities that need a consistent face-selection step before later
processing.

The module ensures that only sufficiently large detected faces are
passed to the next stages such as recognition preprocessing or PAD
crop extraction.
"""

from src.preprocessing.face_preprocessing import to_grayscale
from src.detection.face_detector import detect_faces


DEFAULT_MIN_FACE_AREA = 8000


def get_valid_face(frame, detector, threshold_area):
    """
    Detect and validate a face from a webcam frame using the shared
    MediaPipe detector.

    The function selects the largest detected face and verifies that its
    area satisfies the configured minimum threshold before returning it.

    It returns the face bounding box, the grayscale frame, and a status
    message describing the detection result.
    """

    # grayscale is still useful for later preprocessing stages
    gray_frame = to_grayscale(frame)

    # detect faces from the original frame
    faces = detect_faces(
        detector,
        frame,
    )

    # no faces detected
    if len(faces) == 0:
        return None, None, "No face detected"

    # select the largest detected face
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face

    # reject faces that are too small
    if w * h < threshold_area:
        return None, None, "Move closer to the camera"

    return (x, y, w, h), gray_frame, "Face detected"
