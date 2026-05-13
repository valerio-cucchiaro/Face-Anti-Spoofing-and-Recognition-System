"""
Module: face_landmarks

Description:
This module provides MediaPipe Face Mesh helpers for extracting facial
landmarks from webcam frames.

The project already uses MediaPipe face detection for bounding boxes.
This module adds a separate landmark layer that can be reused later by
runtime pipelines, offline evaluation, and preprocessing utilities.
"""

from dataclasses import dataclass
from typing import Iterable

import cv2
import mediapipe as mp


# ----- FACE MESH DEFAULTS ----- #
FACE_MESH_MAX_NUM_FACES = 1
FACE_MESH_MIN_DETECTION_CONFIDENCE = 0.5
FACE_MESH_MIN_TRACKING_CONFIDENCE = 0.5


# ----- LANDMARK INDICES ----- #
# MediaPipe Face Mesh reference points commonly used for a compact pose
# estimate and a lightweight roll correction.
LEFT_EYE_OUTER_INDEX = 33
LEFT_EYE_INNER_INDEX = 133
RIGHT_EYE_INNER_INDEX = 362
RIGHT_EYE_OUTER_INDEX = 263
NOSE_TIP_INDEX = 1
CHIN_INDEX = 152


@dataclass(frozen=True)
class FaceLandmarksResult:
    """
    Compact landmark output for a single detected face.
    """

    landmarks: list[tuple[float, float]]
    image_width: int
    image_height: int


def load_mediapipe_face_mesh(
    static_image_mode=False,
    max_num_faces=FACE_MESH_MAX_NUM_FACES,
    min_detection_confidence=FACE_MESH_MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=FACE_MESH_MIN_TRACKING_CONFIDENCE,
):
    """
    Create a MediaPipe Face Mesh detector.

    The caller is responsible for keeping the returned object alive for
    as long as landmark extraction is needed.
    """

    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        static_image_mode=static_image_mode,
        max_num_faces=max_num_faces,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def _normalized_landmark_to_pixel(
    landmark,
    image_width,
    image_height,
):
    """
    Convert a normalized MediaPipe landmark into pixel coordinates.
    """

    x = float(landmark.x * image_width)
    y = float(landmark.y * image_height)
    return x, y


def extract_face_mesh_landmarks(detector, frame):
    """
    Extract the first face landmark set from a BGR frame.

    Returns ``None`` when no face landmarks are detected.
    """

    if detector is None or frame is None or frame.size == 0:
        return None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb_frame)

    if not results.multi_face_landmarks:
        return None

    image_height, image_width = frame.shape[:2]
    face_landmarks = results.multi_face_landmarks[0].landmark

    landmarks = [
        _normalized_landmark_to_pixel(landmark, image_width, image_height)
        for landmark in face_landmarks
    ]

    return FaceLandmarksResult(
        landmarks=landmarks,
        image_width=image_width,
        image_height=image_height,
    )


def has_required_indices(landmarks: Iterable[tuple[float, float]], required_indices):
    """
    Check whether the extracted landmark set contains the requested indices.
    """

    landmarks_list = list(landmarks)
    return all(0 <= index < len(landmarks_list) for index in required_indices)


def get_landmark_point(landmarks, index):
    """
    Return one landmark point from a pixel-space landmark list.
    """

    if landmarks is None or index < 0 or index >= len(landmarks):
        return None

    return landmarks[index]


def get_eye_center(landmarks, outer_index, inner_index):
    """
    Compute the midpoint of one eye from its outer and inner corners.
    """

    outer = get_landmark_point(landmarks, outer_index)
    inner = get_landmark_point(landmarks, inner_index)

    if outer is None or inner is None:
        return None

    return ((outer[0] + inner[0]) / 2.0, (outer[1] + inner[1]) / 2.0)


def get_reference_points(landmarks):
    """
    Extract the compact subset of landmarks used by the pose module.
    """

    if landmarks is None:
        return None

    left_eye_center = get_eye_center(
        landmarks,
        LEFT_EYE_OUTER_INDEX,
        LEFT_EYE_INNER_INDEX,
    )
    right_eye_center = get_eye_center(
        landmarks,
        RIGHT_EYE_INNER_INDEX,
        RIGHT_EYE_OUTER_INDEX,
    )
    nose_tip = get_landmark_point(landmarks, NOSE_TIP_INDEX)
    chin = get_landmark_point(landmarks, CHIN_INDEX)

    if (
        left_eye_center is None
        or right_eye_center is None
        or nose_tip is None
        or chin is None
    ):
        return None

    return {
        "left_eye_center": left_eye_center,
        "right_eye_center": right_eye_center,
        "nose_tip": nose_tip,
        "chin": chin,
    }
