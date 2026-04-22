"""
Module: face_pose

Description:
This module provides lightweight facial pose helpers built on top of
landmark coordinates.

It can estimate roll, yaw, and pitch-like metrics, and can optionally
apply a roll correction to an image crop without changing the rest of
the project pipeline.
"""

from math import atan2, degrees

import cv2

from src.detection.face_landmarks import (
    extract_face_mesh_landmarks,
    get_reference_points,
)


# ----- DEFAULT POSE WEIGHTS ----- #
DEFAULT_ROLL_WEIGHT = 0.34
DEFAULT_YAW_WEIGHT = 0.33
DEFAULT_PITCH_WEIGHT = 0.33

# ----- DEFAULT NORMALIZATION LIMITS ----- #
DEFAULT_MAX_ROLL_DEGREES = 45.0


def _distance(point_a, point_b):
    """
    Euclidean distance between two 2D points.
    """

    if point_a is None or point_b is None:
        return None

    dx = float(point_b[0] - point_a[0])
    dy = float(point_b[1] - point_a[1])
    return (dx * dx + dy * dy) ** 0.5


def _clamp(value, min_value=0.0, max_value=1.0):
    """
    Clamp a numeric value to the requested interval.
    """

    return max(min_value, min(max_value, value))


def compute_roll_degrees(left_eye_center, right_eye_center):
    """
    Estimate the in-plane rotation of the face from the eye centers.
    """

    if left_eye_center is None or right_eye_center is None:
        return None

    dx = float(right_eye_center[0] - left_eye_center[0])
    dy = float(right_eye_center[1] - left_eye_center[1])
    return degrees(atan2(dy, dx))


def compute_roll_score(roll_degrees, max_roll_degrees=DEFAULT_MAX_ROLL_DEGREES):
    """
    Convert the roll angle into a normalized frontalness score.
    """

    if roll_degrees is None:
        return None

    if max_roll_degrees <= 0:
        raise ValueError("max_roll_degrees must be greater than zero.")

    return _clamp(1.0 - (abs(float(roll_degrees)) / max_roll_degrees))


def compute_yaw_score(left_eye_center, right_eye_center, nose_tip):
    """
    Estimate a yaw proxy using the asymmetry between eye-to-nose distances.

    The score is high when the face is close to frontal and the two
    distances are similar.
    """

    left_distance = _distance(left_eye_center, nose_tip)
    right_distance = _distance(right_eye_center, nose_tip)

    if left_distance is None or right_distance is None:
        return None

    denominator = left_distance + right_distance
    if denominator <= 0.0:
        return None

    asymmetry = abs(left_distance - right_distance) / denominator
    return _clamp(1.0 - asymmetry)


def compute_pitch_score(left_eye_center, right_eye_center, nose_tip, chin):
    """
    Estimate a pitch proxy using the vertical balance around the nose.

    The score is high when the face is close to frontal and the upper
    and lower facial distances are balanced.
    """

    eye_center = None
    if left_eye_center is not None and right_eye_center is not None:
        eye_center = (
            (left_eye_center[0] + right_eye_center[0]) / 2.0,
            (left_eye_center[1] + right_eye_center[1]) / 2.0,
        )

    upper_distance = _distance(eye_center, nose_tip)
    lower_distance = _distance(nose_tip, chin)

    if upper_distance is None or lower_distance is None:
        return None

    denominator = upper_distance + lower_distance
    if denominator <= 0.0:
        return None

    asymmetry = abs(upper_distance - lower_distance) / denominator
    return _clamp(1.0 - asymmetry)


def compute_pose_metrics_from_landmarks(reference_points):
    """
    Build a compact pose summary from the landmark reference points.
    """

    if reference_points is None:
        return None

    left_eye_center = reference_points.get("left_eye_center")
    right_eye_center = reference_points.get("right_eye_center")
    nose_tip = reference_points.get("nose_tip")
    chin = reference_points.get("chin")

    roll_degrees = compute_roll_degrees(left_eye_center, right_eye_center)
    roll_score = compute_roll_score(roll_degrees)
    yaw_score = compute_yaw_score(left_eye_center, right_eye_center, nose_tip)
    pitch_score = compute_pitch_score(
        left_eye_center,
        right_eye_center,
        nose_tip,
        chin,
    )

    if roll_score is None or yaw_score is None or pitch_score is None:
        sp_score = None
    else:
        sp_score = (
            DEFAULT_ROLL_WEIGHT * roll_score
            + DEFAULT_YAW_WEIGHT * yaw_score
            + DEFAULT_PITCH_WEIGHT * pitch_score
        )

    return {
        "roll_degrees": roll_degrees,
        "roll_score": roll_score,
        "yaw_score": yaw_score,
        "pitch_score": pitch_score,
        "sp_score": sp_score,
        "left_eye_center": left_eye_center,
        "right_eye_center": right_eye_center,
        "nose_tip": nose_tip,
        "chin": chin,
    }


def compute_pose_metrics_from_image(image, face_mesh):
    """
    Compute pose metrics directly from an image using MediaPipe Face Mesh.
    """

    if image is None or image.size == 0 or face_mesh is None:
        return None

    if len(image.shape) == 2:
        bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr_image = image

    landmarks_result = extract_face_mesh_landmarks(face_mesh, bgr_image)
    if landmarks_result is None:
        return None

    reference_points = get_reference_points(landmarks_result.landmarks)
    if reference_points is None:
        return None

    return compute_pose_metrics_from_landmarks(reference_points)


def apply_roll_correction(image, roll_degrees):
    """
    Rotate an image crop to compensate for roll.

    Positive roll angles are corrected by rotating in the opposite
    direction. The output keeps the original canvas size.
    """

    if image is None or image.size == 0:
        return None

    if roll_degrees is None:
        return image

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, -float(roll_degrees), 1.0)

    return cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def format_pose_metrics_for_log(pose_metrics):
    """
    Convert pose metrics into a compact log-friendly string.
    """

    if pose_metrics is None:
        return "pose=None"

    def _fmt(value, digits=2):
        if value is None:
            return "None"
        return f"{value:.{digits}f}"

    return (
        f"roll={_fmt(pose_metrics.get('roll_degrees'))}deg | "
        f"roll_score={_fmt(pose_metrics.get('roll_score'))} | "
        f"yaw_score={_fmt(pose_metrics.get('yaw_score'))} | "
        f"pitch_score={_fmt(pose_metrics.get('pitch_score'))} | "
        f"sp={_fmt(pose_metrics.get('sp_score'))}"
    )
