"""
Module: pad_dataset_preprocessing

Description:
Image-to-feature helpers shared by PAD dataset loading paths. This
module handles photometric augmentation and the dispatch between raw
frames and already localized PAD crops.
"""

import cv2
import numpy as np

from src.antispoofing.pad_crop_utils import extract_pad_face_crop
from src.antispoofing.pad_frame_quality import PAD_MIN_VARIANCE_LAPLACIAN
from src.antispoofing.pad_inference_preprocessing import extract_pad_features_from_crop
from src.preprocessing.face_pose import compute_pose_metrics_from_image
from src.preprocessing.pose_statistics import update_pose_summary
from src.preprocessing.frame_preprocessing import DEFAULT_MIN_FACE_AREA, get_valid_face


THRESHOLD_AREA = DEFAULT_MIN_FACE_AREA
PAD_IMAGE_PROGRESS_STEP = 500
LIVE_PHOTOMETRIC_AUGMENTATION_SEED = 42
LIVE_PHOTOMETRIC_AUGMENTATION_MODE = "mix"


def apply_gamma_adjustment(gray_image, rng):
    """
    Internal helper used by the active ``mix`` live augmentation.
    """
    gamma = float(rng.uniform(0.8, 1.25))
    normalized = gray_image.astype(np.float32) / 255.0
    adjusted = np.power(normalized, gamma)
    return np.clip(adjusted * 255.0, 0, 255).astype(np.uint8), "gamma"


def apply_brightness_contrast_adjustment(gray_image, rng):
    """
    Internal helper used by the active ``mix`` live augmentation.
    """
    alpha = float(rng.uniform(0.9, 1.1))
    beta = float(rng.uniform(-15.0, 15.0))
    adjusted = gray_image.astype(np.float32) * alpha + beta
    return np.clip(adjusted, 0, 255).astype(np.uint8), "brightness_contrast"


def apply_side_shading_adjustment(gray_image, rng):
    """
    Internal helper used by the active ``mix`` live augmentation.
    """
    height, width = gray_image.shape[:2]
    if width <= 1:
        return gray_image.copy(), "side_shading(flat)"

    start = float(rng.uniform(0.88, 1.0))
    end = float(rng.uniform(1.0, 1.12))
    if rng.random() < 0.5:
        start, end = end, start

    gradient = np.linspace(start, end, width, dtype=np.float32)
    shading_map = np.tile(gradient, (height, 1))
    shaded = gray_image.astype(np.float32) * shading_map
    return np.clip(shaded, 0, 255).astype(np.uint8), "side_shading"


def apply_combined_light_adjustment(gray_image, rng):
    """
    Internal helper used by the active ``mix`` live augmentation.
    """
    adjusted_image, _ = apply_gamma_adjustment(gray_image, rng)
    adjusted_image, _ = apply_brightness_contrast_adjustment(adjusted_image, rng)
    return adjusted_image, "combined"


def apply_gamma_side_shading_adjustment(gray_image, rng):
    """
    Internal helper used by the active ``mix`` live augmentation.
    """
    adjusted_image, _ = apply_gamma_adjustment(gray_image, rng)
    adjusted_image, _ = apply_side_shading_adjustment(adjusted_image, rng)
    return adjusted_image, "gamma_side_shading"


def apply_live_photometric_augmentation(gray_image, rng, mode=None):
    """
    Create one mild photometric live augmentation.

    The public PAD training setup now supports only the active
    ``mix`` mode. The helper augmentation functions remain separate
    only so the mix can randomly sample one family per live image.
    """
    selected_mode = mode or LIVE_PHOTOMETRIC_AUGMENTATION_MODE
    if selected_mode != "mix":
        raise ValueError(
            "Unsupported live photometric augmentation mode: "
            f"{selected_mode}. Use 'mix'."
        )

    augmentation_functions = (
        apply_gamma_adjustment,
        apply_brightness_contrast_adjustment,
        apply_side_shading_adjustment,
        apply_combined_light_adjustment,
        apply_gamma_side_shading_adjustment,
    )
    augmentation_function = augmentation_functions[int(rng.integers(len(augmentation_functions)))]
    return augmentation_function(gray_image, rng)


def extract_pad_features_from_raw_frame(
    img,
    detector,
    apply_quality_gate=True,
    collect_pose_metrics=False,
    face_mesh=None,
    pose_summary=None,
):
    """
    Extract PAD features from a raw frame by running face detection and
    shared PAD crop construction, then delegating the post-crop path to
    the canonical PAD crop helper.
    """
    face_box, gray_frame, _ = get_valid_face(
        img,
        detector,
        THRESHOLD_AREA,
    )

    if face_box is None:
        return None

    x, y, w, h = face_box
    pad_face_crop = extract_pad_face_crop(gray_frame, x, y, w, h)
    if pad_face_crop is None or pad_face_crop.size == 0:
        return None

    if collect_pose_metrics:
        pose_metrics = compute_pose_metrics_from_image(pad_face_crop, face_mesh)
        update_pose_summary(pose_summary, pose_metrics)

    return extract_pad_features_from_final_crop(
        pad_face_crop,
        apply_quality_gate=apply_quality_gate,
        collect_pose_metrics=collect_pose_metrics,
        face_mesh=face_mesh,
        pose_summary=pose_summary,
    )


def extract_pad_features_from_final_crop(
    img,
    apply_quality_gate=True,
    collect_pose_metrics=False,
    face_mesh=None,
    pose_summary=None,
):
    """
    Extract PAD features from an already localized crop saved on disk.

    These images are saved custom PAD crops, not precomputed PAD
    features. The canonical post-crop PAD preprocessing path is still
    applied here before feature extraction.
    """
    if collect_pose_metrics:
        pose_metrics = compute_pose_metrics_from_image(img, face_mesh)
        update_pose_summary(pose_summary, pose_metrics)

    return extract_pad_features_from_crop(
        img,
        apply_quality_gate=apply_quality_gate,
        min_variance_laplacian=PAD_MIN_VARIANCE_LAPLACIAN,
    )


def extract_pad_features_from_dataset_image(
    img,
    detector=None,
    dataset_type="raw_frame",
    apply_quality_gate=True,
    collect_pose_metrics=False,
    face_mesh=None,
    pose_summary=None,
):
    """
    Dispatch PAD feature extraction according to the declared dataset
    image type.

    Supported dataset types:
    - ``raw_frame``: original image that still needs detection/cropping
    - ``final_pad_crop``: already localized PAD crop saved on disk
    """
    if dataset_type == "raw_frame":
        if detector is None:
            raise ValueError("A detector is required for dataset_type='raw_frame'.")
        return extract_pad_features_from_raw_frame(
            img,
            detector,
            apply_quality_gate=apply_quality_gate,
            collect_pose_metrics=collect_pose_metrics,
            face_mesh=face_mesh,
            pose_summary=pose_summary,
        )

    if dataset_type == "final_pad_crop":
        return extract_pad_features_from_final_crop(
            img,
            apply_quality_gate=apply_quality_gate,
            collect_pose_metrics=collect_pose_metrics,
            face_mesh=face_mesh,
            pose_summary=pose_summary,
        )

    raise ValueError(
        "Unsupported dataset_type: "
        f"{dataset_type}. Use 'raw_frame' or 'final_pad_crop'."
    )
