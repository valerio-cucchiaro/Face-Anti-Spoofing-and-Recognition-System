"""
Module: recognition_dataset_preprocessing

Description:
Sample-level loading and preprocessing helpers shared by recognition
gallery and probe dataset builders.
"""

import cv2

from src.preprocessing.face_pose import compute_pose_metrics_from_image
from src.preprocessing.frame_preprocessing import get_valid_face
from src.preprocessing.pose_statistics import update_pose_summary
from src.preprocessing.face_preprocessing import (
    preprocess_face_for_recognition,
    preprocess_loaded_face_for_recognition,
)
from src.recognition.recognition_dataset_splits import (
    CUSTOM_RAW_MIN_FACE_AREA,
    VALID_IMAGE_EXTENSIONS,
)
from src.recognition.recognition_quality_filter import (
    assess_recognition_sample_quality,
    build_quality_record,
    export_quality_debug_record,
    normalize_quality_filter_mode,
    save_quality_debug_image,
    should_reject_quality_record,
)


def _compute_pose_metrics_for_loaded_image(
    dataset_name,
    image,
    face_mesh,
    face_rect=None,
):
    """
    Compute pose metrics for one loaded recognition sample.
    """
    if image is None or face_mesh is None:
        return None

    if dataset_name == "custom" and face_rect is not None:
        x, y, w, h = face_rect
        sample_image = image[y:y + h, x:x + w]
    else:
        sample_image = image

    return compute_pose_metrics_from_image(sample_image, face_mesh)


def load_images_from_identity_dir(
    identity_dir,
    dataset_name,
    split_name="gallery",
    detector=None,
    collect_pose_metrics=False,
    pose_summary=None,
    face_mesh=None,
    sample_ratio=1.0,
    sample_seed=None,
    quality_records=None,
    quality_debug_counts=None,
    quality_filter_mode="report_only",
    debug_quality=False,
    quality_output_stem=None,
    sample_image_paths_fn=None,
):
    """
    Load all valid images from one identity directory.
    """
    candidates = []
    normalized_quality_filter_mode = normalize_quality_filter_mode(quality_filter_mode)
    quality_enabled = normalized_quality_filter_mode != "none"
    image_paths = [
        image_path
        for image_path in sorted(identity_dir.iterdir(), key=lambda path: path.name)
        if image_path.is_file() and image_path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]

    if dataset_name == "lfw" and sample_image_paths_fn is not None:
        image_paths = sample_image_paths_fn(
            image_paths,
            ratio=sample_ratio,
            seed=sample_seed,
        )

    for image_index, image_path in enumerate(image_paths, start=1):
        if dataset_name == "custom":
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue

        if dataset_name == "custom":
            if detector is None:
                raise ValueError("A face detector is required to load custom raw images.")

            face_rect, gray_frame, message = get_valid_face(
                image,
                detector,
                CUSTOM_RAW_MIN_FACE_AREA,
            )
            if message != "Face detected" or face_rect is None or gray_frame is None:
                continue

            x, y, w, h = face_rect
            preprocessed_image = preprocess_face_for_recognition(
                gray_frame,
                x,
                y,
                w,
                h,
            )
            pose_face_rect = face_rect
        else:
            preprocessed_image = preprocess_loaded_face_for_recognition(image)
            pose_face_rect = None

        pose_metrics = None
        if collect_pose_metrics:
            pose_metrics = _compute_pose_metrics_for_loaded_image(
                dataset_name,
                image,
                face_mesh,
                face_rect=pose_face_rect,
            )
            update_pose_summary(pose_summary, pose_metrics)

        quality_record = None
        if quality_enabled:
            quality_result = assess_recognition_sample_quality(image, pose_metrics)
            quality_record = build_quality_record(
                dataset_name,
                split_name,
                identity_dir.name,
                image_index,
                image_path,
                quality_result,
            )
            if quality_records is not None:
                quality_records.append(quality_record)
            if debug_quality and quality_debug_counts is not None:
                debug_image_path = save_quality_debug_image(
                    quality_record,
                    image,
                    quality_debug_counts,
                )
                export_quality_debug_record(
                    quality_record,
                    debug_image_path,
                    output_stem=quality_output_stem,
                )

        if preprocessed_image is None:
            continue

        candidates.append(
            {
                "path": image_path,
                "image": preprocessed_image,
                "quality_record": quality_record,
            }
        )

    if quality_enabled:
        kept_candidates = [
            candidate
            for candidate in candidates
            if not should_reject_quality_record(
                candidate["quality_record"],
                mode=normalized_quality_filter_mode,
                split_name=split_name,
            )
        ]
    else:
        kept_candidates = candidates

    return [
        (candidate["path"], candidate["image"])
        for candidate in kept_candidates
    ]
