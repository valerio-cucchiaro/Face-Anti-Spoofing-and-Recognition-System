"""
Module: recognition_dataset_builders

Description:
Recognition dataset builders that turn gallery/probe folders into the
structures required by the training and evaluation pipeline.
"""

import time

import numpy as np

from src.preprocessing.pose_statistics import create_pose_summary
from src.recognition.recognition_quality_filter import (
    export_quality_records,
    export_quality_split_summary,
    export_quality_threshold_summary,
    format_quality_summary,
    normalize_quality_filter_mode,
)


IDENTITY_PROGRESS_STEP = 50


def load_gallery_dataset_impl(
    mode,
    namespace_identities,
    collect_pose_metrics,
    lfw_sample_ratio,
    lfw_sample_seed,
    quality_filter_mode,
    debug_quality,
    quality_output_stem,
    *,
    get_split_directories_fn,
    resolve_dataset_names_fn,
    build_identity_name_fn,
    normalize_identity_name_fn,
    load_images_from_identity_dir_fn,
    sample_image_paths_fn,
    detector_loader,
    face_mesh_loader,
):
    """
    Load one or more gallery datasets for recognition training.
    """
    faces = []
    labels = []
    label_map = {}
    dataset_counts = {}
    pose_summary = create_pose_summary() if collect_pose_metrics else None
    normalized_quality_filter_mode = normalize_quality_filter_mode(quality_filter_mode)
    quality_enabled = normalized_quality_filter_mode != "none"
    quality_records = [] if quality_enabled else None
    quality_debug_counts = {} if quality_enabled and debug_quality else None
    custom_detector = (
        detector_loader() if "custom" in resolve_dataset_names_fn(mode) else None
    )
    face_mesh = (
        face_mesh_loader(static_image_mode=True)
        if collect_pose_metrics
        else None
    )

    try:
        for dataset_name, _split_name, gallery_dir in get_split_directories_fn(mode, "gallery"):
            start_time = time.perf_counter()
            dataset_counts[dataset_name] = 0
            identity_count = 0
            person_dirs = sorted(
                [path for path in gallery_dir.iterdir() if path.is_dir()],
                key=lambda path: path.name,
            )

            print(f"Loading gallery dataset: {dataset_name}")
            print(f"Found {len(person_dirs)} gallery identities")

            for person_dir in person_dirs:
                identity_name = build_identity_name_fn(
                    dataset_name,
                    person_dir.name,
                    namespace_identities=namespace_identities,
                )
                normalized_identity_name = normalize_identity_name_fn(
                    identity_name,
                    preserve_namespace=namespace_identities,
                )

                identity_images = load_images_from_identity_dir_fn(
                    person_dir,
                    dataset_name=dataset_name,
                    split_name="gallery",
                    detector=custom_detector,
                    collect_pose_metrics=collect_pose_metrics,
                    pose_summary=pose_summary,
                    face_mesh=face_mesh,
                    sample_ratio=lfw_sample_ratio,
                    sample_seed=lfw_sample_seed,
                    quality_records=quality_records,
                    quality_debug_counts=quality_debug_counts,
                    quality_filter_mode=quality_filter_mode,
                    debug_quality=debug_quality,
                    quality_output_stem=quality_output_stem,
                    sample_image_paths_fn=sample_image_paths_fn,
                )

                if not identity_images:
                    continue

                current_label = len(label_map)
                label_map[current_label] = normalized_identity_name
                identity_count += 1

                if identity_count % IDENTITY_PROGRESS_STEP == 0:
                    print(f"Processed {identity_count}/{len(person_dirs)} gallery identities")

                for _, image in identity_images:
                    faces.append(image)
                    labels.append(current_label)
                    dataset_counts[dataset_name] += 1

            elapsed = time.perf_counter() - start_time
            print(
                f"Loaded {dataset_counts[dataset_name]} gallery images "
                f"from {identity_count} identities in {elapsed:.2f}s"
            )
            if dataset_name == "lfw" and lfw_sample_ratio < 1.0:
                print(
                    f"LFW gallery sampling: ratio={lfw_sample_ratio:.3f} "
                    f"seed={lfw_sample_seed}"
                )
    finally:
        if quality_enabled:
            split_summary_path = export_quality_split_summary(
                quality_records,
                quality_filter_mode=quality_filter_mode,
                output_stem=quality_output_stem,
            )
            report_path = (
                export_quality_records(
                    quality_records,
                    output_stem=quality_output_stem,
                )
                if debug_quality
                else None
            )
            threshold_summary_path = (
                export_quality_threshold_summary(output_stem=quality_output_stem)
                if debug_quality
                else None
            )
            print(format_quality_summary(quality_records, title="Gallery quality summary:"))
            if report_path is not None:
                print("Recognition quality report:", report_path)
            if split_summary_path is not None:
                print("Recognition quality split summary:", split_summary_path)
            if threshold_summary_path is not None:
                print("Recognition quality thresholds:", threshold_summary_path)
        if face_mesh is not None and hasattr(face_mesh, "close"):
            face_mesh.close()

    return faces, np.array(labels), label_map, dataset_counts, pose_summary


def load_probe_dataset_impl(
    mode,
    split,
    namespace_identities,
    collect_pose_metrics,
    lfw_sample_ratio,
    lfw_sample_seed,
    quality_filter_mode,
    debug_quality,
    quality_output_stem,
    *,
    get_split_directories_fn,
    resolve_dataset_names_fn,
    build_identity_name_fn,
    split_custom_open_set_identity_images_fn,
    assert_identity_subset_consistency_fn,
    load_images_from_identity_dir_fn,
    sample_image_paths_fn,
    detector_loader,
    face_mesh_loader,
):
    """
    Load one or more probe datasets for recognition evaluation.
    """
    probe_records = []
    dataset_counts = {}
    pose_summary = create_pose_summary() if collect_pose_metrics else None
    normalized_quality_filter_mode = normalize_quality_filter_mode(quality_filter_mode)
    quality_enabled = normalized_quality_filter_mode != "none"
    quality_records = [] if quality_enabled else None
    quality_debug_counts = {} if quality_enabled and debug_quality else None
    custom_detector = (
        detector_loader() if "custom" in resolve_dataset_names_fn(mode) else None
    )
    face_mesh = (
        face_mesh_loader(static_image_mode=True)
        if collect_pose_metrics
        else None
    )

    try:
        for dataset_name, split_name, probe_dir in get_split_directories_fn(mode, split):
            start_time = time.perf_counter()
            split_key = f"{dataset_name}/{split_name}"
            dataset_counts[split_key] = 0
            identity_count = 0
            person_dirs = sorted(
                [path for path in probe_dir.iterdir() if path.is_dir()],
                key=lambda path: path.name,
            )

            print(f"Loading recognition evaluation split: {dataset_name} ({split_name})")
            print(f"Found {len(person_dirs)} identities in split")

            for person_dir in person_dirs:
                identity_name = build_identity_name_fn(
                    dataset_name,
                    person_dir.name,
                    namespace_identities=namespace_identities,
                )

                identity_images = load_images_from_identity_dir_fn(
                    person_dir,
                    dataset_name=dataset_name,
                    split_name=split_name,
                    detector=custom_detector,
                    collect_pose_metrics=collect_pose_metrics,
                    pose_summary=pose_summary,
                    face_mesh=face_mesh,
                    sample_ratio=lfw_sample_ratio,
                    sample_seed=lfw_sample_seed,
                    quality_records=quality_records,
                    quality_debug_counts=quality_debug_counts,
                    quality_filter_mode=quality_filter_mode,
                    debug_quality=debug_quality,
                    quality_output_stem=quality_output_stem,
                    sample_image_paths_fn=sample_image_paths_fn,
                )
                if dataset_name == "custom":
                    original_count = len(identity_images)
                    identity_images = split_custom_open_set_identity_images_fn(
                        identity_images,
                        split_name,
                        person_dir.name,
                    )
                    assert_identity_subset_consistency_fn(
                        split_name,
                        person_dir.name,
                        original_count,
                        len(identity_images),
                    )
                identity_count += 1

                if identity_count % IDENTITY_PROGRESS_STEP == 0:
                    print(f"Processed {identity_count}/{len(person_dirs)} probe identities")

                for image_path, image in identity_images:
                    probe_records.append(
                        {
                            "dataset": dataset_name,
                            "split": split_name,
                            "identity": identity_name,
                            "path": image_path,
                            "image": image,
                        }
                    )
                    dataset_counts[split_key] += 1

            elapsed = time.perf_counter() - start_time
            print(
                f"Loaded {dataset_counts[split_key]} recognition images "
                f"from {identity_count} identities in {elapsed:.2f}s"
            )
            if dataset_name == "lfw" and lfw_sample_ratio < 1.0:
                print(
                    f"LFW probe sampling: ratio={lfw_sample_ratio:.3f} "
                    f"seed={lfw_sample_seed}"
                )
    finally:
        if quality_enabled:
            split_summary_path = export_quality_split_summary(
                quality_records,
                quality_filter_mode=quality_filter_mode,
                output_stem=quality_output_stem,
            )
            report_path = (
                export_quality_records(
                    quality_records,
                    output_stem=quality_output_stem,
                )
                if debug_quality
                else None
            )
            threshold_summary_path = (
                export_quality_threshold_summary(output_stem=quality_output_stem)
                if debug_quality
                else None
            )
            print(format_quality_summary(quality_records, title="Probe quality summary:"))
            if report_path is not None:
                print("Recognition quality report:", report_path)
            if split_summary_path is not None:
                print("Recognition quality split summary:", split_summary_path)
            if threshold_summary_path is not None:
                print("Recognition quality thresholds:", threshold_summary_path)
        if face_mesh is not None and hasattr(face_mesh, "close"):
            face_mesh.close()

    return probe_records, dataset_counts, pose_summary
