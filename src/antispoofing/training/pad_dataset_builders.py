"""
Module: pad_dataset_builders

Description:
PAD dataset builders that turn one or more dataset splits into feature
matrices and label arrays for training or evaluation.
"""

import time
from pathlib import Path

import cv2
import numpy as np

from src.detection.face_detector import load_mediapipe_detector
from src.detection.face_landmarks import load_mediapipe_face_mesh
from src.preprocessing.pose_statistics import (
    create_pose_summary,
    merge_pose_summaries,
)
from src.antispoofing.training.pad_dataset_preprocessing import (
    PAD_IMAGE_PROGRESS_STEP,
    LIVE_PHOTOMETRIC_AUGMENTATION_MODE,
    LIVE_PHOTOMETRIC_AUGMENTATION_SEED,
    apply_live_photometric_augmentation,
    extract_pad_features_from_dataset_image,
)
from src.antispoofing.training.pad_dataset_splits import (
    CASIA_SAMPLE_RATIO,
    CUSTOM_PAD_VALIDATION_RATIO,
    CUSTOM_PAD_VALIDATION_SEED,
    CUSTOM_TEST_LIVE_PATH,
    CUSTOM_TEST_SPOOF_PATH,
    CUSTOM_TRAIN_LIVE_PATH,
    CUSTOM_TRAIN_SPOOF_PATH,
    REPLAY_SAMPLE_RATIO,
    REPLAY_TEST_LIVE_PATH,
    REPLAY_TEST_SPOOF_PATH,
    REPLAY_TRAIN_LIVE_PATH,
    REPLAY_TRAIN_SPOOF_PATH,
    REPLAY_VALIDATION_LIVE_PATH,
    REPLAY_VALIDATION_SPOOF_PATH,
    TEST_LIVE_PATH,
    TEST_SPOOF_PATH,
    TRAIN_LIVE_PATH,
    TRAIN_SAMPLING_SEED,
    TRAIN_SPOOF_PATH,
    VALIDATION_LIVE_PATH,
    VALIDATION_SPOOF_PATH,
    collect_image_paths,
    resolve_custom_pad_train_validation_paths,
    sample_image_paths,
)


def load_combined_pad_training_dataset(
    casia_sample_ratio=CASIA_SAMPLE_RATIO,
    replay_sample_ratio=REPLAY_SAMPLE_RATIO,
    sample_seed=TRAIN_SAMPLING_SEED,
    casia_balance=False,
    replay_balance=True,
    collect_pose_metrics=False,
    return_pose_summary=False,
    augment_live_photometric=False,
    augment_live_photometric_mode=None,
    augment_live_photometric_scope="all",
):
    """
    Load the combined PAD training dataset (CASIA + Replay-Attack).
    """
    print("Loading CASIA PAD dataset...")
    X1, y1, pose_summary_1 = load_pad_split_dataset(
        TRAIN_LIVE_PATH,
        TRAIN_SPOOF_PATH,
        balance=casia_balance,
        live_ratio=casia_sample_ratio,
        spoof_ratio=casia_sample_ratio,
        sample_seed=sample_seed,
        dataset_type="raw_frame",
        augment_live_photometric=(
            augment_live_photometric and augment_live_photometric_scope == "all"
        ),
        augment_live_photometric_mode=augment_live_photometric_mode,
        collect_pose_metrics=collect_pose_metrics,
        return_pose_summary=True,
    )

    print("Loading ReplayAttack PAD dataset...")
    X2, y2, pose_summary_2 = load_pad_split_dataset(
        REPLAY_TRAIN_LIVE_PATH,
        REPLAY_TRAIN_SPOOF_PATH,
        balance=replay_balance,
        live_ratio=replay_sample_ratio,
        spoof_ratio=replay_sample_ratio,
        sample_seed=sample_seed,
        dataset_type="raw_frame",
        augment_live_photometric=(
            augment_live_photometric and augment_live_photometric_scope == "all"
        ),
        augment_live_photometric_mode=augment_live_photometric_mode,
        collect_pose_metrics=collect_pose_metrics,
        return_pose_summary=True,
    )

    X = np.concatenate([X1, X2], axis=0)
    y = np.concatenate([y1, y2], axis=0)
    pose_summary = merge_pose_summaries(pose_summary_1, pose_summary_2)

    if return_pose_summary:
        return X, y, pose_summary

    return X, y


def load_casia_pad_training_dataset(
    sample_ratio=CASIA_SAMPLE_RATIO,
    sample_seed=TRAIN_SAMPLING_SEED,
    balance=False,
    collect_pose_metrics=False,
    return_pose_summary=False,
    augment_live_photometric=False,
    augment_live_photometric_mode=None,
):
    """
    Load the CASIA PAD training dataset with deterministic sampling.
    """
    return load_pad_split_dataset(
        TRAIN_LIVE_PATH,
        TRAIN_SPOOF_PATH,
        balance=balance,
        live_ratio=sample_ratio,
        spoof_ratio=sample_ratio,
        sample_seed=sample_seed,
        dataset_type="raw_frame",
        augment_live_photometric=augment_live_photometric,
        augment_live_photometric_mode=augment_live_photometric_mode,
        collect_pose_metrics=collect_pose_metrics,
        return_pose_summary=return_pose_summary,
    )


def load_replay_pad_training_dataset(
    sample_ratio=REPLAY_SAMPLE_RATIO,
    sample_seed=TRAIN_SAMPLING_SEED,
    balance=True,
    collect_pose_metrics=False,
    return_pose_summary=False,
    augment_live_photometric=False,
    augment_live_photometric_mode=None,
):
    """
    Load the Replay-Attack PAD training dataset only.
    """
    return load_pad_split_dataset(
        REPLAY_TRAIN_LIVE_PATH,
        REPLAY_TRAIN_SPOOF_PATH,
        balance=balance,
        live_ratio=sample_ratio,
        spoof_ratio=sample_ratio,
        sample_seed=sample_seed,
        dataset_type="raw_frame",
        augment_live_photometric=augment_live_photometric,
        augment_live_photometric_mode=augment_live_photometric_mode,
        collect_pose_metrics=collect_pose_metrics,
        return_pose_summary=return_pose_summary,
    )


def load_custom_pad_training_dataset(
    collect_pose_metrics=False,
    return_pose_summary=False,
    augment_live_photometric=False,
    augment_live_photometric_mode=None,
):
    """
    Load the full custom PAD training split from pad_custom_raw.
    """
    return load_pad_split_dataset(
        CUSTOM_TRAIN_LIVE_PATH,
        CUSTOM_TRAIN_SPOOF_PATH,
        balance=False,
        dataset_type="final_pad_crop",
        augment_live_photometric=augment_live_photometric,
        augment_live_photometric_mode=augment_live_photometric_mode,
        collect_pose_metrics=collect_pose_metrics,
        return_pose_summary=return_pose_summary,
    )


def load_custom_pad_test_dataset(
    return_image_paths=False,
    collect_pose_metrics=False,
    return_pose_summary=False,
):
    """
    Load the custom PAD test dataset from pad_custom_raw.
    """
    return load_pad_split_dataset(
        CUSTOM_TEST_LIVE_PATH,
        CUSTOM_TEST_SPOOF_PATH,
        dataset_type="final_pad_crop",
        return_image_paths=return_image_paths,
        collect_pose_metrics=collect_pose_metrics,
        return_pose_summary=return_pose_summary,
    )


def load_pad_test_dataset_casia(
    collect_pose_metrics=False,
    return_pose_summary=False,
):
    """
    Load the CASIA PAD test dataset and extract feature vectors.
    """
    return load_pad_split_dataset(
        TEST_LIVE_PATH,
        TEST_SPOOF_PATH,
        dataset_type="raw_frame",
        collect_pose_metrics=collect_pose_metrics,
        return_pose_summary=return_pose_summary,
    )


def load_pad_validation_dataset_casia(
    collect_pose_metrics=False,
    return_pose_summary=False,
):
    """
    Load the CASIA PAD validation dataset.
    """
    return load_pad_split_dataset(
        VALIDATION_LIVE_PATH,
        VALIDATION_SPOOF_PATH,
        dataset_type="raw_frame",
        collect_pose_metrics=collect_pose_metrics,
        return_pose_summary=return_pose_summary,
    )


def load_custom_pad_training_subset(
    validation_ratio=CUSTOM_PAD_VALIDATION_RATIO,
    validation_seed=CUSTOM_PAD_VALIDATION_SEED,
    collect_pose_metrics=False,
    return_pose_summary=False,
    augment_live_photometric=False,
    augment_live_photometric_mode=None,
):
    """
    Load only the effective custom PAD training subset after removing
    the deterministic pseudo-validation samples from the train split.
    """
    split_paths = resolve_custom_pad_train_validation_paths(
        validation_ratio=validation_ratio,
        validation_seed=validation_seed,
    )
    return load_pad_split_dataset(
        split_paths["train_live_paths"],
        split_paths["train_spoof_paths"],
        balance=False,
        dataset_type="final_pad_crop",
        augment_live_photometric=augment_live_photometric,
        augment_live_photometric_mode=augment_live_photometric_mode,
        collect_pose_metrics=collect_pose_metrics,
        return_pose_summary=return_pose_summary,
    )


def load_custom_pad_validation_dataset(
    validation_ratio=CUSTOM_PAD_VALIDATION_RATIO,
    validation_seed=CUSTOM_PAD_VALIDATION_SEED,
    return_image_paths=False,
    collect_pose_metrics=False,
    return_pose_summary=False,
):
    """
    Load the deterministic pseudo-validation subset carved out of the
    custom PAD train split.
    """
    split_paths = resolve_custom_pad_train_validation_paths(
        validation_ratio=validation_ratio,
        validation_seed=validation_seed,
    )
    return load_pad_split_dataset(
        split_paths["validation_live_paths"],
        split_paths["validation_spoof_paths"],
        balance=False,
        dataset_type="final_pad_crop",
        return_image_paths=return_image_paths,
        collect_pose_metrics=collect_pose_metrics,
        return_pose_summary=return_pose_summary,
    )


def load_pad_test_dataset_replay(
    collect_pose_metrics=False,
    return_pose_summary=False,
):
    """
    Load the Replay-Attack PAD test dataset and extract feature vectors.
    """
    return load_pad_split_dataset(
        REPLAY_TEST_LIVE_PATH,
        REPLAY_TEST_SPOOF_PATH,
        balance=False,
        dataset_type="raw_frame",
        collect_pose_metrics=collect_pose_metrics,
        return_pose_summary=return_pose_summary,
    )


def load_pad_validation_dataset_replay(
    collect_pose_metrics=False,
    return_pose_summary=False,
):
    """
    Load the Replay-Attack validation dataset extracted from raw/devel.
    """
    return load_pad_split_dataset(
        REPLAY_VALIDATION_LIVE_PATH,
        REPLAY_VALIDATION_SPOOF_PATH,
        balance=False,
        dataset_type="raw_frame",
        collect_pose_metrics=collect_pose_metrics,
        return_pose_summary=return_pose_summary,
    )


def load_combined_with_custom_pad_training_dataset(
    custom_repeat=5,
    casia_sample_ratio=CASIA_SAMPLE_RATIO,
    replay_sample_ratio=REPLAY_SAMPLE_RATIO,
    sample_seed=TRAIN_SAMPLING_SEED,
    casia_balance=False,
    replay_balance=True,
    collect_pose_metrics=False,
    return_pose_summary=False,
    augment_live_photometric=False,
    augment_live_photometric_mode=None,
    augment_live_photometric_scope="all",
):
    """
    Load the combined PAD training dataset and oversample the custom
    training split so it has a stronger influence during training.
    """
    print("Loading CASIA PAD dataset...")
    X1, y1, pose_summary_1 = load_pad_split_dataset(
        TRAIN_LIVE_PATH,
        TRAIN_SPOOF_PATH,
        balance=casia_balance,
        live_ratio=casia_sample_ratio,
        spoof_ratio=casia_sample_ratio,
        sample_seed=sample_seed,
        dataset_type="raw_frame",
        augment_live_photometric=(
            augment_live_photometric and augment_live_photometric_scope == "all"
        ),
        augment_live_photometric_mode=augment_live_photometric_mode,
        collect_pose_metrics=collect_pose_metrics,
        return_pose_summary=True,
    )

    print("Loading ReplayAttack PAD dataset...")
    X2, y2, pose_summary_2 = load_pad_split_dataset(
        REPLAY_TRAIN_LIVE_PATH,
        REPLAY_TRAIN_SPOOF_PATH,
        balance=replay_balance,
        live_ratio=replay_sample_ratio,
        spoof_ratio=replay_sample_ratio,
        sample_seed=sample_seed,
        dataset_type="raw_frame",
        augment_live_photometric=(
            augment_live_photometric and augment_live_photometric_scope == "all"
        ),
        augment_live_photometric_mode=augment_live_photometric_mode,
        collect_pose_metrics=collect_pose_metrics,
        return_pose_summary=True,
    )

    print("Loading custom PAD dataset...")
    X3, y3, pose_summary_3 = load_custom_pad_training_dataset(
        augment_live_photometric=augment_live_photometric,
        augment_live_photometric_mode=augment_live_photometric_mode,
        collect_pose_metrics=collect_pose_metrics,
        return_pose_summary=True,
    )
    custom_base_count = len(y3)

    if len(y3) > 0 and custom_repeat > 1:
        X3 = np.concatenate([X3] * custom_repeat, axis=0)
        y3 = np.concatenate([y3] * custom_repeat, axis=0)

    X = np.concatenate([X1, X2, X3], axis=0)
    y = np.concatenate([y1, y2, y3], axis=0)

    casia_count = len(y1)
    replay_count = len(y2)
    custom_effective_count = len(y3)
    total_count = len(y)

    casia_weight = (casia_count / total_count * 100) if total_count else 0.0
    replay_weight = (replay_count / total_count * 100) if total_count else 0.0
    custom_weight = (custom_effective_count / total_count * 100) if total_count else 0.0

    print("Number of samples effectively used for training")
    print("Training sampling seed:", sample_seed)
    print(f"CASIA sample ratio: {casia_sample_ratio:.2f}")
    print(f"Replay sample ratio: {replay_sample_ratio:.2f}")
    print("CASIA pre-extraction balance:", casia_balance)
    print("Replay pre-extraction balance:", replay_balance)
    print("CASIA train:", casia_count)
    print("Replay-Attack train:", replay_count)
    print("Custom train (base):", custom_base_count)
    print("Custom repeat:", custom_repeat)
    print("Custom train (effective):", custom_effective_count)
    print("Final training set:", total_count)
    print(f"CASIA weight: {casia_weight:.2f}%")
    print(f"Replay-Attack weight: {replay_weight:.2f}%")
    print(f"Custom weight: {custom_weight:.2f}%")

    pose_summary = merge_pose_summaries(pose_summary_1, pose_summary_2)
    pose_summary = merge_pose_summaries(pose_summary, pose_summary_3)

    if return_pose_summary:
        return X, y, pose_summary

    return X, y


def load_pad_split_dataset(
    live_path,
    spoof_path,
    balance=False,
    live_ratio=1.0,
    spoof_ratio=1.0,
    sample_seed=None,
    dataset_type="raw_frame",
    apply_quality_gate=True,
    return_image_paths=False,
    collect_pose_metrics=False,
    return_pose_summary=False,
    augment_live_photometric=False,
    augment_live_photometric_mode=None,
):
    """
    Load one supported PAD dataset split from directories or preselected
    image path lists.

    This function reads live and spoof images from the provided sources,
    extracts PAD feature vectors, and assigns binary labels.

    If balance=True, the larger class is downsampled before feature
    extraction to match the smaller one. After detection, quality
    filtering, and feature extraction the final valid sample counts
    can still become unbalanced.

    The shared quality gate applies only to structural crop validity in
    the loading path selected by ``apply_quality_gate``; it does not
    duplicate the runtime history gate, which also requires sharpness and
    bounding-box stability before updating temporal smoothing.
    """
    features = []
    labels = []
    sample_paths = []
    pose_summary = create_pose_summary() if collect_pose_metrics else None
    start_time = time.perf_counter()
    augmentation_seed = (
        sample_seed if sample_seed is not None else LIVE_PHOTOMETRIC_AUGMENTATION_SEED
    )
    augmentation_rng = np.random.default_rng(augmentation_seed)
    augmented_live = 0
    augmentation_mode_counts = {}

    detector = None
    if dataset_type == "raw_frame":
        detector = load_mediapipe_detector()
    face_mesh = load_mediapipe_face_mesh() if collect_pose_metrics else None

    live_paths = collect_image_paths(live_path)
    spoof_paths = collect_image_paths(spoof_path)

    live_paths = sample_image_paths(
        live_paths,
        ratio=live_ratio,
        seed=sample_seed,
    )
    spoof_paths = sample_image_paths(
        spoof_paths,
        ratio=spoof_ratio,
        seed=None if sample_seed is None else sample_seed + 1,
    )

    if balance:
        n_live = len(live_paths)
        n_spoof = len(spoof_paths)
        n_target = min(n_live, n_spoof)
        rng = np.random.default_rng(sample_seed)

        if n_live > n_target:
            indices = rng.choice(len(live_paths), size=n_target, replace=False)
            indices = np.sort(indices)
            live_paths = [live_paths[i] for i in indices]

        if n_spoof > n_target:
            indices = rng.choice(len(spoof_paths), size=n_target, replace=False)
            indices = np.sort(indices)
            spoof_paths = [spoof_paths[i] for i in indices]

    live_source = (
        live_path
        if isinstance(live_path, (str, Path))
        else f"selected_list[{len(live_paths)}]"
    )
    spoof_source = (
        spoof_path
        if isinstance(spoof_path, (str, Path))
        else f"selected_list[{len(spoof_paths)}]"
    )

    print(f"Loading PAD split: live={live_source} spoof={spoof_source}")
    print(f"Dataset type: {dataset_type}")
    print(f"Pre-extraction balance enabled: {balance}")
    print(f"Live photometric augmentation enabled: {augment_live_photometric}")
    if augment_live_photometric:
        print(
            "Live photometric augmentation mode:",
            augment_live_photometric_mode or LIVE_PHOTOMETRIC_AUGMENTATION_MODE,
        )
    print(f"Selected live images: {len(live_paths)}")
    print(f"Selected spoof images: {len(spoof_paths)}")

    valid_live = 0
    for index, img_path in enumerate(live_paths, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_features = extract_pad_features_from_dataset_image(
            img,
            detector=detector,
            dataset_type=dataset_type,
            apply_quality_gate=apply_quality_gate,
            collect_pose_metrics=collect_pose_metrics,
            face_mesh=face_mesh,
            pose_summary=pose_summary,
        )
        if img_features is None:
            continue

        features.append(img_features)
        labels.append(1)
        if return_image_paths:
            sample_paths.append(str(img_path))
        valid_live += 1

        if augment_live_photometric:
            gray_img = img
            if img.ndim == 3:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            augmented_img, augmentation_name = apply_live_photometric_augmentation(
                gray_img,
                augmentation_rng,
                mode=augment_live_photometric_mode,
            )
            if dataset_type == "raw_frame" and augmented_img.ndim == 2:
                augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_GRAY2BGR)
            augmented_features = extract_pad_features_from_dataset_image(
                augmented_img,
                detector=detector,
                dataset_type=dataset_type,
                apply_quality_gate=apply_quality_gate,
                collect_pose_metrics=collect_pose_metrics,
                face_mesh=face_mesh,
                pose_summary=pose_summary,
            )
            if augmented_features is not None:
                features.append(augmented_features)
                labels.append(1)
                if return_image_paths:
                    sample_paths.append(f"{img_path} | augmented:{augmentation_name}")
                augmented_live += 1
                augmentation_mode_counts[augmentation_name] = (
                    augmentation_mode_counts.get(augmentation_name, 0) + 1
                )

        if index % PAD_IMAGE_PROGRESS_STEP == 0:
            print(f"Processed live images: {index}/{len(live_paths)}")

    valid_spoof = 0
    for index, img_path in enumerate(spoof_paths, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_features = extract_pad_features_from_dataset_image(
            img,
            detector=detector,
            dataset_type=dataset_type,
            apply_quality_gate=apply_quality_gate,
            collect_pose_metrics=collect_pose_metrics,
            face_mesh=face_mesh,
            pose_summary=pose_summary,
        )
        if img_features is None:
            continue

        features.append(img_features)
        labels.append(0)
        if return_image_paths:
            sample_paths.append(str(img_path))
        valid_spoof += 1

        if index % PAD_IMAGE_PROGRESS_STEP == 0:
            print(f"Processed spoof images: {index}/{len(spoof_paths)}")

    elapsed = time.perf_counter() - start_time
    print(f"Valid live samples extracted: {valid_live}")
    if augment_live_photometric:
        print(f"Valid augmented live samples extracted: {augmented_live}")
        if augmentation_mode_counts:
            print("Augmentation mode counts:", augmentation_mode_counts)
    print(f"Valid spoof samples extracted: {valid_spoof}")
    if valid_live + valid_spoof > 0:
        live_ratio = valid_live / (valid_live + valid_spoof) * 100
        spoof_ratio = valid_spoof / (valid_live + valid_spoof) * 100
        print(f"Final valid live weight: {live_ratio:.2f}%")
        print(f"Final valid spoof weight: {spoof_ratio:.2f}%")
    print(f"PAD split loading completed in {elapsed:.2f}s")

    if face_mesh is not None and hasattr(face_mesh, "close"):
        face_mesh.close()

    features_array = np.array(features, dtype=np.float32)
    labels_array = np.array(labels, dtype=np.int32)

    if return_image_paths:
        if return_pose_summary:
            return features_array, labels_array, sample_paths, pose_summary
        return features_array, labels_array, sample_paths

    if return_pose_summary:
        return features_array, labels_array, pose_summary

    return features_array, labels_array
