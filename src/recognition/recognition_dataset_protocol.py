"""
Module: recognition_dataset_protocol

Description:
Recognition-specific protocol helpers for identity naming and for the
deterministic custom internal validation/probe split logic.
"""

import numpy as np


CUSTOM_OPEN_SET_VALIDATION_RATIO = 0.4
CUSTOM_OPEN_SET_VALIDATION_SEED = 42


def _assert_identity_subset_consistency(
    split_name,
    identity_name,
    source_count,
    selected_count,
):
    """
    Validate one deterministic custom internal split per identity.
    """
    if source_count == 0 and selected_count != 0:
        raise RuntimeError(
            f"Custom recognition split '{split_name}' for identity '{identity_name}' "
            f"selected {selected_count} samples from an empty source set."
        )

    if source_count == 1 and selected_count != 1:
        raise RuntimeError(
            f"Custom recognition split '{split_name}' for identity '{identity_name}' "
            "must keep the single available sample."
        )

    if source_count > 1 and not 0 < selected_count < source_count:
        raise RuntimeError(
            f"Custom recognition split '{split_name}' for identity '{identity_name}' "
            f"produced an invalid subset size: selected={selected_count}, source={source_count}."
        )


def should_namespace_identities(mode):
    """
    Decide whether identity names should be prefixed with the dataset name.
    """
    return mode == "all"


def build_identity_name(dataset_name, person_name, namespace_identities=False):
    """
    Build the identity label used in the current experiment.
    """
    if namespace_identities:
        return f"{dataset_name}:{person_name}"

    return person_name


def normalize_identity_name(identity_name, preserve_namespace=False):
    """
    Normalize an identity label to its canonical comparison form.
    """
    if identity_name is None:
        return None

    normalized_identity_name = identity_name.strip().lower()
    if preserve_namespace:
        return normalized_identity_name

    return normalized_identity_name.split(":", 1)[-1]


def split_custom_open_set_identity_images(
    identity_images,
    split_name,
    identity_name,
    validation_ratio=CUSTOM_OPEN_SET_VALIDATION_RATIO,
    validation_seed=CUSTOM_OPEN_SET_VALIDATION_SEED,
):
    """
    Split one custom probe identity into deterministic validation and
    final-probe subsets after loading and quality filtering.
    """
    if split_name not in {
        "validation_known_internal",
        "probe_known_internal",
        "validation_unknown_internal",
        "probe_unknown_internal",
    }:
        return identity_images

    total_images = len(identity_images)
    if total_images == 0:
        return identity_images

    source_group = "unknown" if "unknown" in split_name else "known"
    identity_seed = validation_seed + sum(
        ord(character) for character in f"{source_group}:{identity_name}"
    )

    if total_images == 1:
        validation_indices = {0}
    else:
        validation_count = int(round(total_images * float(validation_ratio)))
        validation_count = max(1, min(total_images - 1, validation_count))
        rng = np.random.default_rng(identity_seed)
        validation_indices = set(
            int(index)
            for index in np.sort(
                rng.choice(total_images, size=validation_count, replace=False)
            )
        )

    if split_name.startswith("validation_"):
        return [
            image_record
            for index, image_record in enumerate(identity_images)
            if index in validation_indices
        ]

    return [
        image_record
        for index, image_record in enumerate(identity_images)
        if index not in validation_indices
    ]
