from src.recognition.recognition_dataset_loader import (
    normalize_identity_name,
    resolve_split_names,
)


def test_identity_normalization_strips_namespace_by_default():
    assert normalize_identity_name("ORL:S1") == "s1"


def test_identity_normalization_can_preserve_namespace_for_combined_modes():
    assert normalize_identity_name("ORL:S1", preserve_namespace=True) == "orl:s1"


def test_orl_validation_and_probe_aliases_resolve_to_known_and_unknown_splits():
    assert resolve_split_names("orl", "validation") == [
        "validation_known",
        "validation_unknown",
    ]
    assert resolve_split_names("orl", "probe") == [
        "probe_known",
        "probe_unknown",
    ]


def test_multipie_hr_lr_validation_and_probe_aliases_resolve_to_matching_splits():
    assert resolve_split_names("multipie", "validation_hr") == [
        "validation_known_hr",
        "validation_unknown_hr",
    ]
    assert resolve_split_names("multipie", "probe_lr") == [
        "probe_known_lr",
        "probe_unknown_lr",
    ]
