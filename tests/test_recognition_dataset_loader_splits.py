from src.recognition.recognition_dataset_loader import resolve_split_names


def test_orl_validation_and_test_aliases_resolve_to_known_and_unknown_splits():
    assert resolve_split_names("orl", "validation") == [
        "validation_known",
        "validation_unknown",
    ]
    assert resolve_split_names("orl", "test") == [
        "test_known",
        "test_unknown",
    ]


def test_multipie_hr_lr_validation_and_test_aliases_resolve_to_matching_splits():
    assert resolve_split_names("multipie", "validation_hr") == [
        "validation_known_hr",
        "validation_unknown_hr",
    ]
    assert resolve_split_names("multipie", "test_lr") == [
        "test_known_lr",
        "test_unknown_lr",
    ]
