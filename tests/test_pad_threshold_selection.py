from src.evaluation.pad_metrics import select_pad_threshold_by_target_apcer


def test_select_pad_threshold_by_target_apcer_prefers_lowest_bpcer_under_constraint():
    sweep_results = {
        "per_threshold": [
            {"threshold": 0.2, "apcer": 0.05, "bpcer": 0.40, "acer": 0.225},
            {"threshold": 0.4, "apcer": 0.02, "bpcer": 0.20, "acer": 0.11},
            {"threshold": 0.6, "apcer": 0.02, "bpcer": 0.10, "acer": 0.06},
        ]
    }

    selection = select_pad_threshold_by_target_apcer(
        sweep_results,
        target_apcer=0.02,
    )

    assert selection["threshold"] == 0.6
    assert selection["threshold_rule"] == "target_apcer"
    assert selection["constraint_satisfied"] is True
    assert selection["selected_metrics"]["bpcer"] == 0.10


def test_select_pad_threshold_by_target_apcer_falls_back_to_min_apcer_when_needed():
    sweep_results = {
        "per_threshold": [
            {"threshold": 0.2, "apcer": 0.20, "bpcer": 0.05, "acer": 0.125},
            {"threshold": 0.4, "apcer": 0.10, "bpcer": 0.30, "acer": 0.20},
            {"threshold": 0.6, "apcer": 0.10, "bpcer": 0.20, "acer": 0.15},
        ]
    }

    selection = select_pad_threshold_by_target_apcer(
        sweep_results,
        target_apcer=0.02,
    )

    assert selection["threshold"] == 0.6
    assert selection["threshold_rule"] == "min_apcer_fallback"
    assert selection["constraint_satisfied"] is False
    assert selection["selected_metrics"]["apcer"] == 0.10
    assert selection["selected_metrics"]["bpcer"] == 0.20
