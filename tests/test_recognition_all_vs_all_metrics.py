import numpy as np

from src.evaluation.recognition_metrics import (
    compute_all_gallery_vs_all_probe_analysis,
    build_doddington_zoo_analysis,
    build_system_response_reliability_analysis,
)


def test_compute_all_gallery_vs_all_probe_analysis_builds_rank_and_open_set_summary():
    gallery_histograms = [
        np.array([1.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
    ]
    gallery_labels = np.array([0, 1], dtype=np.int64)
    label_map = {
        0: "alice",
        1: "bob",
    }
    probe_records = [
        {
            "true_identity": "alice",
            "split": "probe_known",
            "is_known": True,
        },
        {
            "true_identity": "carol",
            "split": "probe_unknown",
            "is_known": False,
        },
    ]
    probe_histograms = [
        np.array([1.0, 0.0], dtype=np.float64),
        np.array([0.5, 0.5], dtype=np.float64),
    ]

    analysis = compute_all_gallery_vs_all_probe_analysis(
        probe_records,
        probe_histograms,
        gallery_histograms,
        gallery_labels,
        label_map,
        threshold=0.5,
    )

    assert analysis["available"] is True
    assert analysis["protocol"] == "all gallery vs all probe"
    assert analysis["gallery_template_count"] == 2
    assert analysis["gallery_identity_count"] == 2
    assert analysis["rank_summary"]["rank_1_accuracy"] == 1.0
    assert analysis["rank_summary"]["rank_3_accuracy"] == 1.0
    assert analysis["cms_curve"]["available"] is True
    assert analysis["cms_curve"]["points"][0]["rank"] == 1
    assert analysis["cms_curve"]["points"][0]["cms"] == 1.0
    assert analysis["open_set_summary"]["correct_identification"] == 1
    assert analysis["open_set_summary"]["false_accept"] == 0
    assert analysis["open_set_summary"]["true_reject"] == 1
    assert len(analysis["preview_rows"]) >= 1
    assert analysis["preview_rows"][0]["example_type"] in {
        "best_rank",
        "median_rank",
        "hardest_rank",
        "lowest_margin",
        "highest_margin",
        "unknown_best_distance",
        "unknown_low_margin",
    }

    first_row = analysis["rows"][0]
    second_row = analysis["rows"][1]

    assert first_row["best_identity"] == "alice"
    assert first_row["true_identity_rank"] == 1
    assert first_row["final_identity"] == "alice"
    assert first_row["decision_correct"] is True
    assert first_row["distance__alice"] == 0.0
    assert first_row["distance__bob"] > 0.0

    assert second_row["best_identity"] == "alice"
    assert second_row["final_identity"] == "Unknown"
    assert second_row["decision_correct"] is True
    assert second_row["true_identity_rank"] is None


def test_compute_all_gallery_vs_all_probe_analysis_skips_large_matrices():
    gallery_histograms = [
        np.array([1.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
    ]
    gallery_labels = np.array([0, 1], dtype=np.int64)
    label_map = {
        0: "alice",
        1: "bob",
    }
    probe_records = [
        {
            "true_identity": "alice",
            "split": "probe_known",
            "is_known": True,
        },
        {
            "true_identity": "bob",
            "split": "probe_known",
            "is_known": True,
        },
    ]
    probe_histograms = [
        np.array([1.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
    ]

    analysis = compute_all_gallery_vs_all_probe_analysis(
        probe_records,
        probe_histograms,
        gallery_histograms,
        gallery_labels,
        label_map,
        threshold=0.5,
        max_total_comparisons=1,
    )

    assert analysis["available"] is False
    assert analysis["reason"] == "matrix_too_large"


def test_compute_all_gallery_vs_all_probe_analysis_treats_test_unknown_as_unknown():
    gallery_histograms = [
        np.array([1.0, 0.0], dtype=np.float64),
    ]
    gallery_labels = np.array([0], dtype=np.int64)
    label_map = {
        0: "alice",
    }
    probe_records = [
        {
            "true_identity": "unknown_probe",
            "split": "test_unknown",
        },
    ]
    probe_histograms = [
        np.array([0.5, 0.5], dtype=np.float64),
    ]

    analysis = compute_all_gallery_vs_all_probe_analysis(
        probe_records,
        probe_histograms,
        gallery_histograms,
        gallery_labels,
        label_map,
        threshold=0.1,
    )

    assert analysis["rows"][0]["is_known"] is False
    assert analysis["open_set_summary"]["true_reject"] == 1


def test_build_doddington_zoo_analysis_groups_expected_identity_roles():
    pairwise_analysis = {
        "available": True,
        "threshold": 0.5,
        "identity_columns": ["alice", "bob", "carol"],
        "rows": [
            {
                "true_identity": "alice",
                "is_known": True,
                "true_identity_rank": 1,
                "best_identity": "alice",
                "accepted_at_threshold": True,
                "final_identity": "alice",
            },
            {
                "true_identity": "bob",
                "is_known": True,
                "true_identity_rank": 3,
                "best_identity": "carol",
                "accepted_at_threshold": False,
                "final_identity": "Unknown",
            },
            {
                "true_identity": "carol",
                "is_known": True,
                "true_identity_rank": 1,
                "best_identity": "carol",
                "accepted_at_threshold": True,
                "final_identity": "carol",
            },
            {
                "true_identity": "unknown_probe",
                "is_known": False,
                "true_identity_rank": None,
                "best_identity": "alice",
                "accepted_at_threshold": True,
                "final_identity": "alice",
            },
            {
                "true_identity": "alice",
                "is_known": True,
                "true_identity_rank": 2,
                "best_identity": "bob",
                "accepted_at_threshold": True,
                "final_identity": "bob",
            },
        ],
    }

    analysis = build_doddington_zoo_analysis(pairwise_analysis, threshold=0.5)

    assert analysis["available"] is True
    assert analysis["summary_counts"]["identity_count"] == 3
    assert sum(category["count"] for category in analysis["categories"].values()) == 3

    per_identity = {row["identity"]: row for row in analysis["per_identity"]}
    assert per_identity["alice"]["dominant_tag"] in {"sheep", "goat", "lamb", "wolf"}
    assert per_identity["alice"]["second_tag"] in {"sheep", "goat", "lamb", "wolf"}
    assert per_identity["alice"]["dominant_tag"] != per_identity["alice"]["second_tag"]
    assert per_identity["alice"]["score_gap"] >= 0.0
    assert per_identity["bob"]["dominant_tag"] in {"sheep", "goat", "lamb", "wolf"}
    assert per_identity["bob"]["score_gap"] >= 0.0


def test_build_system_response_reliability_analysis_builds_scores_and_threshold():
    pairwise_analysis = {
        "available": True,
        "threshold": 0.5,
        "identity_columns": ["alice", "bob"],
        "rows": [
            {
                "probe_index": 1,
                "split": "probe_known",
                "true_identity": "alice",
                "is_known": True,
                "best_identity": "alice",
                "best_distance": 0.10,
                "second_best_identity": "bob",
                "second_best_distance": 0.90,
                "margin": 0.80,
                "true_identity_rank": 1,
                "accepted_at_threshold": True,
                "final_identity": "alice",
                "decision_correct": True,
                "distance__alice": 0.10,
                "distance__bob": 0.90,
            },
            {
                "probe_index": 2,
                "split": "probe_known",
                "true_identity": "bob",
                "is_known": True,
                "best_identity": "alice",
                "best_distance": 0.12,
                "second_best_identity": "bob",
                "second_best_distance": 0.13,
                "margin": 0.01,
                "true_identity_rank": 2,
                "accepted_at_threshold": True,
                "final_identity": "alice",
                "decision_correct": False,
                "distance__alice": 0.12,
                "distance__bob": 0.13,
            },
            {
                "probe_index": 3,
                "split": "probe_unknown",
                "true_identity": "unknown_probe",
                "is_known": False,
                "best_identity": "alice",
                "best_distance": 0.80,
                "second_best_identity": "bob",
                "second_best_distance": 0.95,
                "margin": 0.15,
                "true_identity_rank": None,
                "accepted_at_threshold": False,
                "final_identity": "Unknown",
                "decision_correct": True,
                "distance__alice": 0.80,
                "distance__bob": 0.95,
            },
        ],
    }

    analysis = build_system_response_reliability_analysis(
        pairwise_analysis,
        threshold=0.5,
    )

    assert analysis["available"] is True
    assert analysis["selected_variant"] == "adaptive_density_ratio"
    assert analysis["selected_threshold"] is not None
    assert analysis["summary_counts"]["total_probes"] == 3
    assert analysis["summary_counts"]["reliable_responses"] >= 1
    assert analysis["summary_counts"]["unreliable_responses"] >= 1
    assert len(analysis["per_probe"]) == 3
    assert set(analysis["variants"]) == {
        "relative_distance",
        "density_ratio",
        "adaptive_density_ratio",
    }
    assert analysis["variants"]["adaptive_density_ratio"]["stats"]["count"] == 3
    assert all(
        0.0 <= row["selected_score"] <= 1.0
        for row in analysis["per_probe"]
    )
