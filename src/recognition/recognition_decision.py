"""
Recognition decision helpers.

This module contains open-set decision logic for recognition. It works
on ranked identity distances and keeps accept/reject rules separate from
metric computation.
"""

import numpy as np


UNKNOWN_IDENTITY = "Unknown"
DEFAULT_RELIABILITY_MODE = "margin"
SUPPORTED_RELIABILITY_MODES = {
    "margin",
    "relative_distance",
    "density_ratio",
    "adaptive_density_ratio",
}


def build_ranked_identity_distances(pairwise_row, identity_columns):
    """
    Build an ordered list of identity distances from one pairwise row.
    """

    ranked_distances = []
    for identity_name in identity_columns:
        distance_key = f"distance__{identity_name}"
        if distance_key not in pairwise_row:
            continue
        ranked_distances.append((identity_name, float(pairwise_row[distance_key])))

    ranked_distances.sort(key=lambda item: (item[1], item[0]))
    return ranked_distances


def compute_margin_score(ranked_distances):
    """
    Return the absolute distance margin between the first two identities.
    """

    if len(ranked_distances) < 2:
        return None

    return float(ranked_distances[1][1] - ranked_distances[0][1])


def compute_relative_distance_score(ranked_distances):
    """
    Compute a normalized best-vs-second distance separation score.
    """

    if not ranked_distances:
        return None

    if len(ranked_distances) == 1:
        return 1.0

    first_distance = float(ranked_distances[0][1])
    second_distance = float(ranked_distances[1][1])
    last_distance = float(ranked_distances[-1][1])
    denominator = max(last_distance - first_distance, 1e-12)
    return float(np.clip((second_distance - first_distance) / denominator, 0.0, 1.0))


def compute_density_ratio_score(ranked_distances):
    """
    Compute a density-style SRR score from identity-level distances.
    """

    if not ranked_distances:
        return None

    first_distance = float(ranked_distances[0][1])
    radius = 2.0 * first_distance
    neighbourhood_size = sum(
        1 for _, distance in ranked_distances if float(distance) < radius
    )
    return float(1.0 - (neighbourhood_size / len(ranked_distances)))


def compute_adaptive_density_ratio_score(ranked_distances):
    """
    Compute an adaptive density-style SRR score from identity distances.
    """

    if not ranked_distances:
        return None

    first_distance = float(ranked_distances[0][1])
    if len(ranked_distances) == 1:
        return 1.0

    second_distance = float(ranked_distances[1][1])
    gap = max(second_distance - first_distance, 0.0)
    adaptive_radius = 2.0 * first_distance * (1.0 + 1.0 / (1.0 + gap))
    neighbourhood_size = sum(
        1 for _, distance in ranked_distances if float(distance) < adaptive_radius
    )
    return float(1.0 - (neighbourhood_size / len(ranked_distances)))


def compute_reliability_score(ranked_distances, mode=DEFAULT_RELIABILITY_MODE):
    """
    Compute the selected reliability score for a ranked identity list.
    """

    if mode == "margin":
        return compute_margin_score(ranked_distances)
    if mode == "relative_distance":
        return compute_relative_distance_score(ranked_distances)
    if mode == "density_ratio":
        return compute_density_ratio_score(ranked_distances)
    if mode == "adaptive_density_ratio":
        return compute_adaptive_density_ratio_score(ranked_distances)

    raise ValueError(
        f"Unsupported reliability mode: {mode}. "
        f"Use one of {sorted(SUPPORTED_RELIABILITY_MODES)}."
    )


def decide_open_set_identity(
    ranked_distances,
    true_identity=None,
    is_known=False,
    distance_threshold=None,
    reliability_threshold=None,
    reliability_mode=DEFAULT_RELIABILITY_MODE,
):
    """
    Decide one open-set recognition response from ranked candidates.
    """

    if not ranked_distances:
        best_identity = None
        best_distance = None
        second_identity = None
        second_distance = None
    else:
        best_identity, best_distance = ranked_distances[0]
        if len(ranked_distances) > 1:
            second_identity, second_distance = ranked_distances[1]
        else:
            second_identity, second_distance = None, None

    reliability_score = compute_reliability_score(
        ranked_distances,
        mode=reliability_mode,
    )
    distance_ok = (
        distance_threshold is not None
        and best_distance is not None
        and float(best_distance) < float(distance_threshold)
    )
    reliability_ok = (
        reliability_threshold is None
        or (
            reliability_score is not None
            and float(reliability_score) >= float(reliability_threshold)
        )
    )
    accepted = bool(distance_ok and reliability_ok)
    final_identity = best_identity if accepted else UNKNOWN_IDENTITY

    if accepted:
        if is_known and final_identity == true_identity:
            decision = "ACCEPTED_CORRECT"
        elif is_known:
            decision = "ACCEPTED_WRONG"
        else:
            decision = "FALSE_ACCEPT"
    else:
        decision = "REJECTED_UNKNOWN"

    return {
        "true_id": true_identity if is_known else None,
        "is_known": bool(is_known),
        "predicted_id": best_identity,
        "final_identity": final_identity,
        "accepted": accepted,
        "decision": decision,
        "distance": best_distance,
        "distance_threshold": distance_threshold,
        "distance_ok": distance_ok,
        "second_identity": second_identity,
        "second_distance": second_distance,
        "reliability_mode": reliability_mode,
        "reliability_score": reliability_score,
        "reliability_threshold": reliability_threshold,
        "reliability_ok": reliability_ok,
    }


def build_open_set_decisions_from_pairwise(
    pairwise_analysis,
    distance_threshold,
    reliability_threshold=None,
    reliability_mode=DEFAULT_RELIABILITY_MODE,
):
    """
    Build final open-set decision records from pairwise identity distances.
    """

    if not pairwise_analysis or not pairwise_analysis.get("available"):
        return []

    identity_columns = pairwise_analysis.get("identity_columns", [])
    decisions = []

    for row in pairwise_analysis.get("rows", []):
        ranked_distances = build_ranked_identity_distances(row, identity_columns)
        true_identity = row.get("true_identity")
        is_known = bool(row.get("is_known", False))
        decision = decide_open_set_identity(
            ranked_distances,
            true_identity=true_identity,
            is_known=is_known,
            distance_threshold=distance_threshold,
            reliability_threshold=reliability_threshold,
            reliability_mode=reliability_mode,
        )
        decision.update(
            {
                "probe_index": row.get("probe_index"),
                "split": row.get("split"),
                "margin": row.get("margin"),
                "true_identity_rank": row.get("true_identity_rank"),
            }
        )
        decisions.append(decision)

    return decisions
