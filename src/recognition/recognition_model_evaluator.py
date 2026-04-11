"""
Module: recognition_model_evaluator

Description:
This module evaluates the trained LBPH face recognition model on one
selected probe dataset or on all supported probe datasets
sequentially.

Evaluation uses the single recognition model currently stored on disk,
which is overwritten every time a new training run is executed. For
selected evaluation pairs, it also stores an offline threshold
reference derived from the current score distributions.

Besides the global accuracy summary, the evaluator also reports:
- score statistics for genuine and impostor predictions
- the worst-performing identities in the current probe split
- a small threshold analysis that approximates runtime accept/reject
  behavior for a few candidate thresholds

Offline threshold updates are intentionally stricter than plain
evaluation: a threshold is persisted only when the selected run is an
approved reference pair and contains both genuine and impostor scores,
so fallback thresholds from overly easy runs are never promoted to the
stored evaluation reference.
"""

import cv2
import json
import pickle
from pathlib import Path
import statistics
import time

from src.preprocessing.face_preprocessing import get_recognition_preprocessing_config
from src.recognition.face_recognizer import DEFAULT_PREDICTION_THRESHOLD
from src.recognition.recognition_dataset_loader import (
    load_probe_dataset,
    normalize_identity_name,
    should_namespace_identities,
)


# ----- CONSTANTS ----- #
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "lbph_model.xml"
LABEL_MAP_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "label_map.pkl"
RECOGNITION_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "recognition_config.json"
EVAL_MODE = "custom"
PROBE_PROGRESS_STEP = 100
DEBUG_LABEL_MAP_PREVIEW_LIMIT = 10
DEBUG_SAMPLE_PREVIEW_LIMIT = 10


def load_trained_recognition_model():
    """
    Load the trained recognition model and label map from disk.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Recognition model not found: {MODEL_PATH}")

    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(f"Recognition label map not found: {LABEL_MAP_PATH}")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))

    with open(LABEL_MAP_PATH, "rb") as file:
        label_map = pickle.load(file)

    return recognizer, label_map


def save_recognition_config(config_data):
    """
    Save the recognition evaluation configuration to disk as JSON.
    """
    RECOGNITION_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RECOGNITION_CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(config_data, file, indent=4)


def load_evaluation_threshold_config():
    """
    Load the currently saved offline threshold configuration, if any.

    This is used only for reporting purposes inside the evaluator so the
    current run can be compared against the last persisted offline
    threshold reference.
    """
    if not RECOGNITION_CONFIG_PATH.exists():
        return None

    with open(RECOGNITION_CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def should_save_evaluation_threshold(
    train_mode,
    eval_mode,
    allow_config_update=False,
):
    """
    Decide whether the current evaluation should update the offline threshold.

    The stored offline threshold is updated only for the approved
    reference pairs used by the project. The run must also contain both
    genuine and impostor scores and must not rely on the fallback
    threshold rule.
    """
    valid_pairs = {
        ("custom", "custom"),
        ("orl", "orl"),
        ("custom_orl", "orl"),
    }
    return allow_config_update and (train_mode, eval_mode) in valid_pairs


def compute_score_statistics(scores):
    """
    Compute compact descriptive statistics for a list of confidence scores.
    """
    if not scores:
        return None

    return {
        "count": len(scores),
        "mean": statistics.fmean(scores),
        "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "min": min(scores),
        "max": max(scores),
    }


def compute_suggested_threshold(genuine_scores, impostor_scores):
    """
    Compute an offline threshold suggestion from the evaluation score
    distributions.

    When both genuine and impostor scores are available, use the midpoint
    between the worst genuine score and the best impostor score. This is
    less sensitive to single-score outliers than the previous fixed-margin
    rule. If impostor scores are not available, fall back to a conservative
    genuine-only estimate.
    """
    if not genuine_scores:
        return None, "unavailable"

    max_genuine = max(genuine_scores)

    if impostor_scores:
        min_impostor = min(impostor_scores)
        return (max_genuine + min_impostor) / 2, "midpoint_max_genuine_min_impostor"

    return max_genuine + 5, "max_genuine_plus_5_fallback"


def compute_open_set_threshold(known_scores, unknown_scores):
    """
    Compute a threshold suggestion for an open-set evaluation.

    The goal is to separate accepted known identities from rejected
    unknown identities using the LBPH confidence score, where lower is
    better.
    """
    if not known_scores:
        return None, "unavailable"

    max_known = max(known_scores)

    if unknown_scores:
        min_unknown = min(unknown_scores)
        return (max_known + min_unknown) / 2, "midpoint_max_known_min_unknown"

    return max_known + 5, "max_known_plus_5_fallback"


def compute_identity_accuracy(records):
    """
    Aggregate probe outcomes by true identity.

    The returned dictionary is keyed by the ground-truth identity so the
    report highlights which enrolled identities are hardest to recognize.
    """
    identity_stats = {}

    for record in records:
        identity_name = record["true_identity"]
        current_stats = identity_stats.setdefault(
            identity_name,
            {"samples": 0, "correct": 0},
        )
        current_stats["samples"] += 1
        current_stats["correct"] += int(record["is_correct"])

    for current_stats in identity_stats.values():
        current_stats["accuracy"] = current_stats["correct"] / current_stats["samples"]

    return identity_stats


def evaluate_threshold_candidates(records, candidates):
    """
    Evaluate a few threshold candidates on the collected records.

    For LBPH, lower confidence is better:
    - confidence < threshold  -> accepted as the predicted identity
    - confidence >= threshold -> rejected as Unknown
    """
    threshold_results = []

    for candidate in candidates:
        threshold = candidate["threshold"]

        if threshold is None:
            continue

        accepted_genuine = 0
        rejected_genuine = 0
        accepted_impostor = 0
        rejected_impostor = 0

        for record in records:
            is_accepted = record["confidence"] < threshold

            if record["is_correct"]:
                if is_accepted:
                    accepted_genuine += 1
                else:
                    rejected_genuine += 1
            else:
                if is_accepted:
                    accepted_impostor += 1
                else:
                    rejected_impostor += 1

        threshold_results.append(
            {
                "label": candidate["label"],
                "threshold": threshold,
                "accepted_genuine": accepted_genuine,
                "rejected_genuine": rejected_genuine,
                "accepted_impostor": accepted_impostor,
                "rejected_impostor": rejected_impostor,
            }
        )

    return threshold_results


def evaluate_open_set_threshold_candidates(records, candidates):
    """
    Evaluate a few thresholds on an open-set probe collection.

    The records are expected to include:
    - ``split``: ``probe_known`` or ``probe_unknown`` for the custom dataset
      (other probe splits are treated as known)
    - ``identity``: ground-truth identity
    - ``predicted_identity``: best-match identity produced by LBPH
    - ``confidence``: LBPH confidence score
    """
    threshold_results = []

    for candidate in candidates:
        threshold = candidate["threshold"]

        if threshold is None:
            continue

        known_accepted_correctly = 0
        known_rejected_as_unknown = 0
        known_accepted_wrongly_as_other_known = 0
        unknown_rejected_correctly = 0
        unknown_accepted_wrongly_as_known = 0

        for record in records:
            is_accepted = record["confidence"] < threshold
            final_identity = (
                record["predicted_identity"] if is_accepted else "Unknown"
            )
            is_unknown_split = record["split"] == "probe_unknown"

            if is_unknown_split:
                if final_identity == "Unknown":
                    unknown_rejected_correctly += 1
                else:
                    unknown_accepted_wrongly_as_known += 1
            else:
                if final_identity == record["identity"]:
                    known_accepted_correctly += 1
                elif final_identity == "Unknown":
                    known_rejected_as_unknown += 1
                else:
                    known_accepted_wrongly_as_other_known += 1

        threshold_results.append(
            {
                "label": candidate["label"],
                "threshold": threshold,
                "known_accepted_correctly": known_accepted_correctly,
                "known_rejected_as_unknown": known_rejected_as_unknown,
                "known_accepted_wrongly_as_other_known": known_accepted_wrongly_as_other_known,
                "unknown_rejected_correctly": unknown_rejected_correctly,
                "unknown_accepted_wrongly_as_known": unknown_accepted_wrongly_as_known,
            }
        )

    return threshold_results


def print_score_statistics(name, stats):
    """
    Print a compact score-statistics block.
    """
    if stats is None:
        print(f"{name} score stats: unavailable")
        return

    print(
        f"{name} score stats: "
        f"count={stats['count']} | "
        f"mean={stats['mean']:.4f} | "
        f"std={stats['std']:.4f} | "
        f"min={stats['min']:.4f} | "
        f"max={stats['max']:.4f}"
    )


def print_identity_breakdown(identity_stats, limit=5):
    """
    Print the worst-performing identities in the current evaluation.
    """
    if not identity_stats:
        print("Worst identities: unavailable")
        return

    print(f"Worst identities (top {min(limit, len(identity_stats))}):")

    sorted_items = sorted(
        identity_stats.items(),
        key=lambda item: (item[1]["accuracy"], -item[1]["samples"], item[0]),
    )

    for identity_name, current_stats in sorted_items[:limit]:
        print(
            f"- {identity_name}: "
            f"{current_stats['correct']}/{current_stats['samples']} correct | "
            f"accuracy={current_stats['accuracy']:.4f}"
        )


def print_threshold_analysis(threshold_results):
    """
    Print a compact accept/reject analysis for candidate thresholds.
    """
    if not threshold_results:
        print("Threshold analysis: unavailable")
        return

    print("Threshold analysis:")

    for result in threshold_results:
        print(
            f"- {result['label']}: "
            f"threshold={result['threshold']:.4f} | "
            f"accepted_genuine={result['accepted_genuine']} | "
            f"rejected_genuine={result['rejected_genuine']} | "
            f"accepted_impostor={result['accepted_impostor']} | "
            f"rejected_impostor={result['rejected_impostor']}"
        )


def print_open_set_threshold_analysis(threshold_results):
    """
    Print a compact open-set accept/reject analysis for candidate thresholds.
    """
    if not threshold_results:
        print("Open-set threshold analysis: unavailable")
        return

    print("Open-set threshold analysis:")

    for result in threshold_results:
        print(
            f"- {result['label']}: "
            f"threshold={result['threshold']:.4f} | "
            f"known_ok={result['known_accepted_correctly']} | "
            f"known_reject={result['known_rejected_as_unknown']} | "
            f"known_wrong_accept={result['known_accepted_wrongly_as_other_known']} | "
            f"unknown_ok={result['unknown_rejected_correctly']} | "
            f"unknown_wrong_accept={result['unknown_accepted_wrongly_as_known']}"
        )


def evaluate_custom_open_set_dataset(
    train_mode,
    recognizer,
    label_map,
    allow_config_update=False,
):
    """
    Evaluate the custom recognition dataset as an open-set problem.

    The custom gallery/probe layout is now split into ``probe_known`` and
    ``probe_unknown`` so the evaluation can report acceptance and rejection
    behavior separately for known and unknown identities.
    """
    print("=== RECOGNITION EVALUATION MODE: CUSTOM OPEN SET ===")
    preprocessing_config = get_recognition_preprocessing_config()
    print(
        "Recognition preprocessing:",
        f"normalization={preprocessing_config['normalization_mode']},",
        f"padding_ratio={preprocessing_config['crop_padding_ratio']:.3f}",
    )

    load_start_time = time.perf_counter()
    known_records, known_dataset_counts = load_probe_dataset(
        "custom",
        split="probe_known",
        namespace_identities=False,
    )
    unknown_records, unknown_dataset_counts = load_probe_dataset(
        "custom",
        split="probe_unknown",
        namespace_identities=False,
    )
    load_elapsed = time.perf_counter() - load_start_time

    probe_records = known_records + unknown_records
    if not probe_records:
        print("No valid probe images found for evaluation.")
        print()
        return

    print("Label map preview:")
    for label_index, identity_name in sorted(label_map.items())[:DEBUG_LABEL_MAP_PREVIEW_LIMIT]:
        print(f"- {label_index} -> {identity_name}")

    known_scores = []
    unknown_scores = []
    evaluation_records = []

    for index, record in enumerate(probe_records, start=1):
        label, confidence = recognizer.predict(record["image"])
        predicted_name = normalize_identity_name(label_map[label])
        true_name = normalize_identity_name(record["identity"])
        is_known_split = record["split"] == "probe_known"

        if is_known_split:
            known_scores.append(confidence)
        else:
            unknown_scores.append(confidence)

        evaluation_records.append(
            {
                "split": record["split"],
                "identity": true_name,
                "predicted_label": label,
                "predicted_identity": predicted_name,
                "confidence": confidence,
                "is_known_split": is_known_split,
            }
        )

        if index <= DEBUG_SAMPLE_PREVIEW_LIMIT:
            print(
                "Sample preview:",
                f"split={record['split']}",
                f"true_raw={record['identity']}",
                f"true_norm={true_name}",
                f"label={label}",
                f"predicted_norm={predicted_name}",
                f"confidence={confidence:.4f}",
            )

        if index % PROBE_PROGRESS_STEP == 0:
            print(f"Evaluated {index}/{len(probe_records)} probe images")

    evaluation_threshold = DEFAULT_PREDICTION_THRESHOLD
    accepted_known_correctly = 0
    known_rejected_as_unknown = 0
    known_accepted_wrongly_as_other_known = 0
    unknown_rejected_correctly = 0
    unknown_accepted_wrongly_as_known = 0

    for record in evaluation_records:
        is_accepted = record["confidence"] < evaluation_threshold
        final_identity = record["predicted_identity"] if is_accepted else "Unknown"

        if record["is_known_split"]:
            if final_identity == record["identity"]:
                accepted_known_correctly += 1
            elif final_identity == "Unknown":
                known_rejected_as_unknown += 1
            else:
                known_accepted_wrongly_as_other_known += 1
        else:
            if final_identity == "Unknown":
                unknown_rejected_correctly += 1
            else:
                unknown_accepted_wrongly_as_known += 1

    print("Probe samples:", len(probe_records))
    print(f"Probe loading completed in {load_elapsed:.2f}s")
    print(
        "probe_known samples:",
        sum(known_dataset_counts.values()),
    )
    print(
        "probe_unknown samples:",
        sum(unknown_dataset_counts.values()),
    )
    print(f"Using runtime threshold: {evaluation_threshold:.2f}")
    print("Known accepted correctly:", accepted_known_correctly)
    print("Known rejected as unknown:", known_rejected_as_unknown)
    print("Known accepted wrongly as other known:", known_accepted_wrongly_as_other_known)
    print("Unknown rejected correctly:", unknown_rejected_correctly)
    print("Unknown accepted wrongly as known:", unknown_accepted_wrongly_as_known)

    known_stats = compute_score_statistics(known_scores)
    unknown_stats = compute_score_statistics(unknown_scores)

    if known_stats is not None:
        print("Mean known confidence:", known_stats["mean"])

    if unknown_stats is not None:
        print("Mean unknown confidence:", unknown_stats["mean"])

    open_set_threshold, open_set_threshold_rule = compute_open_set_threshold(
        known_scores,
        unknown_scores,
    )

    if open_set_threshold is not None:
        print("Suggested open-set threshold:", open_set_threshold)
        print("Suggested open-set threshold rule:", open_set_threshold_rule)

    current_evaluation_config = load_evaluation_threshold_config()
    saved_evaluation_threshold = None
    saved_evaluation_threshold_source = None
    if current_evaluation_config is not None:
        saved_evaluation_threshold = current_evaluation_config.get("unknown_threshold")
        saved_evaluation_threshold_source = current_evaluation_config.get("threshold_source")
        print(
            "Saved evaluation threshold source:",
            saved_evaluation_threshold_source or "unknown",
        )

    conservative_threshold = None
    if open_set_threshold is not None and unknown_scores:
        conservative_threshold = max(0.0, open_set_threshold - 5.0)

    threshold_candidates = [
        {
            "label": "runtime_threshold",
            "threshold": evaluation_threshold,
        },
        {
            "label": "saved_evaluation_threshold",
            "threshold": saved_evaluation_threshold,
        },
        {
            "label": "suggested_open_set_threshold",
            "threshold": open_set_threshold,
        },
        {
            "label": "conservative_threshold",
            "threshold": conservative_threshold,
        },
    ]
    threshold_results = evaluate_open_set_threshold_candidates(
        evaluation_records,
        threshold_candidates,
    )

    print_score_statistics("Known", known_stats)
    print_score_statistics("Unknown", unknown_stats)
    print_open_set_threshold_analysis(threshold_results)

    print(f"Evaluation completed in {time.perf_counter() - load_start_time:.2f}s")
    print()


def evaluate_single_recognition_dataset(
    train_mode,
    eval_mode,
    recognizer,
    label_map,
    namespace_identities=False,
    allow_config_update=False,
):
    """
    Evaluate the trained recognition model on one selected probe dataset.

    The function computes the global summary, richer score statistics,
    a per-identity breakdown, a small threshold analysis, and optionally
    updates the stored offline threshold reference on disk.
    """
    if eval_mode == "custom":
        evaluate_custom_open_set_dataset(
            train_mode,
            recognizer,
            label_map,
            allow_config_update=allow_config_update,
        )
        return

    print(f"=== RECOGNITION EVALUATION MODE: {eval_mode.upper()} ===")
    preprocessing_config = get_recognition_preprocessing_config()
    print(
        "Recognition preprocessing:",
        f"normalization={preprocessing_config['normalization_mode']},",
        f"padding_ratio={preprocessing_config['crop_padding_ratio']:.3f}",
    )

    load_start_time = time.perf_counter()
    probe_records, dataset_counts = load_probe_dataset(
        eval_mode,
        namespace_identities=namespace_identities,
    )
    load_elapsed = time.perf_counter() - load_start_time

    if not probe_records:
        print("No valid probe images found for evaluation.")
        print()
        return

    genuine_scores = []
    impostor_scores = []
    evaluation_records = []
    correct = 0
    total = 0
    evaluation_start_time = time.perf_counter()

    for index, record in enumerate(probe_records, start=1):
        label, confidence = recognizer.predict(record["image"])
        predicted_name = normalize_identity_name(label_map[label])
        true_name = normalize_identity_name(record["identity"])

        if predicted_name == true_name:
            genuine_scores.append(confidence)
            correct += 1
        else:
            impostor_scores.append(confidence)

        evaluation_records.append(
            {
                "true_identity": true_name,
                "predicted_identity": predicted_name,
                "confidence": confidence,
                "is_correct": predicted_name == true_name,
            }
        )

        total += 1

        if index % PROBE_PROGRESS_STEP == 0:
            print(f"Evaluated {index}/{len(probe_records)} probe images")

    print("Probe samples:", total)
    print(f"Probe loading completed in {load_elapsed:.2f}s")

    for dataset_name, sample_count in dataset_counts.items():
        print(f"{dataset_name} probe samples: {sample_count}")

    print("Correct:", correct)
    print("Accuracy:", correct / total)
    print("Genuine matches:", len(genuine_scores))
    print("Impostor matches:", len(impostor_scores))

    genuine_stats = compute_score_statistics(genuine_scores)
    impostor_stats = compute_score_statistics(impostor_scores)
    identity_stats = compute_identity_accuracy(evaluation_records)

    current_evaluation_config = load_evaluation_threshold_config()
    saved_evaluation_threshold = None
    saved_evaluation_threshold_source = None
    if current_evaluation_config is not None:
        saved_evaluation_threshold = current_evaluation_config.get("unknown_threshold")
        saved_evaluation_threshold_source = current_evaluation_config.get("threshold_source")

    suggested_threshold = None
    suggested_threshold_rule = None

    if genuine_stats is not None:
        print("Mean genuine confidence:", genuine_stats["mean"])

    if impostor_stats is not None:
        print("Mean impostor confidence:", impostor_stats["mean"])

    suggested_threshold, suggested_threshold_rule = compute_suggested_threshold(
        genuine_scores,
        impostor_scores,
    )

    if suggested_threshold is not None:
        print("Suggested threshold:", suggested_threshold)
        print("Suggested threshold rule:", suggested_threshold_rule)

    conservative_threshold = None
    if suggested_threshold is not None and impostor_scores:
        conservative_threshold = max(0.0, suggested_threshold - 5.0)

    threshold_candidates = [
        {
            "label": "saved_evaluation_threshold",
            "threshold": saved_evaluation_threshold,
        },
        {
            "label": "suggested_threshold",
            "threshold": suggested_threshold,
        },
        {
            "label": "conservative_threshold",
            "threshold": conservative_threshold,
        },
    ]
    threshold_results = evaluate_threshold_candidates(
        evaluation_records,
        threshold_candidates,
    )

    print_score_statistics("Genuine", genuine_stats)
    print_score_statistics("Impostor", impostor_stats)
    print_identity_breakdown(identity_stats, limit=5)
    if saved_evaluation_threshold is not None:
        print(
            "Saved evaluation threshold source:",
            saved_evaluation_threshold_source or "unknown",
        )
    print_threshold_analysis(threshold_results)

    has_both_classes = bool(genuine_scores) and bool(impostor_scores)
    non_fallback_rule = suggested_threshold_rule != "max_genuine_plus_5_fallback"
    can_update_evaluation_threshold = should_save_evaluation_threshold(
        train_mode,
        eval_mode,
        allow_config_update=allow_config_update,
    )

    if (
        can_update_evaluation_threshold
        and suggested_threshold is not None
        and has_both_classes
        and non_fallback_rule
    ):
        config_data = {
            "unknown_threshold": suggested_threshold,
            "threshold_source": f"{train_mode}_to_{eval_mode}",
            "train_mode": train_mode,
            "eval_mode": eval_mode,
            "genuine_matches": len(genuine_scores),
            "impostor_matches": len(impostor_scores),
            "mean_genuine_confidence": genuine_stats["mean"] if genuine_stats else None,
            "mean_impostor_confidence": impostor_stats["mean"] if impostor_stats else None,
            "suggested_threshold_rule": suggested_threshold_rule,
        }
        save_recognition_config(config_data)
        print(f"Recognition evaluation config saved to: {RECOGNITION_CONFIG_PATH}")
    elif allow_config_update:
        if not can_update_evaluation_threshold:
            print(
                "Evaluation threshold not updated: "
                "train/eval pair is not allowed to define the stored reference."
            )
        elif not has_both_classes:
            print(
                "Evaluation threshold not updated: evaluation does not contain both "
                "genuine and impostor scores."
            )
        elif not non_fallback_rule:
            print(
                "Evaluation threshold not updated: fallback threshold rule is not "
                "allowed for the stored reference."
            )

    print(f"Evaluation completed in {time.perf_counter() - evaluation_start_time:.2f}s")
    print()


def evaluate_recognition_model(train_mode=None, eval_mode=None):
    """
    Evaluate the recognition model on one dataset or on all supported
    datasets sequentially.

    The standalone defaults match the current project reference setup:
    `custom -> custom`.
    """
    selected_train_mode = train_mode or "custom"
    selected_eval_mode = eval_mode or EVAL_MODE
    namespace_identities = should_namespace_identities(selected_train_mode)
    total_start_time = time.perf_counter()

    recognizer, label_map = load_trained_recognition_model()

    if selected_eval_mode == "all":
        for mode in ("custom", "orl", "lfw"):
            evaluate_single_recognition_dataset(
                selected_train_mode,
                mode,
                recognizer,
                label_map,
                namespace_identities=namespace_identities,
                allow_config_update=False,
            )
        print(f"Total evaluation time: {time.perf_counter() - total_start_time:.2f}s")
        return

    evaluate_single_recognition_dataset(
        selected_train_mode,
        selected_eval_mode,
        recognizer,
        label_map,
        namespace_identities=namespace_identities,
        allow_config_update=True,
    )
    print(f"Total evaluation time: {time.perf_counter() - total_start_time:.2f}s")


if __name__ == "__main__":
    evaluate_recognition_model()
