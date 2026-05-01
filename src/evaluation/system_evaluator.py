"""
Module: system_evaluator

Description:
Offline system-level evaluation for the complete biometric pipeline.

The evaluator measures an open-set identification scenario with PAD
gating. It keeps three input classes separate:
- known_live: enrolled live users
- unknown_live: non-enrolled live users / biometric impostors
- spoof_attack: presentation attacks

The runtime webcam pipeline is not used or modified here. This module
loads the saved PAD and recognition artifacts, computes recognition-only
metrics, PAD-only metrics, and the final PAD-gated recognition decision,
then exports report-ready JSON and CSV files.
"""

import argparse
import cv2
import csv
import json
from pathlib import Path
import time

import numpy as np

from src.antispoofing.pad_inference_preprocessing import extract_pad_features_from_crop
from src.antispoofing.pad_runtime_detector import AntiSpoofingDetector
from src.evaluation.pad_metrics import evaluate_pad_at_threshold
from src.antispoofing.training.pad_dataset_loader import load_custom_pad_test_dataset
from src.recognition.face_recognizer import DEFAULT_PREDICTION_THRESHOLD
from src.recognition.recognition_dataset_loader import (
    load_probe_dataset,
    normalize_identity_name,
)
from src.evaluation.recognition_metrics import (
    compute_recognition_eer,
    evaluate_recognition_at_threshold,
    sweep_recognition_thresholds,
)
from src.evaluation.recognition_model_evaluator import load_trained_recognition_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYSTEM_METRICS_ROOT = PROJECT_ROOT / "logs" / "metrics" / "system"
DEFAULT_OUTPUT_STEM = "system_evaluation"
DEFAULT_PAD_THRESHOLD = 0.5


def safe_div(numerator, denominator):
    """
    Divide two counts while keeping empty subsets reportable.
    """

    if denominator == 0:
        return 0.0
    return numerator / denominator


def predict_pad_from_crop(pad_detector, face_crop, pad_threshold):
    """
    Return a PAD decision and live score for one already localized face crop.

    The score is always the live-class probability, so it can be used for
    APCER/BPCER computation with the project convention:
    score >= threshold -> live, score < threshold -> spoof.
    """

    features = extract_pad_features_from_crop(face_crop)
    if features is None or len(features) == 0:
        return {
            "status": "invalid_input",
            "label": "spoof",
            "live_score": 0.0,
        }

    features = features.reshape(1, -1)
    extracted_feature_dim = int(features.shape[1])
    if extracted_feature_dim != pad_detector.expected_feature_dim:
        raise ValueError(
            "PAD feature mismatch: extracted "
            f"{extracted_feature_dim} features, but the loaded model expects "
            f"{pad_detector.expected_feature_dim}."
        )

    features_scaled = pad_detector.scaler.transform(features)
    probabilities = pad_detector.classifier.predict_proba(features_scaled)[0]
    live_score = float(probabilities[1])
    label = "live" if live_score >= pad_threshold else "spoof"

    return {
        "status": "ok",
        "label": label,
        "live_score": live_score,
    }


def predict_pad_from_features(pad_detector, features, pad_threshold):
    """
    Return a PAD decision and live score for one pre-extracted PAD feature row.
    """

    features = np.asarray(features, dtype=np.float32).reshape(1, -1)
    extracted_feature_dim = int(features.shape[1])
    if extracted_feature_dim != pad_detector.expected_feature_dim:
        raise ValueError(
            "PAD feature mismatch: extracted "
            f"{extracted_feature_dim} features, but the loaded model expects "
            f"{pad_detector.expected_feature_dim}."
        )

    features_scaled = pad_detector.scaler.transform(features)
    probabilities = pad_detector.classifier.predict_proba(features_scaled)[0]
    live_score = float(probabilities[1])
    label = "live" if live_score >= pad_threshold else "spoof"

    return {
        "status": "ok",
        "label": label,
        "live_score": live_score,
    }


def build_live_samples():
    """
    Load known and unknown live samples from the custom recognition probes.
    """

    known_records, _, _ = load_probe_dataset(
        "custom",
        split="probe_known",
        namespace_identities=False,
        collect_pose_metrics=False,
    )
    unknown_records, _, _ = load_probe_dataset(
        "custom",
        split="probe_unknown",
        namespace_identities=False,
        collect_pose_metrics=False,
    )

    samples = []
    for record in known_records:
        pad_image = cv2.imread(str(record["path"]), cv2.IMREAD_GRAYSCALE)
        if pad_image is None:
            pad_image = record["image"]
        samples.append(
            {
                "sample_type": "known_live",
                "path": str(record["path"]),
                "true_id": normalize_identity_name(record["identity"]),
                "image": record["image"],
                "pad_image": pad_image,
            }
        )

    for record in unknown_records:
        pad_image = cv2.imread(str(record["path"]), cv2.IMREAD_GRAYSCALE)
        if pad_image is None:
            pad_image = record["image"]
        samples.append(
            {
                "sample_type": "unknown_live",
                "path": str(record["path"]),
                "true_id": normalize_identity_name(record["identity"]),
                "image": record["image"],
                "pad_image": pad_image,
            }
        )

    return samples


def build_spoof_samples():
    """
    Load presentation-attack samples from the custom PAD test split.
    """

    X_test, y_test, sample_paths = load_custom_pad_test_dataset(
        return_image_paths=True,
        collect_pose_metrics=False,
    )

    samples = []
    for features, label, sample_path in zip(X_test, y_test, sample_paths):
        if int(label) != 0:
            continue
        samples.append(
            {
                "sample_type": "spoof_attack",
                "path": str(sample_path),
                "true_id": None,
                "features": features,
            }
        )

    return samples


def recognition_decision(sample, recognizer, label_map, threshold):
    """
    Compute the recognition-only decision for one live sample.
    """

    label, distance = recognizer.predict(sample["image"])
    predicted_id = normalize_identity_name(label_map[int(label)])
    accepted = float(distance) < float(threshold)

    if not accepted:
        decision = "REJECTED_UNKNOWN"
    elif sample["sample_type"] == "known_live" and predicted_id == sample["true_id"]:
        decision = "ACCEPTED_CORRECT"
    else:
        decision = "ACCEPTED_WRONG"

    return {
        "predicted_id": predicted_id,
        "distance": float(distance),
        "accepted": accepted,
        "decision": decision,
    }


def final_system_decision(sample, pad_result, recognition_result, recognition_threshold):
    """
    Compute the final PAD-gated recognition decision.
    """

    if pad_result["label"] == "spoof":
        return "PAD_BLOCKED"

    if sample["sample_type"] == "spoof_attack":
        return "ATTACK_PASSED_PAD"

    if recognition_result["distance"] >= recognition_threshold:
        return "REJECTED_UNKNOWN"

    if (
        sample["sample_type"] == "known_live"
        and recognition_result["predicted_id"] == sample["true_id"]
    ):
        return "ACCEPTED_CORRECT"

    return "ACCEPTED_WRONG"


def build_recognition_only_report(rows, recognition_threshold):
    """
    Compute recognition-only open-set identification metrics for live probes.
    """

    records = []
    for row in rows:
        if row["sample_type"] == "spoof_attack":
            continue
        records.append(
            {
                "is_known": row["sample_type"] == "known_live",
                "true_id": row["true_id"] if row["sample_type"] == "known_live" else None,
                "predicted_id": row["predicted_id"],
                "score": row["recognition_distance"],
            }
        )

    if not records:
        return {
            "threshold": recognition_threshold,
            "far": 0.0,
            "frr": 0.0,
            "eer": None,
            "eer_threshold": None,
            "dir": 0.0,
            "misidentification_rate": 0.0,
            "total_known": 0,
            "total_unknown": 0,
        }

    selected_metrics = evaluate_recognition_at_threshold(records, recognition_threshold)
    sweep_results = sweep_recognition_thresholds(records)
    eer_results = compute_recognition_eer(sweep_results)

    return {
        "threshold": float(recognition_threshold),
        "far": selected_metrics["FAR"],
        "frr": selected_metrics["FRR"],
        "eer": eer_results["eer_value"],
        "eer_threshold": eer_results["eer_threshold"],
        "dir": selected_metrics["DIR"],
        "misidentification_rate": selected_metrics["misidentification_rate"],
        "total_known": selected_metrics["total_known"],
        "total_unknown": selected_metrics["total_unknown"],
    }


def build_pad_only_report(rows, pad_threshold):
    """
    Compute PAD-only APCER/BPCER/ACER on live and spoof samples.
    """

    y_true = []
    y_score = []
    for row in rows:
        y_true.append(0 if row["sample_type"] == "spoof_attack" else 1)
        y_score.append(row["pad_live_score"])

    if not y_true:
        return {
            "threshold": pad_threshold,
            "apcer": 0.0,
            "bpcer": 0.0,
            "acer": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    metrics = evaluate_pad_at_threshold(y_true, y_score, pad_threshold)
    return {
        "threshold": float(pad_threshold),
        "apcer": metrics["apcer"],
        "bpcer": metrics["bpcer"],
        "acer": metrics["acer"],
        "confusion_matrix": metrics["confusion_matrix"],
    }


def build_system_level_report(rows):
    """
    Aggregate final PAD-gated recognition outcomes.
    """

    known_rows = [row for row in rows if row["sample_type"] == "known_live"]
    unknown_rows = [row for row in rows if row["sample_type"] == "unknown_live"]
    spoof_rows = [row for row in rows if row["sample_type"] == "spoof_attack"]
    live_rows = known_rows + unknown_rows

    known_rejected = sum(
        row["final_decision"] in {"PAD_BLOCKED", "REJECTED_UNKNOWN"}
        for row in known_rows
    )
    known_wrong = sum(row["final_decision"] == "ACCEPTED_WRONG" for row in known_rows)
    known_correct = sum(
        row["final_decision"] == "ACCEPTED_CORRECT" for row in known_rows
    )
    unknown_accepted = sum(
        row["final_decision"] == "ACCEPTED_WRONG" for row in unknown_rows
    )
    unknown_rejected = sum(
        row["final_decision"] in {"PAD_BLOCKED", "REJECTED_UNKNOWN"}
        for row in unknown_rows
    )
    spoof_blocked = sum(row["final_decision"] == "PAD_BLOCKED" for row in spoof_rows)
    spoof_passed = sum(
        row["final_decision"] == "ATTACK_PASSED_PAD" for row in spoof_rows
    )
    live_blocked_by_pad = sum(row["final_decision"] == "PAD_BLOCKED" for row in live_rows)
    live_passed_pad = sum(row["pad_label"] == "live" for row in live_rows)

    return {
        "final_unknown_live_far": safe_div(unknown_accepted, len(unknown_rows)),
        "final_known_live_frr": safe_div(known_rejected, len(known_rows)),
        "final_known_live_dir": safe_div(known_correct, len(known_rows)),
        "final_known_live_misidentification_rate": safe_div(
            known_wrong,
            len(known_rows),
        ),
        "spoof_attacks_blocked": spoof_blocked,
        "spoof_attacks_passed_pad": spoof_passed,
        "live_samples_blocked_by_pad": live_blocked_by_pad,
        "live_samples_passed_pad": live_passed_pad,
        "known_live_total": len(known_rows),
        "unknown_live_total": len(unknown_rows),
        "spoof_attack_total": len(spoof_rows),
        "accepted_correct_known_live": known_correct,
        "rejected_known_live": known_rejected,
        "accepted_unknown_live": unknown_accepted,
        "rejected_unknown_live": unknown_rejected,
    }


def evaluate_system(recognition_threshold, pad_threshold, output_stem):
    """
    Run the complete offline system-level evaluation.
    """

    start_time = time.perf_counter()
    recognizer, label_map = load_trained_recognition_model()
    pad_detector = AntiSpoofingDetector()

    live_samples = build_live_samples()
    spoof_samples = build_spoof_samples()
    rows = []

    for sample in live_samples:
        pad_result = predict_pad_from_crop(
            pad_detector,
            sample["pad_image"],
            pad_threshold,
        )
        recognition_result = recognition_decision(
            sample,
            recognizer,
            label_map,
            recognition_threshold,
        )
        final_decision = final_system_decision(
            sample,
            pad_result,
            recognition_result,
            recognition_threshold,
        )

        rows.append(
            {
                "sample_type": sample["sample_type"],
                "path": sample["path"],
                "true_id": sample["true_id"],
                "pad_label": pad_result["label"],
                "pad_live_score": pad_result["live_score"],
                "predicted_id": recognition_result["predicted_id"],
                "recognition_distance": recognition_result["distance"],
                "recognition_only_decision": recognition_result["decision"],
                "final_decision": final_decision,
            }
        )

    for sample in spoof_samples:
        pad_result = predict_pad_from_features(
            pad_detector,
            sample["features"],
            pad_threshold,
        )
        final_decision = final_system_decision(
            sample,
            pad_result,
            {
                "predicted_id": None,
                "distance": None,
            },
            recognition_threshold,
        )

        rows.append(
            {
                "sample_type": sample["sample_type"],
                "path": sample["path"],
                "true_id": None,
                "pad_label": pad_result["label"],
                "pad_live_score": pad_result["live_score"],
                "predicted_id": None,
                "recognition_distance": None,
                "recognition_only_decision": "NOT_APPLICABLE",
                "final_decision": final_decision,
            }
        )

    recognition_report = build_recognition_only_report(rows, recognition_threshold)
    pad_report = build_pad_only_report(rows, pad_threshold)
    system_report = build_system_level_report(rows)

    report = {
        "scenario": "open-set identification system-level evaluation with PAD gating",
        "methodological_note": (
            "Spoof attacks are evaluated at PAD level only. They are not "
            "included in recognition FAR, which is computed only on unknown "
            "live users. PAD-gated recognition is evaluated offline and does "
            "not reproduce webcam temporal smoothing."
        ),
        "input_classes": {
            "known_live": "enrolled live users",
            "unknown_live": "non-enrolled live users / biometric impostors",
            "spoof_attack": "presentation attacks",
        },
        "recognition_only": recognition_report,
        "pad_only": pad_report,
        "pad_gated_recognition": system_report,
        "sample_count": len(rows),
        "elapsed_seconds": time.perf_counter() - start_time,
    }

    export_paths = export_system_evaluation(rows, report, output_stem)
    report["export_paths"] = export_paths
    print_summary(report)
    return report


def export_system_evaluation(rows, report, output_stem):
    """
    Export the system-level report and per-sample decisions.
    """

    SYSTEM_METRICS_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = SYSTEM_METRICS_ROOT / f"{output_stem}_report.json"
    csv_path = SYSTEM_METRICS_ROOT / f"{output_stem}_samples.csv"

    fieldnames = [
        "sample_type",
        "path",
        "true_id",
        "pad_label",
        "pad_live_score",
        "predicted_id",
        "recognition_distance",
        "recognition_only_decision",
        "final_decision",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    return {
        "json": str(json_path),
        "samples_csv": str(csv_path),
    }


def format_optional_float(value):
    """
    Format a possibly missing metric value.
    """

    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def print_summary(report):
    """
    Print a compact English summary for terminal runs.
    """

    recognition = report["recognition_only"]
    pad = report["pad_only"]
    system = report["pad_gated_recognition"]

    print("=== SYSTEM-LEVEL EVALUATION ===")
    print("Scenario: open-set identification with PAD gating")
    print(
        "Samples:",
        (
            f"known_live={system['known_live_total']} | "
            f"unknown_live={system['unknown_live_total']} | "
            f"spoof_attack={system['spoof_attack_total']}"
        ),
    )
    print(
        "Recognition-only:",
        (
            f"threshold={recognition['threshold']:.4f} | "
            f"FAR={recognition['far']:.4f} | "
            f"FRR={recognition['frr']:.4f} | "
            f"EER={format_optional_float(recognition['eer'])} | "
            f"DIR={recognition['dir']:.4f} | "
            f"misidentification_rate={recognition['misidentification_rate']:.4f}"
        ),
    )
    print(
        "PAD-only:",
        (
            f"threshold={pad['threshold']:.4f} | "
            f"APCER={pad['apcer']:.4f} | "
            f"BPCER={pad['bpcer']:.4f} | "
            f"ACER={pad['acer']:.4f}"
        ),
    )
    print(
        "PAD-gated recognition:",
        (
            f"final_unknown_live_far={system['final_unknown_live_far']:.4f} | "
            f"final_known_live_frr={system['final_known_live_frr']:.4f} | "
            f"final_known_live_dir={system['final_known_live_dir']:.4f}"
        ),
    )
    print(
        "PAD gate counts:",
        (
            f"spoof_blocked={system['spoof_attacks_blocked']} | "
            f"spoof_passed={system['spoof_attacks_passed_pad']} | "
            f"live_blocked={system['live_samples_blocked_by_pad']} | "
            f"live_passed={system['live_samples_passed_pad']}"
        ),
    )
    print("Exports:")
    print(f"- json: {report['export_paths']['json']}")
    print(f"- samples_csv: {report['export_paths']['samples_csv']}")
    print(f"Elapsed: {report['elapsed_seconds']:.2f}s")


def build_arg_parser():
    """
    Build the command-line interface for system-level evaluation.
    """

    parser = argparse.ArgumentParser(
        description="Run offline system-level PAD-gated recognition evaluation."
    )
    parser.add_argument(
        "--recognition-threshold",
        type=float,
        default=DEFAULT_PREDICTION_THRESHOLD,
        help="LBPH distance threshold used by recognition. Lower distance is better.",
    )
    parser.add_argument(
        "--pad-threshold",
        type=float,
        default=DEFAULT_PAD_THRESHOLD,
        help="PAD live-score threshold. score >= threshold is classified as live.",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help="Filename stem used for exported JSON and CSV files.",
    )
    return parser


def main():
    """
    CLI entry point.
    """

    parser = build_arg_parser()
    args = parser.parse_args()
    evaluate_system(
        recognition_threshold=args.recognition_threshold,
        pad_threshold=args.pad_threshold,
        output_stem=args.output_stem,
    )


if __name__ == "__main__":
    main()
