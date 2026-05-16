"""
Module: pad_model_evaluator

Description:
This module evaluates the performance of the trained presentation
attack detection (PAD) classifier on one or more selected evaluation
datasets.

Supported evaluation flows:
- standard public test evaluation (`casia`, `replay`)
- public validation protocols (`casia_validation`, `replay_validation`)
- custom PAD final test evaluation (`custom`)
- sequential execution across supported datasets (`all`)

For the public validation protocols, the evaluator selects the live-score
threshold on the validation split by enforcing a target APCER operating
point and then freezes that threshold for the final test split.

For custom PAD, the evaluator uses a deterministic pseudo-validation
subset carved out of the custom training split to select the threshold,
while keeping the custom test split as the final frozen evaluation set.

Evaluation metrics such as accuracy, precision, recall, F1 score,
and the confusion matrix are computed to assess the ability of the
model to distinguish between live faces and spoof attacks. Official PAD
metrics are threshold-based and include FAR, FRR, GAR, APCER, BPCER,
ACER, threshold sweeps, and EER diagnostics.
"""

import joblib
import time
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from src.antispoofing.pad_model_config import (
    configure_pad_feature_extractor_from_config,
    get_expected_feature_dim_from_scaler,
    get_pad_artifact_paths,
    load_pad_model_config,
)
from src.antispoofing.pad_probability_utils import (
    get_pad_class_probability_indices,
)
from src.evaluation.pad_metrics import (
    build_pad_threshold_summary,
    compute_pad_eer,
    export_pad_metrics_report,
    evaluate_pad_at_threshold,
    select_pad_threshold_by_target_apcer,
    sweep_pad_thresholds,
)
from src.preprocessing.pose_statistics import format_pose_summary
from src.antispoofing.training.pad_dataset_loader import (
    CUSTOM_PAD_VALIDATION_RATIO,
    CUSTOM_PAD_VALIDATION_SEED,
    load_custom_pad_validation_dataset,
    load_pad_test_dataset_casia,
    load_pad_validation_dataset_casia,
    load_pad_test_dataset_replay,
    load_pad_validation_dataset_replay,
    load_custom_pad_test_dataset,
)


EVAL_MODE = "custom"     # casia, replay, casia_validation, replay_validation, custom
LIVE_FALSE_NEGATIVE_LOG_LIMIT = 20
PAD_VALIDATION_TARGET_APCER = 0.02


def load_configured_test_dataset(
    eval_mode,
    include_paths=False,
    collect_pose_metrics=False,
):
    """
    Load the concrete PAD dataset split selected by ``eval_mode``.

    Supported modes:
    - casia
    - casia_validation
    - replay
    - replay_validation
    - custom
    """

    if eval_mode == "casia":
        return "CASIA test", load_pad_test_dataset_casia(
            collect_pose_metrics=collect_pose_metrics,
            return_pose_summary=collect_pose_metrics,
        )

    if eval_mode == "casia_validation":
        return "CASIA validation", load_pad_validation_dataset_casia(
            collect_pose_metrics=collect_pose_metrics,
            return_pose_summary=collect_pose_metrics,
        )

    if eval_mode == "replay":
        return "Replay-Attack test", load_pad_test_dataset_replay(
            collect_pose_metrics=collect_pose_metrics,
            return_pose_summary=collect_pose_metrics,
        )

    if eval_mode == "replay_validation":
        return "Replay-Attack validation", load_pad_validation_dataset_replay(
            collect_pose_metrics=collect_pose_metrics,
            return_pose_summary=collect_pose_metrics,
        )

    if eval_mode == "custom":
        return "Custom test", load_custom_pad_test_dataset(
            return_image_paths=include_paths,
            collect_pose_metrics=collect_pose_metrics,
            return_pose_summary=collect_pose_metrics,
        )

    raise ValueError(
        f"Unsupported EVAL_MODE: {eval_mode}. "
        "Use 'casia', 'casia_validation', 'replay', 'replay_validation', "
        "'custom', or 'all'."
    )


def get_live_score_details(classifier, X_test_scaled):
    """
    Compute per-sample live confidence details when supported by the classifier.
    """

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(X_test_scaled)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            live_index = get_pad_class_probability_indices(classifier)["live"]
            return probabilities[:, live_index], "live_prob"

    if hasattr(classifier, "decision_function"):
        decision = classifier.decision_function(X_test_scaled)
        decision = np.asarray(decision, dtype=np.float64).reshape(-1)
        return decision, "decision"

    return None, None


def build_prediction_probability_mismatch_summary(y_pred, score_values, score_label):
    """
    Compare hard PAD predictions with probability-threshold labels.

    This is diagnostic only. It helps decide whether runtime should keep
    using ``classifier.predict`` for current labels or rely only on the
    live probability used by official PAD metrics.
    """

    if score_label != "live_prob" or score_values is None:
        return None

    y_pred = np.asarray(y_pred, dtype=np.int32).reshape(-1)
    live_scores = np.asarray(score_values, dtype=np.float64).reshape(-1)
    probability_labels = (live_scores >= 0.5).astype(np.int32)
    mismatch_mask = y_pred != probability_labels
    mismatch_count = int(np.sum(mismatch_mask))
    total = int(y_pred.size)
    near_threshold_mask = np.abs(live_scores - 0.5) <= 0.05
    near_threshold_mismatches = int(np.sum(mismatch_mask & near_threshold_mask))

    if mismatch_count > 0:
        mismatch_scores = live_scores[mismatch_mask]
        min_score = float(np.min(mismatch_scores))
        max_score = float(np.max(mismatch_scores))
        mean_score = float(np.mean(mismatch_scores))
    else:
        min_score = None
        max_score = None
        mean_score = None

    return {
        "total": total,
        "mismatch_count": mismatch_count,
        "mismatch_rate": mismatch_count / total if total > 0 else 0.0,
        "near_threshold_mismatches": near_threshold_mismatches,
        "min_live_score": min_score,
        "max_live_score": max_score,
        "mean_live_score": mean_score,
    }


def compute_pad_evaluation_metrics(
    y_test,
    score_values,
    score_label,
    selected_threshold=None,
    threshold_source=None,
    threshold_selection=None,
    output_stem=None,
    export_curves=False,
):
    """
    Compute the official PAD metrics used by the project reports.

    The evaluator keeps the raw predictions and score arrays, while this
    function turns them into the reportable PAD metrics:
    FAR, FRR, GAR, APCER, BPCER, ACER, threshold summaries, ROC/DET
    exports when requested, and EER diagnostics.
    """

    if selected_threshold is not None:
        selected_threshold = float(selected_threshold)
    elif score_label == "decision":
        selected_threshold = 0.0
    else:
        selected_threshold = 0.5

    selected_metrics = evaluate_pad_at_threshold(
        y_test,
        score_values,
        selected_threshold,
    )
    sweep_results = sweep_pad_thresholds(y_test, score_values)
    eer_results = compute_pad_eer(sweep_results)
    threshold_summary = build_pad_threshold_summary(sweep_results)

    report = {
        "selected_threshold": selected_threshold,
        "threshold_source": threshold_source,
        "threshold_selection": threshold_selection,
        "selected_metrics": selected_metrics,
        "sweep_results": sweep_results,
        "eer_results": eer_results,
        "threshold_summary": threshold_summary,
    }

    if output_stem is not None:
        report["export_paths"] = export_pad_metrics_report(
            report,
            output_stem,
            export_curves=export_curves,
        )

    return report


def log_pad_evaluation_diagnostics(
    dataset_name,
    y_test,
    y_pred,
    score_values,
    score_label,
    sample_paths,
    load_elapsed,
    evaluation_elapsed,
    metrics_report,
    mismatch_summary=None,
    pose_summary=None,
    log_live_false_negatives=False,
    live_false_negative_limit=LIVE_FALSE_NEGATIVE_LOG_LIMIT,
):
    """
    Print the project diagnostics for PAD.

    This function intentionally keeps the human-readable logs separate
    from the official metric computation. The actual PAD metrics are
    produced by ``pad_metrics.py`` and only summarized here.
    """

    print(f"=== PAD DIAGNOSTICS: {dataset_name} ===")
    selected_metrics = metrics_report["selected_metrics"]
    threshold_summary = metrics_report["threshold_summary"]
    eer_results = metrics_report["eer_results"]

    print(f"PAD evaluation dataset: {dataset_name}")
    print(
        "PAD evaluation samples:",
        (
            f"total={len(y_test)} | "
            f"live={(y_test == 1).sum()} | spoof={(y_test == 0).sum()} | "
            f"pred_live={(y_pred == 1).sum()} | pred_spoof={(y_pred == 0).sum()}"
        ),
    )
    if pose_summary is not None:
        print(format_pose_summary(pose_summary, title="PAD pose summary:"))
    print(f"PAD evaluation dataset loading completed in {load_elapsed:.2f}s")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print(
        "PAD project logs are diagnostic hard-label metrics; "
        "official PAD metrics use live_score thresholding."
    )
    print(
        "PAD project logs:",
        f"accuracy={accuracy:.4f} | precision={precision:.4f} | recall={recall:.4f} | f1={f1:.4f}",
    )
    print("PAD confusion matrix: rows=true, cols=predicted, order=[spoof(0), live(1)]\n", conf_matrix)
    if mismatch_summary is not None:
        min_score = mismatch_summary["min_live_score"]
        max_score = mismatch_summary["max_live_score"]
        mean_score = mismatch_summary["mean_live_score"]
        score_range = (
            "n/a"
            if min_score is None
            else f"min={min_score:.4f} | mean={mean_score:.4f} | max={max_score:.4f}"
        )
        print(
            "PAD predict/proba diagnostic:",
            (
                f"mismatch={mismatch_summary['mismatch_count']}/"
                f"{mismatch_summary['total']} "
                f"({mismatch_summary['mismatch_rate']:.4f}) | "
                f"near_0.5={mismatch_summary['near_threshold_mismatches']} | "
                f"mismatch_live_score={score_range}"
            ),
        )
    print(
        "PAD official metrics:",
        (
            f"threshold={selected_metrics['threshold']:.4f} | "
            f"FAR={selected_metrics['far']:.4f} | "
            f"FRR={selected_metrics['frr']:.4f} | "
            f"GAR={selected_metrics['gar']:.4f} | "
            f"APCER={selected_metrics['apcer']:.4f} | "
            f"BPCER={selected_metrics['bpcer']:.4f} | "
            f"ACER={selected_metrics['acer']:.4f}"
        ),
    )
    if eer_results.get("eer_value") is not None:
        print(
            "PAD EER:",
            (
                f"eer={eer_results['eer_value']:.4f} | "
                f"eer_threshold={eer_results['eer_threshold']:.4f}"
            ),
        )

    export_paths = metrics_report.get("export_paths")
    if export_paths:
        print("PAD metric exports:")
        print(f"- json: {export_paths['json']}")
        print(f"- summary_csv: {export_paths['summary_csv']}")
        if "roc_csv" in export_paths:
            print(f"- roc_csv: {export_paths['roc_csv']}")
        if "det_csv" in export_paths:
            print(f"- det_csv: {export_paths['det_csv']}")

    summary_rows = []
    for label in ("low", "mid", "high"):
        row = threshold_summary.get(label)
        if row is None:
            continue
        summary_rows.append(
            (
                f"{label}={row['threshold']:.4f} | "
                f"FAR={row['far']:.4f} | "
                f"FRR={row['frr']:.4f} | "
                f"GAR={row['gar']:.4f} | "
                f"ACER={(row['apcer'] + row['bpcer']) / 2.0:.4f}"
            )
        )

    if summary_rows:
        print("PAD threshold summary:")
        for row in summary_rows:
            print(f"- {row}")

    if log_live_false_negatives and sample_paths is not None:
        print_live_false_negative_analysis(
            dataset_name,
            y_test,
            y_pred,
            sample_paths,
            score_values,
            score_label,
            live_false_negative_limit,
        )
        print_spoof_false_positive_analysis(
            dataset_name,
            y_test,
            y_pred,
            sample_paths,
            score_values,
            score_label,
            live_false_negative_limit,
        )

    print(f"PAD evaluation completed in {evaluation_elapsed:.2f}s")
    print()


def compute_crop_diagnostics(sample_path):
    """
    Compute simple image diagnostics for a PAD crop.

    These are lightweight proxies for lighting, contrast, blur, and
    coarse asymmetry so false results can be inspected more
    usefully from the batch logs.
    """

    if sample_path is None:
        return None

    image = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "lap_var": float("nan"),
            "dark_ratio": float("nan"),
            "bright_ratio": float("nan"),
            "lr_delta": float("nan"),
            "tb_delta": float("nan"),
        }

    image_f = image.astype(np.float32)
    height, width = image.shape[:2]
    half_w = max(width // 2, 1)
    half_h = max(height // 2, 1)
    left = image_f[:, :half_w]
    right = image_f[:, half_w:]
    top = image_f[:half_h, :]
    bottom = image_f[half_h:, :]

    mean_value = float(image_f.mean())
    std_value = float(image_f.std())
    lap_var = float(cv2.Laplacian(image, cv2.CV_64F).var())
    dark_ratio = float((image_f < 80.0).mean())
    bright_ratio = float((image_f > 200.0).mean())
    lr_delta = float(abs(left.mean() - right.mean())) if right.size > 0 else 0.0
    tb_delta = float(abs(top.mean() - bottom.mean())) if bottom.size > 0 else 0.0

    return {
        "mean": mean_value,
        "std": std_value,
        "lap_var": lap_var,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "lr_delta": lr_delta,
        "tb_delta": tb_delta,
    }


def print_live_false_negative_analysis(dataset_name, y_test, y_pred, sample_paths, score_values, score_label, limit):
    """
    Print the live samples predicted as spoof so they can be inspected later.
    """

    if not sample_paths:
        return

    false_negative_indices = np.where((y_test == 1) & (y_pred == 0))[0]
    print(f"Live false negatives ({dataset_name}): {len(false_negative_indices)}")

    if len(false_negative_indices) == 0:
        return

    if score_values is not None:
        false_negative_indices = sorted(false_negative_indices, key=lambda idx: score_values[idx])

    selected_indices = false_negative_indices if limit is None else false_negative_indices[:limit]

    for rank, index in enumerate(selected_indices, start=1):
        line = f"[LIVE_FN {rank:02d}] path={sample_paths[index]}"
        if score_values is not None:
            line += f" | {score_label}={score_values[index]:.4f}"
        diagnostics = compute_crop_diagnostics(sample_paths[index])
        if diagnostics is not None:
            line += (
                f" | mean={diagnostics['mean']:.1f}"
                f" | std={diagnostics['std']:.1f}"
                f" | lap={diagnostics['lap_var']:.1f}"
                f" | dark={diagnostics['dark_ratio']:.2f}"
                f" | bright={diagnostics['bright_ratio']:.2f}"
                f" | lr={diagnostics['lr_delta']:.1f}"
                f" | tb={diagnostics['tb_delta']:.1f}"
            )
        print(line)

    if limit is not None and len(false_negative_indices) > limit:
        print(
            f"... truncated {len(false_negative_indices) - limit} additional live false negatives "
            f"(limit={limit})"
        )


def print_spoof_false_positive_analysis(
    dataset_name,
    y_test,
    y_pred,
    sample_paths,
    score_values,
    score_label,
    limit,
):
    """
    Print the spoof samples predicted as live so they can be inspected later.
    """

    if not sample_paths:
        return

    false_positive_indices = np.where((y_test == 0) & (y_pred == 1))[0]
    print(f"Spoof false positives ({dataset_name}): {len(false_positive_indices)}")

    if len(false_positive_indices) == 0:
        return

    if score_values is not None:
        false_positive_indices = sorted(false_positive_indices, key=lambda idx: score_values[idx], reverse=True)

    selected_indices = false_positive_indices if limit is None else false_positive_indices[:limit]

    for rank, index in enumerate(selected_indices, start=1):
        line = f"[SPOOF_FP {rank:02d}] path={sample_paths[index]}"
        if score_values is not None:
            line += f" | {score_label}={score_values[index]:.4f}"
        diagnostics = compute_crop_diagnostics(sample_paths[index])
        if diagnostics is not None:
            line += (
                f" | mean={diagnostics['mean']:.1f}"
                f" | std={diagnostics['std']:.1f}"
                f" | lap={diagnostics['lap_var']:.1f}"
                f" | dark={diagnostics['dark_ratio']:.2f}"
                f" | bright={diagnostics['bright_ratio']:.2f}"
                f" | lr={diagnostics['lr_delta']:.1f}"
                f" | tb={diagnostics['tb_delta']:.1f}"
            )
        print(line)

    if limit is not None and len(false_positive_indices) > limit:
        print(
            f"... truncated {len(false_positive_indices) - limit} additional spoof false positives "
            f"(limit={limit})"
        )


def evaluate_single_pad_dataset(
    eval_mode,
    scaler,
    classifier,
    log_live_false_negatives=False,
    live_false_negative_limit=LIVE_FALSE_NEGATIVE_LOG_LIMIT,
    output_stem=None,
    export_curves=False,
):
    """
    Evaluate the trained PAD model on one direct evaluation split.

    This path is used for the plain public test modes and does not run
    an additional threshold-selection phase beforehand.
    """

    load_start_time = time.perf_counter()
    dataset_name, dataset = load_configured_test_dataset(
        eval_mode,
        include_paths=log_live_false_negatives and eval_mode == "custom",
        collect_pose_metrics=True,
    )
    load_elapsed = time.perf_counter() - load_start_time

    if log_live_false_negatives and eval_mode == "custom":
        X_test, y_test, sample_paths, pose_summary = dataset
    else:
        X_test, y_test, pose_summary = dataset
        sample_paths = None

    if len(y_test) == 0:
        print(f"PAD evaluation dataset: {dataset_name}")
        print("No valid samples found for evaluation.")
        return

    evaluation_start_time = time.perf_counter()
    X_test_scaled = scaler.transform(X_test)
    y_pred = classifier.predict(X_test_scaled)
    score_values, score_label = get_live_score_details(classifier, X_test_scaled)
    if score_values is None:
        score_values = y_pred.astype(np.float64)
        score_label = "prediction_fallback"

    metrics_report = compute_pad_evaluation_metrics(
        y_test,
        score_values,
        score_label,
        output_stem=output_stem,
        export_curves=export_curves,
    )

    log_pad_evaluation_diagnostics(
        dataset_name,
        y_test,
        y_pred,
        score_values,
        score_label,
        sample_paths if log_live_false_negatives and eval_mode == "custom" else None,
        load_elapsed,
        time.perf_counter() - evaluation_start_time,
        metrics_report,
        mismatch_summary=build_prediction_probability_mismatch_summary(
            y_pred,
            score_values,
            score_label,
        ),
        pose_summary=pose_summary,
        log_live_false_negatives=log_live_false_negatives and eval_mode == "custom",
        live_false_negative_limit=live_false_negative_limit,
    )


def evaluate_pad_validation_protocol(
    validation_mode,
    test_mode,
    protocol_label,
    scaler,
    classifier,
    output_stem=None,
    target_apcer=PAD_VALIDATION_TARGET_APCER,
    export_curves=False,
):
    """
    Evaluate one PAD dataset with a side validation protocol.

    The validation split selects the live-score threshold under a target
    APCER constraint; the final test
    split is then evaluated with that frozen threshold.
    """

    validation_load_start_time = time.perf_counter()
    validation_name, validation_dataset = load_configured_test_dataset(
        validation_mode,
        collect_pose_metrics=True,
    )
    validation_load_elapsed = time.perf_counter() - validation_load_start_time
    X_validation, y_validation, validation_pose_summary = validation_dataset

    if len(y_validation) == 0:
        print(f"PAD validation dataset: {validation_name}")
        print("No valid validation samples found for threshold selection.")
        return

    validation_start_time = time.perf_counter()
    X_validation_scaled = scaler.transform(X_validation)
    y_validation_pred = classifier.predict(X_validation_scaled)
    validation_scores, validation_score_label = get_live_score_details(
        classifier,
        X_validation_scaled,
    )
    if validation_scores is None:
        validation_scores = y_validation_pred.astype(np.float64)
        validation_score_label = "prediction_fallback"

    selected_target_apcer = (
        PAD_VALIDATION_TARGET_APCER
        if target_apcer is None
        else float(target_apcer)
    )
    validation_sweep = sweep_pad_thresholds(y_validation, validation_scores)
    threshold_selection = select_pad_threshold_by_target_apcer(
        validation_sweep,
        target_apcer=selected_target_apcer,
    )
    selected_threshold = threshold_selection["threshold"]
    if selected_threshold is None:
        selected_threshold = 0.5 if validation_score_label != "decision" else 0.0

    print(f"{protocol_label} validation selected threshold:", selected_threshold)
    print(
        f"{protocol_label} validation threshold rule:",
        threshold_selection["threshold_rule"],
    )
    print(f"{protocol_label} validation target APCER:", selected_target_apcer)
    print(
        f"{protocol_label} validation APCER constraint satisfied:",
        threshold_selection["constraint_satisfied"],
    )
    selected_validation_metrics = threshold_selection.get("selected_metrics")
    if selected_validation_metrics is not None:
        print(
            f"{protocol_label} validation selected metrics:",
            (
                f"APCER={selected_validation_metrics['apcer']:.4f} | "
                f"BPCER={selected_validation_metrics['bpcer']:.4f} | "
                f"ACER={selected_validation_metrics['acer']:.4f}"
            ),
        )

    validation_report = compute_pad_evaluation_metrics(
        y_validation,
        validation_scores,
        validation_score_label,
        selected_threshold=selected_threshold,
        threshold_source=f"{validation_mode}_target_apcer_selection",
        threshold_selection=threshold_selection,
        output_stem=(
            f"{output_stem}_selection"
            if output_stem is not None
            else f"{validation_mode}_selection"
        ),
        export_curves=export_curves,
    )
    log_pad_evaluation_diagnostics(
        validation_name,
        y_validation,
        y_validation_pred,
        validation_scores,
        validation_score_label,
        None,
        validation_load_elapsed,
        time.perf_counter() - validation_start_time,
        validation_report,
        mismatch_summary=build_prediction_probability_mismatch_summary(
            y_validation_pred,
            validation_scores,
            validation_score_label,
        ),
        pose_summary=validation_pose_summary,
    )

    test_load_start_time = time.perf_counter()
    test_name, test_dataset = load_configured_test_dataset(
        test_mode,
        collect_pose_metrics=True,
    )
    test_load_elapsed = time.perf_counter() - test_load_start_time
    X_test, y_test, test_pose_summary = test_dataset

    if len(y_test) == 0:
        print(f"PAD evaluation dataset: {test_name}")
        print("No valid samples found for final evaluation.")
        return

    test_start_time = time.perf_counter()
    X_test_scaled = scaler.transform(X_test)
    y_test_pred = classifier.predict(X_test_scaled)
    test_scores, test_score_label = get_live_score_details(classifier, X_test_scaled)
    if test_scores is None:
        test_scores = y_test_pred.astype(np.float64)
        test_score_label = "prediction_fallback"

    test_report = compute_pad_evaluation_metrics(
        y_test,
        test_scores,
        test_score_label,
        selected_threshold=selected_threshold,
        threshold_source=f"{validation_mode}_target_apcer",
        threshold_selection=threshold_selection,
        output_stem=output_stem,
        export_curves=export_curves,
    )

    log_pad_evaluation_diagnostics(
        f"{test_name} (threshold from {protocol_label} validation)",
        y_test,
        y_test_pred,
        test_scores,
        test_score_label,
        None,
        test_load_elapsed,
        time.perf_counter() - test_start_time,
        test_report,
        mismatch_summary=build_prediction_probability_mismatch_summary(
            y_test_pred,
            test_scores,
            test_score_label,
        ),
        pose_summary=test_pose_summary,
    )


def evaluate_custom_pad_protocol(
    scaler,
    classifier,
    pad_config,
    output_stem=None,
    target_apcer=PAD_VALIDATION_TARGET_APCER,
    export_curves=False,
    log_live_false_negatives=False,
    live_false_negative_limit=LIVE_FALSE_NEGATIVE_LOG_LIMIT,
):
    """
    Evaluate custom PAD by selecting the threshold on a deterministic
    pseudo-validation subset extracted from the custom train split.

    Unlike the public datasets, custom PAD does not have an explicit
    validation folder on disk. The threshold is therefore selected on
    an internal holdout carved out of the custom training split, while
    the custom test split remains untouched until final evaluation.
    """

    validation_ratio = float(
        pad_config.get("custom_validation_ratio", CUSTOM_PAD_VALIDATION_RATIO)
    )
    validation_seed = int(
        pad_config.get("custom_validation_seed", CUSTOM_PAD_VALIDATION_SEED)
    )

    print(
        "Custom PAD pseudo-validation:",
        f"enabled | ratio={validation_ratio} | seed={validation_seed}",
    )

    validation_load_start_time = time.perf_counter()
    X_validation, y_validation, validation_pose_summary = load_custom_pad_validation_dataset(
        validation_ratio=validation_ratio,
        validation_seed=validation_seed,
        collect_pose_metrics=True,
        return_pose_summary=True,
    )
    validation_load_elapsed = time.perf_counter() - validation_load_start_time

    if len(y_validation) == 0:
        print("PAD validation dataset: Custom pseudo-validation")
        print("No valid validation samples found for threshold selection.")
        return

    validation_start_time = time.perf_counter()
    X_validation_scaled = scaler.transform(X_validation)
    y_validation_pred = classifier.predict(X_validation_scaled)
    validation_scores, validation_score_label = get_live_score_details(
        classifier,
        X_validation_scaled,
    )
    if validation_scores is None:
        validation_scores = y_validation_pred.astype(np.float64)
        validation_score_label = "prediction_fallback"

    selected_target_apcer = (
        PAD_VALIDATION_TARGET_APCER
        if target_apcer is None
        else float(target_apcer)
    )
    validation_sweep = sweep_pad_thresholds(y_validation, validation_scores)
    threshold_selection = select_pad_threshold_by_target_apcer(
        validation_sweep,
        target_apcer=selected_target_apcer,
    )
    selected_threshold = threshold_selection["threshold"]
    if selected_threshold is None:
        selected_threshold = 0.5 if validation_score_label != "decision" else 0.0

    print("Custom PAD validation selected threshold:", selected_threshold)
    print(
        "Custom PAD validation threshold rule:",
        threshold_selection["threshold_rule"],
    )
    print("Custom PAD validation target APCER:", selected_target_apcer)
    print(
        "Custom PAD validation APCER constraint satisfied:",
        threshold_selection["constraint_satisfied"],
    )
    selected_validation_metrics = threshold_selection.get("selected_metrics")
    if selected_validation_metrics is not None:
        print(
            "Custom PAD validation selected metrics:",
            (
                f"APCER={selected_validation_metrics['apcer']:.4f} | "
                f"BPCER={selected_validation_metrics['bpcer']:.4f} | "
                f"ACER={selected_validation_metrics['acer']:.4f}"
            ),
        )

    validation_report = compute_pad_evaluation_metrics(
        y_validation,
        validation_scores,
        validation_score_label,
        selected_threshold=selected_threshold,
        threshold_source="custom_internal_validation_target_apcer_selection",
        threshold_selection=threshold_selection,
        output_stem=(
            f"{output_stem}_selection"
            if output_stem is not None
            else "custom_selection"
        ),
        export_curves=export_curves,
    )
    log_pad_evaluation_diagnostics(
        "Custom pseudo-validation",
        y_validation,
        y_validation_pred,
        validation_scores,
        validation_score_label,
        None,
        validation_load_elapsed,
        time.perf_counter() - validation_start_time,
        validation_report,
        mismatch_summary=build_prediction_probability_mismatch_summary(
            y_validation_pred,
            validation_scores,
            validation_score_label,
        ),
        pose_summary=validation_pose_summary,
    )

    test_load_start_time = time.perf_counter()
    test_dataset = load_custom_pad_test_dataset(
        return_image_paths=log_live_false_negatives,
        collect_pose_metrics=True,
        return_pose_summary=True,
    )
    test_load_elapsed = time.perf_counter() - test_load_start_time
    if log_live_false_negatives:
        X_test, y_test, sample_paths, test_pose_summary = test_dataset
    else:
        X_test, y_test, test_pose_summary = test_dataset
        sample_paths = None

    if len(y_test) == 0:
        print("PAD evaluation dataset: Custom test")
        print("No valid samples found for final evaluation.")
        return

    test_start_time = time.perf_counter()
    X_test_scaled = scaler.transform(X_test)
    y_test_pred = classifier.predict(X_test_scaled)
    test_scores, test_score_label = get_live_score_details(classifier, X_test_scaled)
    if test_scores is None:
        test_scores = y_test_pred.astype(np.float64)
        test_score_label = "prediction_fallback"

    test_report = compute_pad_evaluation_metrics(
        y_test,
        test_scores,
        test_score_label,
        selected_threshold=selected_threshold,
        threshold_source="custom_internal_validation_target_apcer",
        threshold_selection=threshold_selection,
        output_stem=output_stem,
        export_curves=export_curves,
    )
    log_pad_evaluation_diagnostics(
        "Custom test (threshold from custom pseudo-validation)",
        y_test,
        y_test_pred,
        test_scores,
        test_score_label,
        sample_paths if log_live_false_negatives else None,
        test_load_elapsed,
        time.perf_counter() - test_start_time,
        test_report,
        mismatch_summary=build_prediction_probability_mismatch_summary(
            y_test_pred,
            test_scores,
            test_score_label,
        ),
        pose_summary=test_pose_summary,
        log_live_false_negatives=log_live_false_negatives,
        live_false_negative_limit=live_false_negative_limit,
    )


def _build_target_apcer_suffix(target_apcer):
    """
    Build a compact output suffix for non-default APCER targets.
    """

    target_tag = f"{float(target_apcer):.4f}".rstrip("0").rstrip(".")
    target_tag = target_tag.replace(".", "p")
    return f"_apcer{target_tag}"


def evaluate_pad_model(
    eval_mode=None,
    artifact_profile="runtime",
    log_live_false_negatives=False,
    live_false_negative_limit=LIVE_FALSE_NEGATIVE_LOG_LIMIT,
    pad_validation_target_apcer=None,
    export_curves=False,
):
    """
    Evaluate the trained PAD model on one dataset or on all supported
    datasets sequentially.

    ``artifact_profile`` selects which saved PAD artifacts must be
    evaluated. This keeps fast and full experiment outputs separate
    while preserving a canonical runtime artifact set. Validation modes
    run threshold selection before the final test; plain test modes
    evaluate the selected dataset directly.
    """

    selected_mode = eval_mode or EVAL_MODE
    total_start_time = time.perf_counter()

    artifact_paths = get_pad_artifact_paths(artifact_profile)
    scaler = joblib.load(artifact_paths["scaler_path"])
    classifier = joblib.load(artifact_paths["model_path"])
    expected_feature_dim = get_expected_feature_dim_from_scaler(scaler)
    pad_config = load_pad_model_config(
        expected_feature_dim=expected_feature_dim,
        config_path=artifact_paths["config_path"],
    )
    configure_pad_feature_extractor_from_config(pad_config)

    print("PAD artifact profile:", artifact_profile)
    print(
        "PAD evaluation feature setup:",
        (
            f"debug={pad_config['pad_debug_feature_mode']} | "
            f"lft={pad_config['pad_lft_mode']} | "
            f"tgsd_tau={pad_config.get('tgsd_tau')} | "
            f"tgsd_top_percent={pad_config.get('tgsd_top_percent')} | "
            f"tgsd_grad={pad_config.get('tgsd_gradient_operator')} | "
            f"feature_dim={pad_config['feature_dim']}"
        ),
    )
    if log_live_false_negatives:
        print("PAD live false-negative logging:", f"enabled | limit={live_false_negative_limit}")
    else:
        print("PAD live false-negative logging: disabled")

    if selected_mode == "all":
        for mode in ("casia", "replay"):
            evaluate_single_pad_dataset(
                mode,
                scaler,
                classifier,
                log_live_false_negatives=log_live_false_negatives,
                live_false_negative_limit=live_false_negative_limit,
                output_stem=f"{artifact_profile}_{mode}",
                export_curves=export_curves,
            )
        evaluate_custom_pad_protocol(
            scaler,
            classifier,
            pad_config,
            output_stem=f"{artifact_profile}_custom",
            target_apcer=pad_validation_target_apcer,
            export_curves=export_curves,
            log_live_false_negatives=log_live_false_negatives,
            live_false_negative_limit=live_false_negative_limit,
        )
        print(f"Total PAD evaluation time: {time.perf_counter() - total_start_time:.2f}s")
        return

    validation_modes = {
        "casia_validation": ("casia", "CASIA"),
        "replay_validation": ("replay", "Replay-Attack"),
    }
    if selected_mode in validation_modes:
        test_mode, protocol_label = validation_modes[selected_mode]
        output_stem = f"{artifact_profile}_{selected_mode}"
        if pad_validation_target_apcer is not None:
            output_stem += _build_target_apcer_suffix(pad_validation_target_apcer)
        evaluate_pad_validation_protocol(
            selected_mode,
            test_mode,
            protocol_label,
            scaler,
            classifier,
            output_stem=output_stem,
            target_apcer=pad_validation_target_apcer,
            export_curves=export_curves,
        )
        print(f"Total PAD evaluation time: {time.perf_counter() - total_start_time:.2f}s")
        return

    if selected_mode == "custom":
        output_stem = f"{artifact_profile}_{selected_mode}"
        if pad_validation_target_apcer is not None:
            output_stem += _build_target_apcer_suffix(pad_validation_target_apcer)
        evaluate_custom_pad_protocol(
            scaler,
            classifier,
            pad_config,
            output_stem=output_stem,
            target_apcer=pad_validation_target_apcer,
            export_curves=export_curves,
            log_live_false_negatives=log_live_false_negatives,
            live_false_negative_limit=live_false_negative_limit,
        )
        print(f"Total PAD evaluation time: {time.perf_counter() - total_start_time:.2f}s")
        return

    evaluate_single_pad_dataset(
        selected_mode,
        scaler,
        classifier,
        log_live_false_negatives=log_live_false_negatives,
        live_false_negative_limit=live_false_negative_limit,
        output_stem=f"{artifact_profile}_{selected_mode}",
        export_curves=export_curves,
    )
    print(f"Total PAD evaluation time: {time.perf_counter() - total_start_time:.2f}s")


if __name__ == "__main__":
    evaluate_pad_model()
