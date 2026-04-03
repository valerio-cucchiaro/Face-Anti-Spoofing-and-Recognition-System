"""
Module: pad_model_evaluator

Description:
This module evaluates the performance of the trained presentation
attack detection (PAD) classifier on one or more selected evaluation
datasets.

The script can evaluate the trained model on:
- CASIA test
- Replay-Attack test
- custom runtime PAD test

It can also run the three evaluations sequentially while keeping the
output of each dataset separated, so the behavior remains equivalent to
running each evaluation mode individually.

Evaluation metrics such as accuracy, precision, recall, F1 score,
and the confusion matrix are computed to assess the ability of the
model to distinguish between live faces and spoof attacks.
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
from src.antispoofing.training.pad_dataset_loader import (
    load_pad_test_dataset_casia,
    load_pad_test_dataset_replay,
    load_pad_split_dataset,
    RUNTIME_TEST_LIVE_PATH,
    RUNTIME_TEST_SPOOF_PATH,
)


EVAL_MODE = "runtime"     # casia, replay, runtime
LIVE_FALSE_NEGATIVE_LOG_LIMIT = 20


def load_runtime_pad_test_dataset(return_image_paths=False):
    """
    Load the custom runtime PAD test dataset.
    """

    return load_pad_split_dataset(
        RUNTIME_TEST_LIVE_PATH,
        RUNTIME_TEST_SPOOF_PATH,
        balance=False,
        dataset_type="final_pad_crop",
        return_image_paths=return_image_paths,
    )


def load_configured_test_dataset(eval_mode, include_paths=False):
    """
    Load the PAD evaluation dataset selected by ``eval_mode``.

    Supported modes:
    - casia
    - replay
    - runtime
    """

    if eval_mode == "casia":
        return "CASIA test", load_pad_test_dataset_casia()

    if eval_mode == "replay":
        return "Replay-Attack test", load_pad_test_dataset_replay()

    if eval_mode == "runtime":
        return "Runtime test", load_runtime_pad_test_dataset(return_image_paths=include_paths)

    raise ValueError(
        f"Unsupported EVAL_MODE: {eval_mode}. Use 'casia', 'replay', 'runtime', or 'all'."
    )


def get_live_score_details(classifier, X_test_scaled):
    """
    Compute per-sample live confidence details when supported by the classifier.
    """

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(X_test_scaled)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1], "live_prob"

    if hasattr(classifier, "decision_function"):
        decision = classifier.decision_function(X_test_scaled)
        decision = np.asarray(decision, dtype=np.float64).reshape(-1)
        return decision, "decision"

    return None, None


def compute_runtime_crop_diagnostics(sample_path):
    """
    Compute simple image diagnostics for a PAD crop.

    These are lightweight proxies for lighting, contrast, blur, and
    coarse asymmetry so runtime false results can be inspected more
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
        diagnostics = compute_runtime_crop_diagnostics(sample_paths[index])
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
        diagnostics = compute_runtime_crop_diagnostics(sample_paths[index])
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
):
    """
    Evaluate the trained PAD model on one selected PAD test dataset.
    """

    load_start_time = time.perf_counter()
    dataset_name, dataset = load_configured_test_dataset(
        eval_mode,
        include_paths=log_live_false_negatives and eval_mode == "runtime",
    )
    load_elapsed = time.perf_counter() - load_start_time

    if log_live_false_negatives and eval_mode == "runtime":
        X_test, y_test, sample_paths = dataset
    else:
        X_test, y_test = dataset
        sample_paths = None

    if len(y_test) == 0:
        print(f"PAD evaluation dataset: {dataset_name}")
        print("No valid samples found for evaluation.")
        return

    evaluation_start_time = time.perf_counter()
    X_test_scaled = scaler.transform(X_test)
    y_pred = classifier.predict(X_test_scaled)
    score_values, score_label = get_live_score_details(classifier, X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"PAD evaluation dataset: {dataset_name}")
    print(
        "PAD evaluation samples:",
        (
            f"total={len(y_test)} | "
            f"live={(y_test == 1).sum()} | spoof={(y_test == 0).sum()} | "
            f"pred_live={(y_pred == 1).sum()} | pred_spoof={(y_pred == 0).sum()}"
        ),
    )
    print(f"PAD evaluation dataset loading completed in {load_elapsed:.2f}s")
    print(
        "PAD metrics:",
        f"accuracy={accuracy:.4f} | precision={precision:.4f} | recall={recall:.4f} | f1={f1:.4f}",
    )
    print("PAD confusion matrix:\n", conf_matrix)

    if log_live_false_negatives and eval_mode == "runtime":
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
    print(f"PAD evaluation completed in {time.perf_counter() - evaluation_start_time:.2f}s")
    print()


def evaluate_pad_model(
    eval_mode=None,
    artifact_profile="runtime",
    log_live_false_negatives=False,
    live_false_negative_limit=LIVE_FALSE_NEGATIVE_LOG_LIMIT,
):
    """
    Evaluate the trained PAD model on one dataset or on all supported
    datasets sequentially.

    ``artifact_profile`` selects which saved PAD artifacts must be
    evaluated. This keeps fast and full experiment outputs separate
    while preserving a canonical runtime artifact set.
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
        for mode in ("casia", "replay", "runtime"):
            evaluate_single_pad_dataset(
                mode,
                scaler,
                classifier,
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
    )
    print(f"Total PAD evaluation time: {time.perf_counter() - total_start_time:.2f}s")


if __name__ == "__main__":
    evaluate_pad_model()
