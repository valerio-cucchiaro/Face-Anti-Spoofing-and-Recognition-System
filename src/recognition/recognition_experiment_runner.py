"""
Module: recognition_experiment_runner

Description:
This module runs a complete recognition experiment by executing
training followed by evaluation with the currently selected
recognition dataset settings.

Training and evaluation modes are configured independently so the
project can train one model and then evaluate it on one dataset or on
all supported probe datasets.

The mode vocabulary is shared across training and evaluation:
- `custom`, `orl`, `lfw`, `multipie`: single-dataset runs
- `all`: full combined setup including LFW and Multi-PIE

LBPH hyperparameters can also be passed to the trainer through this
runner so small tuning sweeps can reuse the same execution path.
"""

import argparse
import time

from src.preprocessing.face_preprocessing import (
    configure_recognition_preprocessing,
    get_recognition_preprocessing_config,
)
from src.recognition.recognition_model_trainer import train_recognition_model
from src.evaluation.recognition_model_evaluator import evaluate_recognition_model


# ----- CONSTANTS ----- #
TRAIN_MODE = "custom"      # options: custom, orl, lfw, multipie, all
EVAL_MODE = "custom"       # options: custom, orl, lfw, multipie, all


def run(
    train_mode=None,
    eval_mode=None,
    crop_padding_ratio=None,
    lfw_sample_ratio=None,
    quality_filter_mode="report_only",
    export_curves=False,
    debug_recognition=False,
    debug_quality=False,
    save_system_threshold=False,
    enable_reliability_diagnostics=False,
):
    """
    Execute one full recognition experiment.

    The runner always trains first and then evaluates the model
    produced by that same run.
    """
    return run_with_lbph_params(
        train_mode=train_mode,
        eval_mode=eval_mode,
        lbph_params=None,
        crop_padding_ratio=crop_padding_ratio,
        lfw_sample_ratio=lfw_sample_ratio,
        quality_filter_mode=quality_filter_mode,
        export_curves=export_curves,
        debug_recognition=debug_recognition,
        debug_quality=debug_quality,
        save_system_threshold=save_system_threshold,
        enable_reliability_diagnostics=enable_reliability_diagnostics,
    )


def run_with_lbph_params(
    train_mode=None,
    eval_mode=None,
    lbph_params=None,
    crop_padding_ratio=None,
    lfw_sample_ratio=None,
    quality_filter_mode="report_only",
    export_curves=False,
    debug_recognition=False,
    debug_quality=False,
    save_system_threshold=False,
    enable_reliability_diagnostics=False,
):
    """
    Execute a full recognition experiment with optional LBPH overrides.

    ``lbph_params`` may contain one or more of:
    - `radius`
    - `neighbors`
    - `grid_x`
    - `grid_y`
    """
    selected_train_mode = train_mode or TRAIN_MODE
    selected_eval_mode = eval_mode or EVAL_MODE
    total_start_time = time.perf_counter()
    selected_lfw_sample_ratio = 1.0 if lfw_sample_ratio is None else lfw_sample_ratio

    configure_recognition_preprocessing(
        crop_padding_ratio=crop_padding_ratio,
    )
    preprocessing_config = get_recognition_preprocessing_config()

    print(
        "=== RECOGNITION PREPROCESSING "
        f"(normalization={preprocessing_config['normalization_mode']}, "
        f"padding_ratio={preprocessing_config['crop_padding_ratio']:.3f}) ==="
    )
    if selected_lfw_sample_ratio < 1.0:
        print(f"=== LFW SAMPLING (sample_ratio={selected_lfw_sample_ratio:.3f}) ===")
    print(f"=== TRAINING ({selected_train_mode}) ===")
    train_recognition_model(
        selected_train_mode,
        lbph_params=lbph_params,
        lfw_sample_ratio=selected_lfw_sample_ratio,
        quality_filter_mode=quality_filter_mode,
        debug_quality=debug_quality,
    )

    print(f"=== EVALUATION ({selected_eval_mode}) ===")
    evaluate_recognition_model(
        selected_train_mode,
        selected_eval_mode,
        lfw_sample_ratio=selected_lfw_sample_ratio,
        quality_filter_mode=quality_filter_mode,
        export_curves=export_curves,
        debug_recognition=debug_recognition,
        debug_quality=debug_quality,
        save_system_threshold=save_system_threshold,
        enable_reliability_diagnostics=enable_reliability_diagnostics,
    )
    print(f"=== TOTAL EXPERIMENT TIME: {time.perf_counter() - total_start_time:.2f}s ===")


def build_arg_parser():
    """
    Build the command-line interface for the recognition experiment runner.

    The CLI mirrors ``run_with_lbph_params`` so batch scripts can launch
    recognition experiments without editing this file for every run.
    """
    parser = argparse.ArgumentParser(
        description="Run a recognition experiment with optional LBPH overrides."
    )
    parser.add_argument(
        "--train-mode",
        choices=["custom", "orl", "lfw", "multipie", "all"],
    )
    parser.add_argument(
        "--eval-mode",
        choices=["custom", "orl", "lfw", "multipie", "all"],
    )
    parser.add_argument("--radius", type=int)
    parser.add_argument("--neighbors", type=int)
    parser.add_argument("--grid-x", type=int)
    parser.add_argument("--grid-y", type=int)
    parser.add_argument("--preprocess-padding-ratio", type=float)
    parser.add_argument("--lfw-sample-ratio", type=float)
    parser.add_argument(
        "--quality-filter-mode",
        choices=[
            "none",
            "report_only",
            "probe_only",
        ],
        default="report_only",
    )
    parser.add_argument("--export-curves", action="store_true")
    parser.add_argument("--debug-recognition", action="store_true")
    parser.add_argument("--debug-quality", action="store_true")
    parser.add_argument(
        "--enable-reliability-diagnostics",
        action="store_true",
        help=(
            "Enable optional offline SRR/margin diagnostics. These reports are "
            "not used by the live runtime decision and are disabled by default."
        ),
    )
    parser.add_argument(
        "--save-system-threshold",
        action="store_true",
        help=(
            "Save the custom-to-custom recognition threshold for offline system "
            "evaluation."
        ),
    )
    return parser


def lbph_params_from_args(args):
    """
    Convert parsed CLI arguments into an optional LBPH override dict.
    """
    lbph_params = {
        key: value
        for key, value in {
            "radius": args.radius,
            "neighbors": args.neighbors,
            "grid_x": args.grid_x,
            "grid_y": args.grid_y,
        }.items()
        if value is not None
    }
    return lbph_params or None


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    run_with_lbph_params(
        train_mode=cli_args.train_mode,
        eval_mode=cli_args.eval_mode,
        lbph_params=lbph_params_from_args(cli_args),
        crop_padding_ratio=cli_args.preprocess_padding_ratio,
        lfw_sample_ratio=cli_args.lfw_sample_ratio,
        quality_filter_mode=cli_args.quality_filter_mode,
        export_curves=cli_args.export_curves,
        debug_recognition=cli_args.debug_recognition,
        debug_quality=cli_args.debug_quality,
        save_system_threshold=cli_args.save_system_threshold,
        enable_reliability_diagnostics=cli_args.enable_reliability_diagnostics,
    )
