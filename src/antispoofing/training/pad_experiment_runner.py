"""
Module: pad_experiment_runner

Description:
This module runs a complete PAD experiment by executing training
followed by evaluation with the currently selected dataset settings.

Training and evaluation modes are configured independently because the
project may train on one dataset combination and evaluate on another.
"""

import argparse
import time

from src.antispoofing.pad_feature_extractor import configure_pad_feature_modes
from src.antispoofing.training.pad_model_trainer import train_pad_model
from src.antispoofing.training.pad_model_evaluator import evaluate_pad_model

# ----- EXPERIMENT CONFIGURATION ----- #
TRAIN_MODE = "combined_runtime"
EVAL_MODE = "runtime"

# ----- SPEED / SAMPLING PRESET ----- #
EXPERIMENT_SPEED = "fast"   # full or fast

# ----- OPTIONAL OVERRIDES ----- #
RUNTIME_REPEAT = None
CASIA_BALANCE = None
REPLAY_BALANCE = None


def run(
    train_mode=None,
    eval_mode=None,
    experiment_speed=None,
    runtime_repeat=None,
    casia_balance=None,
    replay_balance=None,
    pad_debug_feature_mode=None,
    pad_lft_mode=None,
):
    """
    Execute a full PAD experiment using the current training
    and evaluation configuration.

    Supported training modes:
    - casia
    - replay
    - combined
    - runtime
    - combined_runtime

    Supported evaluation modes:
    - casia
    - replay
    - runtime
    - all

    Supported experiment speeds:
    - full
    - fast

    If runtime_repeat is set, it overrides the default runtime weight
    associated with the selected speed preset.

    If casia_balance or replay_balance is set, it overrides the
    corresponding pre-extraction balancing flag associated with the
    selected speed preset.

    pad_debug_feature_mode and pad_lft_mode can be used to switch the
    PAD feature extractor configuration for controlled ablation runs.
    """
    selected_train_mode = train_mode or TRAIN_MODE
    selected_eval_mode = eval_mode or EVAL_MODE
    selected_speed = experiment_speed or EXPERIMENT_SPEED
    selected_runtime_repeat = runtime_repeat if runtime_repeat is not None else RUNTIME_REPEAT
    selected_casia_balance = casia_balance if casia_balance is not None else CASIA_BALANCE
    selected_replay_balance = replay_balance if replay_balance is not None else REPLAY_BALANCE
    total_start_time = time.perf_counter()
    artifact_profile = selected_speed
    update_runtime_artifacts = selected_speed == "full"

    configure_pad_feature_modes(
        debug_mode=pad_debug_feature_mode,
        lft_mode=pad_lft_mode,
    )

    print(f"=== EXPERIMENT SPEED ({selected_speed}) ===")
    if selected_runtime_repeat is not None:
        print(f"=== RUNTIME REPEAT ({selected_runtime_repeat}) ===")
    if selected_casia_balance is not None:
        print(f"=== CASIA PRE-EXTRACTION BALANCE ({selected_casia_balance}) ===")
    if selected_replay_balance is not None:
        print(f"=== REPLAY PRE-EXTRACTION BALANCE ({selected_replay_balance}) ===")
    print(f"=== TRAINING ({selected_train_mode}) ===")
    train_pad_model(
        selected_train_mode,
        experiment_speed=selected_speed,
        runtime_repeat_override=selected_runtime_repeat,
        casia_balance_override=selected_casia_balance,
        replay_balance_override=selected_replay_balance,
        artifact_profile=artifact_profile,
        update_runtime_artifacts=update_runtime_artifacts,
    )

    print(f"=== EVALUATION ({selected_eval_mode}) ===")
    evaluate_pad_model(
        selected_eval_mode,
        artifact_profile=artifact_profile,
    )
    print(f"=== TOTAL PAD EXPERIMENT TIME: {time.perf_counter() - total_start_time:.2f}s ===")


def build_arg_parser():
    """
    Build the command-line interface for PAD experiments.

    The CLI exposes the runner parameters needed for scripted PAD
    campaigns, including the feature-ablation switches used in
    `antispoofing.md`.
    """
    parser = argparse.ArgumentParser(
        description="Run a PAD experiment with optional sampling and feature overrides."
    )
    parser.add_argument("--train-mode", choices=["casia", "replay", "combined", "runtime", "combined_runtime"])
    parser.add_argument("--eval-mode", choices=["casia", "replay", "runtime", "all"])
    parser.add_argument("--experiment-speed", choices=["full", "fast"])
    parser.add_argument("--runtime-repeat", type=int)
    parser.add_argument("--casia-balance", choices=["true", "false"])
    parser.add_argument("--replay-balance", choices=["true", "false"])
    parser.add_argument("--pad-debug-feature-mode", choices=["baseline", "tgsd", "full"])
    parser.add_argument("--pad-lft-mode", choices=["summary", "ggd_light"])
    return parser


def parse_optional_bool(value):
    """
    Convert an optional CLI boolean string into ``True`` or ``False``.
    """
    if value is None:
        return None
    return value.lower() == "true"


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    run(
        train_mode=cli_args.train_mode,
        eval_mode=cli_args.eval_mode,
        experiment_speed=cli_args.experiment_speed,
        runtime_repeat=cli_args.runtime_repeat,
        casia_balance=parse_optional_bool(cli_args.casia_balance),
        replay_balance=parse_optional_bool(cli_args.replay_balance),
        pad_debug_feature_mode=cli_args.pad_debug_feature_mode,
        pad_lft_mode=cli_args.pad_lft_mode,
    )
