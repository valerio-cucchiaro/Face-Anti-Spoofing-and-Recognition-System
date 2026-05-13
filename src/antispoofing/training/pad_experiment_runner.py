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
from src.evaluation.pad_model_evaluator import evaluate_pad_model

# ----- EXPERIMENT CONFIGURATION ----- #
TRAIN_MODE = "custom"
EVAL_MODE = "custom"

# ----- SPEED / SAMPLING PRESET ----- #
EXPERIMENT_SPEED = "full"   # full or fast

# ----- OPTIONAL OVERRIDES ----- #
CUSTOM_REPEAT = None
CASIA_BALANCE = None
REPLAY_BALANCE = None


def run(
    train_mode=None,
    eval_mode=None,
    experiment_speed=None,
    custom_repeat=None,
    casia_balance=None,
    replay_balance=None,
    pad_debug_feature_mode=None,
    pad_lft_mode=None,
    tgsd_tau=None,
    tgsd_top_percent=None,
    tgsd_gradient_operator=None,
    log_live_false_negatives=False,
    live_false_negative_limit=None,
    augment_live_photometric=False,
    augment_live_photometric_mode=None,
    augment_live_photometric_scope="all",
    export_curves=False,
):
    """
    Execute a full PAD experiment using the current training
    and evaluation configuration.

    Supported training modes:
    - casia
    - replay
    - combined
    - custom
    - combined_custom

    Supported evaluation modes:
    - casia
    - replay
    - custom
    - all

    Supported experiment speeds:
    - full
    - fast

    If custom_repeat is set, it overrides the default custom repeat
    associated with the selected speed preset.

    If casia_balance or replay_balance is set, it overrides the
    corresponding pre-extraction balancing flag associated with the
    selected speed preset.

    pad_debug_feature_mode can be used to switch between the promoted
    baseline and TGSD feature branches. pad_lft_mode is kept explicit
    so the runner can persist the active PAD config coherently, even
    though the promoted LFT branch is now fixed to ``ggd_light``.
    """
    selected_train_mode = train_mode or TRAIN_MODE
    selected_eval_mode = eval_mode or EVAL_MODE
    selected_speed = experiment_speed or EXPERIMENT_SPEED
    selected_custom_repeat = custom_repeat if custom_repeat is not None else CUSTOM_REPEAT
    selected_casia_balance = casia_balance if casia_balance is not None else CASIA_BALANCE
    selected_replay_balance = replay_balance if replay_balance is not None else REPLAY_BALANCE
    total_start_time = time.perf_counter()
    artifact_profile = selected_speed
    update_runtime_artifacts = selected_speed == "full"

    configure_pad_feature_modes(
        debug_mode=pad_debug_feature_mode,
        lft_mode=pad_lft_mode,
        tgsd_tau=tgsd_tau,
        tgsd_top_percent=tgsd_top_percent,
        tgsd_gradient_operator=tgsd_gradient_operator,
    )

    print(f"=== EXPERIMENT SPEED ({selected_speed}) ===")
    print(
        "PAD runtime artifact update:",
        "enabled" if update_runtime_artifacts else "disabled",
    )
    if selected_custom_repeat is not None:
        print(f"=== CUSTOM REPEAT ({selected_custom_repeat}) ===")
    if selected_casia_balance is not None:
        print(f"=== CASIA PRE-EXTRACTION BALANCE ({selected_casia_balance}) ===")
    if selected_replay_balance is not None:
        print(f"=== REPLAY PRE-EXTRACTION BALANCE ({selected_replay_balance}) ===")
    print(f"=== TRAINING ({selected_train_mode}) ===")
    train_pad_model(
        selected_train_mode,
        experiment_speed=selected_speed,
        custom_repeat_override=selected_custom_repeat,
        casia_balance_override=selected_casia_balance,
        replay_balance_override=selected_replay_balance,
        artifact_profile=artifact_profile,
        update_runtime_artifacts=update_runtime_artifacts,
        augment_live_photometric=augment_live_photometric,
        augment_live_photometric_mode=augment_live_photometric_mode,
        augment_live_photometric_scope=augment_live_photometric_scope,
    )

    print(f"=== EVALUATION ({selected_eval_mode}) ===")
    evaluate_pad_model(
        selected_eval_mode,
        artifact_profile=artifact_profile,
        log_live_false_negatives=log_live_false_negatives,
        live_false_negative_limit=live_false_negative_limit,
        export_curves=export_curves,
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
    parser.add_argument("--train-mode", choices=["casia", "replay", "combined", "custom", "combined_custom"])
    parser.add_argument("--eval-mode", choices=["casia", "replay", "custom", "all"])
    parser.add_argument("--experiment-speed", choices=["full", "fast"])
    parser.add_argument("--custom-repeat", type=int)
    parser.add_argument("--casia-balance", choices=["true", "false"])
    parser.add_argument("--replay-balance", choices=["true", "false"])
    parser.add_argument("--pad-debug-feature-mode", choices=["baseline", "tgsd"])
    parser.add_argument("--pad-lft-mode", choices=["ggd_light"])
    parser.add_argument("--tgsd-tau", type=float)
    parser.add_argument("--tgsd-top-percent", type=float)
    parser.add_argument("--tgsd-gradient-operator", choices=["sobel"])
    parser.add_argument("--log-live-false-negatives", action="store_true")
    parser.add_argument("--export-curves", action="store_true")
    parser.add_argument("--live-false-negative-limit", type=int)
    parser.add_argument("--augment-live-photometric", action="store_true")
    parser.add_argument(
        "--augment-live-photometric-mode",
        choices=["mix"],
    )
    parser.add_argument(
        "--augment-live-photometric-scope",
        choices=["all", "runtime_only"],
    )
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
        custom_repeat=cli_args.custom_repeat,
        casia_balance=parse_optional_bool(cli_args.casia_balance),
        replay_balance=parse_optional_bool(cli_args.replay_balance),
        pad_debug_feature_mode=cli_args.pad_debug_feature_mode,
        pad_lft_mode=cli_args.pad_lft_mode,
        tgsd_tau=cli_args.tgsd_tau,
        tgsd_top_percent=cli_args.tgsd_top_percent,
        tgsd_gradient_operator=cli_args.tgsd_gradient_operator,
        log_live_false_negatives=cli_args.log_live_false_negatives,
        live_false_negative_limit=cli_args.live_false_negative_limit,
        export_curves=cli_args.export_curves,
        augment_live_photometric=cli_args.augment_live_photometric,
        augment_live_photometric_mode=cli_args.augment_live_photometric_mode,
        augment_live_photometric_scope=(
            cli_args.augment_live_photometric_scope or "all"
        ),
    )
