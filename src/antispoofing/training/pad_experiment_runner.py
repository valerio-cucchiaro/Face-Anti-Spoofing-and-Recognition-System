"""
Module: pad_experiment_runner

Description:
This module runs a complete PAD experiment by executing training
followed by evaluation with the currently selected dataset settings.

Training and evaluation modes are configured independently because the
project may train on one dataset combination and evaluate on another.
Only the canonical `all -> all | full` run is allowed to promote
the webcam runtime PAD artifacts.
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
    pad_feature_mode=None,
    pad_lft_mode=None,
    tgsd_tau=None,
    tgsd_top_percent=None,
    tgsd_gradient_operator=None,
    log_live_false_negatives=False,
    live_false_negative_limit=None,
    augment_live_photometric=False,
    augment_live_photometric_mode=None,
    augment_live_photometric_scope="all",
    pad_validation_target_apcer=None,
    export_curves=False,
    custom_pad_validation_ratio=None,
    custom_pad_validation_seed=None,
    save_system_threshold=False,
    artifact_tag=None,
):
    """
    Execute a full PAD experiment using the current training
    and evaluation configuration.

    Supported training modes:
    - casia
    - replay
    - combined
    - custom
    - all
    - old_all

    Supported evaluation modes:
    - casia
    - replay
    - custom
    - all
    - old_all

    Supported experiment speeds:
    - full
    - fast

    If custom_repeat is set, it overrides the default custom repeat
    associated with the selected speed preset.

    If casia_balance or replay_balance is set, it overrides the
    corresponding pre-extraction balancing flag associated with the
    selected speed preset.

    pad_feature_mode selects the PAD feature branch used for this
    experiment. The current supported choices are the baseline stack and
    the TGSD-extended branch. pad_lft_mode is kept explicit so the
    runner can persist the active PAD config coherently, even though the
    active LFT branch is now fixed to ``ggd_light``.

    pad_validation_target_apcer is used by the validation-backed
    protocols. `casia` and `replay` select their threshold on the
    corresponding validation split before testing. `custom` uses a
    deterministic pseudo-validation subset carved out of the custom
    train split. `all` selects one global threshold on the combined
    validation pool formed by CASIA validation, Replay-Attack
    validation, and custom pseudo-validation.

    If save_system_threshold is enabled, it is valid only for the
    canonical ``all -> all | full`` run. In that case the custom
    combined-validation operating point is saved as the offline
    system-level PAD threshold aligned with the canonical runtime PAD
    artifacts.
    """
    selected_train_mode = train_mode or TRAIN_MODE
    selected_eval_mode = eval_mode or EVAL_MODE
    selected_speed = experiment_speed or EXPERIMENT_SPEED
    selected_custom_repeat = custom_repeat if custom_repeat is not None else CUSTOM_REPEAT
    selected_casia_balance = casia_balance if casia_balance is not None else CASIA_BALANCE
    selected_replay_balance = replay_balance if replay_balance is not None else REPLAY_BALANCE
    total_start_time = time.perf_counter()
    artifact_profile = selected_speed
    update_runtime_artifacts = (
        selected_speed == "full"
        and selected_train_mode == "all"
        and selected_eval_mode == "all"
        and artifact_tag is None
    )
    allow_system_threshold_update = update_runtime_artifacts or (artifact_tag is not None)

    if save_system_threshold and not allow_system_threshold_update:
        raise ValueError(
            "PAD --save-system-threshold is allowed only for the canonical "
            "'all -> all | full' run, or for explicitly tagged artifact runs."
        )

    configure_pad_feature_modes(
        debug_mode=pad_feature_mode,
        lft_mode=pad_lft_mode,
        tgsd_tau=tgsd_tau,
        tgsd_top_percent=tgsd_top_percent,
        tgsd_gradient_operator=tgsd_gradient_operator,
    )

    print(f"=== EXPERIMENT SPEED ({selected_speed}) ===")
    print(
        "PAD runtime artifact update:",
        (
            "enabled (canonical all->all full run)"
            if update_runtime_artifacts
            else "disabled"
        ),
    )
    if artifact_tag is not None:
        print(f"=== PAD ARTIFACT TAG ({artifact_tag}) ===")
    if selected_custom_repeat is not None:
        print(f"=== CUSTOM REPEAT ({selected_custom_repeat}) ===")
    if selected_casia_balance is not None:
        print(f"=== CASIA PRE-EXTRACTION BALANCE ({selected_casia_balance}) ===")
    if selected_replay_balance is not None:
        print(f"=== REPLAY PRE-EXTRACTION BALANCE ({selected_replay_balance}) ===")
    if custom_pad_validation_ratio is not None:
        print(f"=== CUSTOM PAD VALIDATION RATIO ({custom_pad_validation_ratio}) ===")
    if custom_pad_validation_seed is not None:
        print(f"=== CUSTOM PAD VALIDATION SEED ({custom_pad_validation_seed}) ===")
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
        custom_validation_ratio=custom_pad_validation_ratio,
        custom_validation_seed=custom_pad_validation_seed,
        artifact_tag=artifact_tag,
    )

    print(f"=== EVALUATION ({selected_eval_mode}) ===")
    evaluate_pad_model(
        selected_eval_mode,
        artifact_profile=artifact_profile,
        log_live_false_negatives=log_live_false_negatives,
        live_false_negative_limit=live_false_negative_limit,
        pad_validation_target_apcer=pad_validation_target_apcer,
        export_curves=export_curves,
        save_system_threshold=allow_system_threshold_update and save_system_threshold,
        artifact_tag=artifact_tag,
    )
    print(f"=== TOTAL PAD EXPERIMENT TIME: {time.perf_counter() - total_start_time:.2f}s ===")


def build_arg_parser():
    """
    Build the command-line interface for PAD experiments.

    The CLI exposes the runner parameters needed for scripted PAD
    campaigns, including the PAD feature-mode overrides documented in
    `antispoofing.md`.
    """
    parser = argparse.ArgumentParser(
        description="Run a PAD experiment with optional sampling and feature overrides."
    )
    parser.add_argument(
        "--train-mode",
        choices=["casia", "replay", "combined", "custom", "all", "old_all"],
    )
    parser.add_argument(
        "--eval-mode",
        choices=["casia", "replay", "custom", "all", "old_all"],
    )
    parser.add_argument("--experiment-speed", choices=["full", "fast"])
    parser.add_argument("--custom-repeat", type=int)
    parser.add_argument("--casia-balance", choices=["true", "false"])
    parser.add_argument("--replay-balance", choices=["true", "false"])
    parser.add_argument(
        "--pad-feature-mode",
        choices=["baseline", "tgsd"],
        dest="pad_feature_mode",
    )
    parser.add_argument("--pad-lft-mode", choices=["ggd_light"])
    parser.add_argument("--tgsd-tau", type=float)
    parser.add_argument("--tgsd-top-percent", type=float)
    parser.add_argument("--tgsd-gradient-operator", choices=["sobel"])
    parser.add_argument("--log-live-false-negatives", action="store_true")
    parser.add_argument("--export-curves", action="store_true")
    parser.add_argument(
        "--save-system-threshold",
        action="store_true",
        help=(
            "Save the offline system PAD threshold only for the canonical "
            "'all -> all | full' run."
        ),
    )
    parser.add_argument(
        "--artifact-tag",
        help=(
            "Optional PAD artifact tag used to save and evaluate a separate "
            "non-canonical PAD artifact bundle."
        ),
    )
    parser.add_argument("--live-false-negative-limit", type=int)
    parser.add_argument("--pad-validation-target-apcer", type=float)
    parser.add_argument("--custom-pad-validation-ratio", type=float)
    parser.add_argument("--custom-pad-validation-seed", type=int)
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
        pad_feature_mode=cli_args.pad_feature_mode,
        pad_lft_mode=cli_args.pad_lft_mode,
        tgsd_tau=cli_args.tgsd_tau,
        tgsd_top_percent=cli_args.tgsd_top_percent,
        tgsd_gradient_operator=cli_args.tgsd_gradient_operator,
        log_live_false_negatives=cli_args.log_live_false_negatives,
        live_false_negative_limit=cli_args.live_false_negative_limit,
        pad_validation_target_apcer=cli_args.pad_validation_target_apcer,
        export_curves=cli_args.export_curves,
        save_system_threshold=cli_args.save_system_threshold,
        artifact_tag=cli_args.artifact_tag,
        custom_pad_validation_ratio=cli_args.custom_pad_validation_ratio,
        custom_pad_validation_seed=cli_args.custom_pad_validation_seed,
        augment_live_photometric=cli_args.augment_live_photometric,
        augment_live_photometric_mode=cli_args.augment_live_photometric_mode,
        augment_live_photometric_scope=(
            cli_args.augment_live_photometric_scope or "all"
        ),
    )
