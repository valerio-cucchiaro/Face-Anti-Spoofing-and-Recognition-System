"""
Module: pad_experiment_runner

Description:
This module runs a complete PAD experiment by executing training
followed by evaluation with the currently selected dataset settings.

Training and evaluation modes are configured independently because the
project may train on one dataset combination and evaluate on another.
"""

from src.antispoofing.training.pad_model_trainer import train_pad_model
from src.antispoofing.training.pad_model_evaluator import evaluate_pad_model
import time

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
    """
    selected_train_mode = train_mode or TRAIN_MODE
    selected_eval_mode = eval_mode or EVAL_MODE
    selected_speed = experiment_speed or EXPERIMENT_SPEED
    selected_runtime_repeat = runtime_repeat if runtime_repeat is not None else RUNTIME_REPEAT
    selected_casia_balance = casia_balance if casia_balance is not None else CASIA_BALANCE
    selected_replay_balance = replay_balance if replay_balance is not None else REPLAY_BALANCE
    total_start_time = time.perf_counter()

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
    )

    print(f"=== EVALUATION ({selected_eval_mode}) ===")
    evaluate_pad_model(selected_eval_mode)
    print(f"=== TOTAL PAD EXPERIMENT TIME: {time.perf_counter() - total_start_time:.2f}s ===")

if __name__ == "__main__":
    run()
