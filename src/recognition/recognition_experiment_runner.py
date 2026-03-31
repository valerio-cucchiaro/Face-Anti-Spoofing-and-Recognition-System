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
- `custom`, `orl`, `lfw`: single-dataset runs
- `custom_orl`: mixed setup without LFW
- `all`: full combined setup including LFW

LBPH hyperparameters can also be passed to the trainer through this
runner so small tuning sweeps can reuse the same high-level execution
path.
"""

from src.recognition.recognition_model_trainer import train_recognition_model
from src.recognition.recognition_model_evaluator import evaluate_recognition_model
import time


# ----- CONSTANTS ----- #
TRAIN_MODE = "all"  # options: custom, orl, lfw, custom_orl, all
EVAL_MODE = "all"   # options: custom, orl, lfw, custom_orl, all


def run(train_mode=None, eval_mode=None):
    """
    Execute a full recognition experiment using the current training
    and evaluation configuration.

    Supported training modes:
    - custom
    - orl
    - lfw
    - custom_orl
    - all

    Supported evaluation modes:
    - custom
    - orl
    - lfw
    - custom_orl
    - all

    The runner always executes training first and then evaluates the
    freshly saved model produced by that training run.
    """
    return run_with_lbph_params(
        train_mode=train_mode,
        eval_mode=eval_mode,
        lbph_params=None,
    )


def run_with_lbph_params(train_mode=None, eval_mode=None, lbph_params=None):
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

    print(f"=== TRAINING ({selected_train_mode}) ===")
    train_recognition_model(selected_train_mode, lbph_params=lbph_params)

    print(f"=== EVALUATION ({selected_eval_mode}) ===")
    evaluate_recognition_model(selected_train_mode, selected_eval_mode)
    print(f"=== TOTAL EXPERIMENT TIME: {time.perf_counter() - total_start_time:.2f}s ===")


if __name__ == "__main__":
    # run()

    run_with_lbph_params(
        train_mode="custom_orl",
        eval_mode="orl",
        lbph_params={
            "radius": 2,
            "neighbors": 8,
            "grid_x": 10,
            "grid_y": 10,
        },
    )
    run_with_lbph_params(
        train_mode="custom_orl",
        eval_mode="custom",
        lbph_params={
            "radius": 2,
            "neighbors": 8,
            "grid_x": 10,
            "grid_y": 10,
        },
    )

