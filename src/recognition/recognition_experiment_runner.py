"""
Module: recognition_experiment_runner

Description:
This module runs a complete recognition experiment by executing
training followed by evaluation with the currently selected
recognition dataset settings.

Training and evaluation modes are configured independently so the
project can train one model and then evaluate it on one dataset or on
all supported probe datasets.
"""

from src.recognition.recognition_model_trainer import train_recognition_model
from src.recognition.recognition_model_evaluator import evaluate_recognition_model
import time


# ----- CONSTANTS ----- #
TRAIN_MODE = "all"  # options: custom, orl, lfw, all
EVAL_MODE = "all"   # options: custom, orl, lfw, all


def run(train_mode=None, eval_mode=None):
    """
    Execute a full recognition experiment using the current training
    and evaluation configuration.

    Supported training modes:
    - custom
    - orl
    - lfw
    - all

    Supported evaluation modes:
    - custom
    - orl
    - lfw
    - all
    """
    selected_train_mode = train_mode or TRAIN_MODE
    selected_eval_mode = eval_mode or EVAL_MODE
    total_start_time = time.perf_counter()

    print(f"=== TRAINING ({selected_train_mode}) ===")
    train_recognition_model(selected_train_mode)

    print(f"=== EVALUATION ({selected_eval_mode}) ===")
    evaluate_recognition_model(selected_train_mode, selected_eval_mode)
    print(f"=== TOTAL EXPERIMENT TIME: {time.perf_counter() - total_start_time:.2f}s ===")


if __name__ == "__main__":
    run()
