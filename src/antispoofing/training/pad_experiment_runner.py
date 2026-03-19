"""
Module: pad_experiment_runner

Description:
This module runs a complete PAD experiment by executing training
followed by evaluation with the currently selected dataset settings.
"""

from src.antispoofing.training.pad_model_trainer import train_pad_model
from src.antispoofing.training.pad_model_evaluator import evaluate_pad_model


# ----- CONSTANTS ----- #
TRAIN_MODE = "combined"
EVAL_MODE = "casia"


def run(train_mode=None, eval_mode=None):
    """
    Execute a full PAD experiment using the current training
    and evaluation configuration.
    """
    selected_train_mode = train_mode or TRAIN_MODE
    selected_eval_mode = eval_mode or EVAL_MODE

    print(f"=== TRAINING ({selected_train_mode}) ===")
    train_pad_model(selected_train_mode)

    print(f"=== EVALUATION ({selected_eval_mode}) ===")
    evaluate_pad_model(selected_eval_mode)

if __name__ == "__main__":
    run()
