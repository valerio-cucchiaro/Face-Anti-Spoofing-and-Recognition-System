"""
Module: pad_model_trainer

Description:
This module implements the training pipeline for the face anti-spoofing
presentation attack detection (PAD) model.

The script loads the selected PAD training dataset, trains the SVM
classifier, and saves both the trained model and the fitted scaler
to disk.

The saved artifacts will later be used during runtime inference to
classify detected faces as live or spoof before face recognition.
"""

import joblib
from pathlib import Path
import time

from src.antispoofing.training.pad_dataset_loader import (
    CASIA_SAMPLE_RATIO,
    REPLAY_SAMPLE_RATIO,
    TRAIN_SAMPLING_SEED,
    load_combined_pad_training_dataset,
    load_casia_pad_training_dataset,
    load_combined_with_runtime_pad_training_dataset,
    load_replay_pad_training_dataset,
    load_runtime_pad_training_dataset,
)
from src.antispoofing.training.pad_svm_trainer import train_pad_classifier
from src.antispoofing.pad_feature_extractor import PAD_DEBUG_FEATURE_MODE


# ----- CONSTANTS ----- #
MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "pad_svm.joblib"
SCALER_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "pad_scaler.joblib"
TRAIN_MODE = "replay"
EXPERIMENT_SPEED = "full"
FULL_RUNTIME_REPEAT = 10
FAST_RUNTIME_REPEAT = 1


def get_speed_config(
    experiment_speed=None,
    runtime_repeat_override=None,
    casia_balance_override=None,
    replay_balance_override=None,
):
    """
    Resolve the sampling and balancing configuration associated with
    the selected PAD experiment speed.

    Each speed preset defines stable defaults. The runtime repeat can
    still be overridden explicitly by the experiment runner. The same
    applies to the per-dataset pre-extraction balancing flags.
    """

    selected_speed = experiment_speed or EXPERIMENT_SPEED

    if selected_speed == "full":
        config = {
            "name": "full",
            "casia_sample_ratio": CASIA_SAMPLE_RATIO,
            "replay_sample_ratio": REPLAY_SAMPLE_RATIO,
            "runtime_repeat": FULL_RUNTIME_REPEAT,
            "casia_balance": False,
            "replay_balance": True,
        }
        if runtime_repeat_override is not None:
            config["runtime_repeat"] = runtime_repeat_override
        if casia_balance_override is not None:
            config["casia_balance"] = casia_balance_override
        if replay_balance_override is not None:
            config["replay_balance"] = replay_balance_override
        return config

    if selected_speed == "fast":
        config = {
            "name": "fast",
            "casia_sample_ratio": 0.1,
            "replay_sample_ratio": 0.1,
            "runtime_repeat": FAST_RUNTIME_REPEAT,
            "casia_balance": False,
            "replay_balance": True,
        }
        if runtime_repeat_override is not None:
            config["runtime_repeat"] = runtime_repeat_override
        if casia_balance_override is not None:
            config["casia_balance"] = casia_balance_override
        if replay_balance_override is not None:
            config["replay_balance"] = replay_balance_override
        return config

    raise ValueError(
        f"Unsupported EXPERIMENT_SPEED: {selected_speed}. Use 'full' or 'fast'."
    )


def load_configured_training_dataset(
    train_mode=None,
    experiment_speed=None,
    runtime_repeat_override=None,
    casia_balance_override=None,
    replay_balance_override=None,
):
    """
    Load the PAD training split selected by TRAIN_MODE.

    Supported modes:
    - casia
    - replay
    - combined
    - runtime
    - combined_runtime

    Supported experiment speeds:
    - full
    - fast

    If runtime_repeat_override is provided, it overrides the default
    runtime weight defined by the selected speed preset.

    If casia_balance_override or replay_balance_override is provided,
    it overrides the corresponding pre-extraction balancing flag from
    the selected speed preset.
    """

    selected_mode = train_mode or TRAIN_MODE
    speed_config = get_speed_config(
        experiment_speed,
        runtime_repeat_override=runtime_repeat_override,
        casia_balance_override=casia_balance_override,
        replay_balance_override=replay_balance_override,
    )
    casia_sample_ratio = speed_config["casia_sample_ratio"]
    replay_sample_ratio = speed_config["replay_sample_ratio"]
    runtime_repeat = speed_config["runtime_repeat"]
    casia_balance = speed_config["casia_balance"]
    replay_balance = speed_config["replay_balance"]

    print(f"=== PAD EXPERIMENT SPEED: {speed_config['name'].upper()} ===")
    print(f"CASIA sample ratio: {casia_sample_ratio:.2f}")
    print(f"Replay sample ratio: {replay_sample_ratio:.2f}")
    print("CASIA pre-extraction balance:", casia_balance)
    print("Replay pre-extraction balance:", replay_balance)
    print("Runtime repeat:", runtime_repeat)

    if selected_mode == "casia":
        print("=== PAD TRAINING MODE: CASIA TRAIN ===")
        dataset_name = (
            f"CASIA train (sample_ratio={casia_sample_ratio}, "
            f"sampling_seed={TRAIN_SAMPLING_SEED})"
        )
        return dataset_name, load_casia_pad_training_dataset(
            sample_ratio=casia_sample_ratio,
            sample_seed=TRAIN_SAMPLING_SEED,
            balance=casia_balance,
        )

    if selected_mode == "replay":
        print("=== PAD TRAINING MODE: REPLAY-ATTACK TRAIN ===")
        dataset_name = (
            f"Replay-Attack train (sample_ratio={replay_sample_ratio}, "
            f"sampling_seed={TRAIN_SAMPLING_SEED})"
        )
        return dataset_name, load_replay_pad_training_dataset(
            sample_ratio=replay_sample_ratio,
            sample_seed=TRAIN_SAMPLING_SEED,
            balance=replay_balance,
        )

    if selected_mode == "combined":
        print("=== PAD TRAINING MODE: COMBINED TRAIN ===")
        dataset_name = (
            f"Combined train (casia_ratio={casia_sample_ratio}, "
            f"replay_ratio={replay_sample_ratio}, sampling_seed={TRAIN_SAMPLING_SEED})"
        )
        return dataset_name, load_combined_pad_training_dataset(
            casia_sample_ratio=casia_sample_ratio,
            replay_sample_ratio=replay_sample_ratio,
            sample_seed=TRAIN_SAMPLING_SEED,
            casia_balance=casia_balance,
            replay_balance=replay_balance,
        )

    if selected_mode == "runtime":
        print("=== PAD TRAINING MODE: RUNTIME TRAIN ===")
        return "Runtime train", load_runtime_pad_training_dataset()

    if selected_mode == "combined_runtime":
        print("=== PAD TRAINING MODE: COMBINED + RUNTIME TRAIN ===")
        dataset_name = (
            "Combined + runtime train "
            f"(casia_ratio={casia_sample_ratio}, replay_ratio={replay_sample_ratio}, "
            f"runtime_repeat={runtime_repeat})"
        )
        return dataset_name, load_combined_with_runtime_pad_training_dataset(
            runtime_repeat=runtime_repeat,
            casia_sample_ratio=casia_sample_ratio,
            replay_sample_ratio=replay_sample_ratio,
            sample_seed=TRAIN_SAMPLING_SEED,
            casia_balance=casia_balance,
            replay_balance=replay_balance,
        )

    raise ValueError(
        "Unsupported TRAIN_MODE: "
        f"{selected_mode}. Use 'casia', 'replay', 'combined', 'runtime', or 'combined_runtime'."
    )


def train_pad_model(
    train_mode=None,
    experiment_speed=None,
    runtime_repeat_override=None,
    casia_balance_override=None,
    replay_balance_override=None,
):
    """
    Train the PAD model using the selected PAD training dataset.

    The experiment speed controls the default sampling and balancing
    preset. runtime_repeat_override can be used to force a custom
    runtime oversampling factor for a specific experiment. The same
    override mechanism is available for CASIA and Replay-Attack
    pre-extraction balancing.
    """

    total_start_time = time.perf_counter()

    # Load the training dataset
    load_start_time = time.perf_counter()
    dataset_name, (X_train, y_train) = load_configured_training_dataset(
        train_mode,
        experiment_speed=experiment_speed,
        runtime_repeat_override=runtime_repeat_override,
        casia_balance_override=casia_balance_override,
        replay_balance_override=replay_balance_override,
    )
    load_elapsed = time.perf_counter() - load_start_time

    # print control information about the loaded dataset
    live_count = (y_train == 1).sum()
    spoof_count = (y_train == 0).sum()

    print("Dataset:", dataset_name)
    print("PAD debug feature mode:", PAD_DEBUG_FEATURE_MODE)
    print("Total samples:", len(y_train))
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("Live samples:", live_count)
    print("Spoof samples:", spoof_count)
    print(f"PAD training dataset loading completed in {load_elapsed:.2f}s")

    # Train the PAD classifier
    training_start_time = time.perf_counter()
    scaler, classifier = train_pad_classifier(X_train, y_train)
    training_elapsed = time.perf_counter() - training_start_time

    # create models directory if it does not exist
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save the trained model and scaler to disk
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(classifier, MODEL_PATH)
 
    print("PAD model trained and saved successfully")
    print("Model path:", MODEL_PATH)
    print("Scaler path:", SCALER_PATH)
    print(f"PAD classifier training completed in {training_elapsed:.2f}s")
    print(f"Total PAD training time: {time.perf_counter() - total_start_time:.2f}s")




if __name__ == "__main__":
    train_pad_model()
