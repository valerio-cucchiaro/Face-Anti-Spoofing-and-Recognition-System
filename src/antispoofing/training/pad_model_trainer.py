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
import time

from src.antispoofing.pad_model_config import (
    build_pad_model_config,
    get_pad_artifact_paths,
    save_pad_model_config,
)
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
from src.antispoofing.pad_feature_extractor import get_pad_feature_modes


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
    augment_live_photometric=False,
    augment_live_photometric_mode=None,
    augment_live_photometric_scope="all",
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

    if selected_mode == "casia":
        dataset_name = (
            f"CASIA train (sample_ratio={casia_sample_ratio}, "
            f"sampling_seed={TRAIN_SAMPLING_SEED})"
        )
        return dataset_name, load_casia_pad_training_dataset(
            sample_ratio=casia_sample_ratio,
            sample_seed=TRAIN_SAMPLING_SEED,
            balance=casia_balance,
            augment_live_photometric=augment_live_photometric,
            augment_live_photometric_mode=augment_live_photometric_mode,
        )

    if selected_mode == "replay":
        dataset_name = (
            f"Replay-Attack train (sample_ratio={replay_sample_ratio}, "
            f"sampling_seed={TRAIN_SAMPLING_SEED})"
        )
        return dataset_name, load_replay_pad_training_dataset(
            sample_ratio=replay_sample_ratio,
            sample_seed=TRAIN_SAMPLING_SEED,
            balance=replay_balance,
            augment_live_photometric=augment_live_photometric,
            augment_live_photometric_mode=augment_live_photometric_mode,
        )

    if selected_mode == "combined":
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
            augment_live_photometric=augment_live_photometric,
            augment_live_photometric_mode=augment_live_photometric_mode,
            augment_live_photometric_scope=augment_live_photometric_scope,
        )

    if selected_mode == "runtime":
        return "Runtime train", load_runtime_pad_training_dataset(
            augment_live_photometric=augment_live_photometric,
            augment_live_photometric_mode=augment_live_photometric_mode,
        )

    if selected_mode == "combined_runtime":
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
            augment_live_photometric=augment_live_photometric,
            augment_live_photometric_mode=augment_live_photometric_mode,
            augment_live_photometric_scope=augment_live_photometric_scope,
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
    artifact_profile="runtime",
    update_runtime_artifacts=False,
    augment_live_photometric=False,
    augment_live_photometric_mode=None,
    augment_live_photometric_scope="all",
):
    """
    Train the PAD model using the selected PAD training dataset.

    The experiment speed controls the default sampling and balancing
    preset. runtime_repeat_override can be used to force a custom
    runtime oversampling factor for a specific experiment. The same
    override mechanism is available for CASIA and Replay-Attack
    pre-extraction balancing.

    The trained artifacts can be stored under a dedicated profile
    (for example ``fast`` or ``full``). When ``update_runtime_artifacts``
    is enabled, the same trained model is also copied to the canonical
    runtime paths used by webcam inference.
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
        augment_live_photometric=augment_live_photometric,
        augment_live_photometric_mode=augment_live_photometric_mode,
        augment_live_photometric_scope=augment_live_photometric_scope,
    )
    load_elapsed = time.perf_counter() - load_start_time

    # print control information about the loaded dataset
    live_count = (y_train == 1).sum()
    spoof_count = (y_train == 0).sum()
    active_feature_modes = get_pad_feature_modes()

    print("PAD training dataset:", dataset_name)
    print(
        "PAD feature setup:",
        (
            f"debug={active_feature_modes['pad_debug_feature_mode']} | "
            f"lft={active_feature_modes['pad_lft_mode']} | "
            f"tgsd_tau={active_feature_modes['tgsd_tau']} | "
            f"tgsd_top_percent={active_feature_modes['tgsd_top_percent']} | "
            f"tgsd_grad={active_feature_modes['tgsd_gradient_operator']}"
        ),
    )
    if augment_live_photometric:
        print(
            "PAD live photometric augmentation:",
            f"enabled | mode={augment_live_photometric_mode or 'mix'} | scope={augment_live_photometric_scope}",
        )
    else:
        print("PAD live photometric augmentation: disabled")
    print(
        "PAD training samples:",
        f"total={len(y_train)} | live={live_count} | spoof={spoof_count} | feature_dim={X_train.shape[1]}",
    )
    print(f"PAD training dataset loading completed in {load_elapsed:.2f}s")

    # Train the PAD classifier
    training_start_time = time.perf_counter()
    scaler, classifier = train_pad_classifier(X_train, y_train)
    training_elapsed = time.perf_counter() - training_start_time

    artifact_paths = get_pad_artifact_paths(artifact_profile)

    # create models directory if it does not exist
    artifact_paths["model_path"].parent.mkdir(parents=True, exist_ok=True)

    # Save the trained model and scaler to disk
    joblib.dump(scaler, artifact_paths["scaler_path"])
    joblib.dump(classifier, artifact_paths["model_path"])

    # Save the feature configuration used to produce the trained PAD
    # artifacts so runtime inference can restore the exact same setup.
    pad_model_config = build_pad_model_config(
        debug_feature_mode=active_feature_modes["pad_debug_feature_mode"],
        lft_mode=active_feature_modes["pad_lft_mode"],
        feature_dim=X_train.shape[1],
        tgsd_tau=active_feature_modes.get("tgsd_tau"),
        tgsd_top_percent=active_feature_modes.get("tgsd_top_percent"),
        tgsd_gradient_operator=active_feature_modes.get("tgsd_gradient_operator"),
    )
    save_pad_model_config(
        pad_model_config,
        config_path=artifact_paths["config_path"],
    )

    if update_runtime_artifacts and artifact_profile != "runtime":
        runtime_paths = get_pad_artifact_paths("runtime")
        joblib.dump(scaler, runtime_paths["scaler_path"])
        joblib.dump(classifier, runtime_paths["model_path"])
        save_pad_model_config(
            pad_model_config,
            config_path=runtime_paths["config_path"],
        )
 
    print("PAD model trained and saved successfully")
    print("PAD artifact profile:", artifact_profile)
    print(
        "PAD artifacts:",
        f"model={artifact_paths['model_path']} | scaler={artifact_paths['scaler_path']} | config={artifact_paths['config_path']}",
    )
    if update_runtime_artifacts and artifact_profile != "runtime":
        runtime_paths = get_pad_artifact_paths("runtime")
        print(
            "PAD runtime artifacts:",
            f"model={runtime_paths['model_path']} | scaler={runtime_paths['scaler_path']} | config={runtime_paths['config_path']}",
        )
    print(f"PAD classifier training completed in {training_elapsed:.2f}s")
    print(f"Total PAD training time: {time.perf_counter() - total_start_time:.2f}s")




if __name__ == "__main__":
    train_pad_model()
