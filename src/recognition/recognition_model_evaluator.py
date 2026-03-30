"""
Module: recognition_model_evaluator

Description:
This module evaluates the trained LBPH face recognition model on one
selected probe dataset or on all supported probe datasets
sequentially.

Evaluation uses the single recognition model currently stored on disk,
which is overwritten every time a new training run is executed.
"""

import cv2
import pickle
from pathlib import Path
import time

from src.recognition.recognition_dataset_loader import load_probe_dataset, should_namespace_identities


# ----- CONSTANTS ----- #
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "lbph_model.xml"
LABEL_MAP_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "label_map.pkl"
EVAL_MODE = "custom"
PROBE_PROGRESS_STEP = 100


def load_trained_recognition_model():
    """
    Load the trained recognition model and label map from disk.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Recognition model not found: {MODEL_PATH}")

    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(f"Recognition label map not found: {LABEL_MAP_PATH}")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))

    with open(LABEL_MAP_PATH, "rb") as file:
        label_map = pickle.load(file)

    return recognizer, label_map


def evaluate_single_recognition_dataset(eval_mode, recognizer, label_map, namespace_identities=False):
    """
    Evaluate the trained recognition model on one selected probe dataset.
    """
    print(f"=== RECOGNITION EVALUATION MODE: {eval_mode.upper()} ===")

    load_start_time = time.perf_counter()
    probe_records, dataset_counts = load_probe_dataset(
        eval_mode,
        namespace_identities=namespace_identities,
    )
    load_elapsed = time.perf_counter() - load_start_time

    if not probe_records:
        print("No valid probe images found for evaluation.")
        print()
        return

    genuine_scores = []
    impostor_scores = []
    correct = 0
    total = 0
    evaluation_start_time = time.perf_counter()

    for index, record in enumerate(probe_records, start=1):
        label, confidence = recognizer.predict(record["image"])
        predicted_name = label_map[label]
        true_name = record["identity"]

        if predicted_name == true_name:
            genuine_scores.append(confidence)
            correct += 1
        else:
            impostor_scores.append(confidence)

        total += 1

        if index % PROBE_PROGRESS_STEP == 0:
            print(f"Evaluated {index}/{len(probe_records)} probe images")

    print("Probe samples:", total)
    print(f"Probe loading completed in {load_elapsed:.2f}s")

    for dataset_name, sample_count in dataset_counts.items():
        print(f"{dataset_name} probe samples: {sample_count}")

    print("Correct:", correct)
    print("Accuracy:", correct / total)
    print("Genuine matches:", len(genuine_scores))
    print("Impostor matches:", len(impostor_scores))

    if genuine_scores:
        print("Mean genuine confidence:", sum(genuine_scores) / len(genuine_scores))
        print("Suggested threshold:", max(genuine_scores) + 5)

    if impostor_scores:
        print("Mean impostor confidence:", sum(impostor_scores) / len(impostor_scores))

    print(f"Evaluation completed in {time.perf_counter() - evaluation_start_time:.2f}s")
    print()


def evaluate_recognition_model(train_mode=None, eval_mode=None):
    """
    Evaluate the recognition model on one dataset or on all supported
    datasets sequentially.
    """
    selected_train_mode = train_mode or "custom"
    selected_eval_mode = eval_mode or EVAL_MODE
    namespace_identities = should_namespace_identities(selected_train_mode)
    total_start_time = time.perf_counter()

    recognizer, label_map = load_trained_recognition_model()

    if selected_eval_mode == "all":
        for mode in ("custom", "orl", "lfw"):
            evaluate_single_recognition_dataset(
                mode,
                recognizer,
                label_map,
                namespace_identities=namespace_identities,
            )
        print(f"Total evaluation time: {time.perf_counter() - total_start_time:.2f}s")
        return

    evaluate_single_recognition_dataset(
        selected_eval_mode,
        recognizer,
        label_map,
        namespace_identities=namespace_identities,
    )
    print(f"Total evaluation time: {time.perf_counter() - total_start_time:.2f}s")


if __name__ == "__main__":
    evaluate_recognition_model()
