"""
Module: pad_runtime_dataset_collector

Description:
This module collects real runtime face samples for offline PAD
training dataset creation.

It captures frames from the webcam, detects and validates a face,
builds the same padded grayscale crop used by the runtime PAD flow,
filters it with shared quality checks, and saves it into one of two
folders:

data/pad_runtime_dataset/train/live
data/pad_runtime_dataset/train/spoof

The saved images are not converted into PAD features here. They remain
image crops so that the offline PAD training and evaluation tools can
reapply their own normalization and feature extraction pipeline later.
"""

import cv2
from pathlib import Path

from src.antispoofing.pad_crop_utils import extract_pad_face_crop
from src.antispoofing.pad_frame_quality import evaluate_pad_crop_quality
from src.detection.face_detector import load_mediapipe_detector
from src.preprocessing.frame_preprocessing import get_valid_face


# ----- CONSTANTS ----- #
THRESHOLD_AREA = 8000

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RUNTIME_DATASET_ROOT = PROJECT_ROOT / "data" / "pad_runtime_dataset"
TRAIN_ROOT = RUNTIME_DATASET_ROOT / "train"
LIVE_DATASET = TRAIN_ROOT / "live"
SPOOF_DATASET = TRAIN_ROOT / "spoof"

MAX_IMAGES_PER_CLASS = 60

# ----- PAD FRAME QUALITY FILTERING ----- #
PAD_MIN_VARIANCE_LAPLACIAN = 80.0


def get_next_image_index(folder_path):
    """
    Return the next sequential image index for a dataset folder.
    """
    existing_images = sorted(folder_path.glob("*.png"))
    return len(existing_images) + 1


def collect_pad_runtime_dataset():
    """
    Collect live and spoof PAD crops from the webcam for runtime-aware
    PAD dataset creation.

    The function reuses the shared face validation, PAD crop
    extraction, and PAD quality checks so that saved samples follow the
    same geometric assumptions as the runtime PAD pipeline.

    Controls:
        l -> save current valid face crop in the live folder
        s -> save current valid face crop in the spoof folder
        q -> quit

    Notes:
        - live  = real face from webcam
        - spoof = face shown through phone/photo attack
    """

    # Create dataset directories if they do not exist
    LIVE_DATASET.mkdir(parents=True, exist_ok=True)
    SPOOF_DATASET.mkdir(parents=True, exist_ok=True)

    live_count = len(list(LIVE_DATASET.glob("*.png")))
    spoof_count = len(list(SPOOF_DATASET.glob("*.png")))

    live_next_idx = get_next_image_index(LIVE_DATASET)
    spoof_next_idx = get_next_image_index(SPOOF_DATASET)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    detector = load_mediapipe_detector()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        valid_face = False
        pad_face_crop = None
        quality_text = "PAD frame: WAITING"
        quality_color = (200, 200, 200)

        face_rect, gray_frame, message = get_valid_face(
            frame,
            detector,
            THRESHOLD_AREA,
        )

        # Default overlay info
        cv2.putText(frame, f"LIVE samples: {live_count} / {MAX_IMAGES_PER_CLASS}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, f"SPOOF samples: {spoof_count} / {MAX_IMAGES_PER_CLASS}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, "Press L = save live | S = save spoof | Q = quit", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

        if message == "No face detected":
            cv2.putText(frame, message, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        elif message == "Move closer to the camera":
            cv2.putText(frame, message, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        elif message == "Face detected":
            x, y, w, h = face_rect

            # Build the PAD crop through the same helper used at runtime.
            pad_face_crop = extract_pad_face_crop(gray_frame, x, y, w, h)
            quality_result = evaluate_pad_crop_quality(
                pad_face_crop,
                PAD_MIN_VARIANCE_LAPLACIAN,
            )
            laplacian_variance = quality_result["laplacian_variance"]

            # Only allow saving crops that pass the same structural and blur
            # checks used by the runtime PAD history update.
            valid_face = quality_result["is_frame_usable_for_history"]

            # Draw face bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Show crop preview
            cv2.imshow("PAD Face Crop", pad_face_crop)

            if not quality_result["is_valid_crop"]:
                if not quality_result["is_large_enough"]:
                    quality_text = (
                        f"PAD frame: SKIPPED | too small "
                        f"({quality_result['crop_width']}x{quality_result['crop_height']})"
                    )
                elif not quality_result["has_plausible_aspect_ratio"]:
                    quality_text = (
                        f"PAD frame: SKIPPED | bad ratio "
                        f"({quality_result['aspect_ratio']:.2f})"
                    )
                else:
                    quality_text = "PAD frame: SKIPPED | invalid crop"
                quality_color = (0, 165, 255)
            elif not quality_result["is_sharp_enough"]:
                quality_text = f"PAD frame: SKIPPED | blur={laplacian_variance:.1f}"
                quality_color = (0, 165, 255)
            else:
                quality_text = (
                    f"PAD frame: READY | {quality_result['crop_width']}x"
                    f"{quality_result['crop_height']} | blur={laplacian_variance:.1f}"
                )
                quality_color = (0, 255, 0)

            # Show crop and detector size for quick sanity checks.
            cv2.putText(frame, f"Face size: {w}x{h}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, quality_text, (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2, cv2.LINE_AA)

        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("l") and valid_face and pad_face_crop is not None:
            if live_count < MAX_IMAGES_PER_CLASS:
                save_path = LIVE_DATASET / f"{live_next_idx:03d}.png"
                cv2.imwrite(str(save_path), pad_face_crop)
                print(f"[LIVE] Saved: {save_path.name}")
                live_count += 1
                live_next_idx += 1
            else:
                print("Maximum number of LIVE images reached")

        elif key == ord("s") and valid_face and pad_face_crop is not None:
            if spoof_count < MAX_IMAGES_PER_CLASS:
                save_path = SPOOF_DATASET / f"{spoof_next_idx:03d}.png"
                cv2.imwrite(str(save_path), pad_face_crop)
                print(f"[SPOOF] Saved: {save_path.name}")
                spoof_count += 1
                spoof_next_idx += 1
            else:
                print("Maximum number of SPOOF images reached")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    collect_pad_runtime_dataset()
