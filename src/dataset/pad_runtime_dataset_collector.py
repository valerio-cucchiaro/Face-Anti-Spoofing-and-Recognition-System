"""
Module: pad_runtime_dataset_collector

Description:
This module collects real runtime face samples for offline PAD testing.

It captures frames from the webcam, detects and validates a face,
and saves the raw cropped face region into one of two folders:

data/pad_runtime_test/live
data/pad_runtime_test/spoof

The saved images are NOT preprocessed for recognition, because the PAD
offline tester must reapply its own face normalization and feature
extraction pipeline exactly as done during training.
"""

import cv2
from pathlib import Path

from src.detection.face_detector import load_mediapipe_detector
from src.preprocessing.frame_preprocessing import get_valid_face


# ----- CONSTANTS ----- #
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 6
MIN_SIZE = (30, 30)
THRESHOLD_AREA = 3000

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "haarcascade_frontalface_default.xml"

RUNTIME_TEST_ROOT = PROJECT_ROOT / "data" / "pad_runtime_test"
LIVE_DATASET = RUNTIME_TEST_ROOT / "live"
SPOOF_DATASET = RUNTIME_TEST_ROOT / "spoof"

MAX_IMAGES_PER_CLASS = 10


def get_next_image_index(folder_path):
    """
    Return the next sequential image index for a dataset folder.
    """
    existing_images = sorted(folder_path.glob("*.png"))
    return len(existing_images) + 1


def collect_pad_runtime_dataset():
    """
    Collect live and spoof face crops from the webcam for PAD testing.

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

    #detector = load_face_detector(MODEL_PATH)
    detector = load_mediapipe_detector()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        valid_face = False
        pad_face_crop = None

        face_rect, gray_frame, message = get_valid_face(
            frame,
            detector,
            SCALE_FACTOR,
            MIN_NEIGHBORS,
            MIN_SIZE,
            THRESHOLD_AREA,
            method="mediapipe"
        )

        # Default overlay info
        cv2.putText(
            frame,
            f"LIVE samples: {live_count} / {MAX_IMAGES_PER_CLASS}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"SPOOF samples: {spoof_count} / {MAX_IMAGES_PER_CLASS}",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "Press L = save live | S = save spoof | Q = quit",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
            cv2.LINE_AA
        )

        if message == "No face detected":
            cv2.putText(
                frame,
                message,
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        elif message == "Move closer to the camera":
            cv2.putText(
                frame,
                message,
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        elif message == "Face detected":
            x, y, w, h = face_rect
            valid_face = True

            # PAD crop aligned with training flow: crop from grayscale frame
            pad_margin_x = int(w * 0.12)
            pad_margin_y = int(h * 0.18)

            x1 = max(0, x - pad_margin_x)
            y1 = max(0, y - pad_margin_y)
            x2 = min(gray_frame.shape[1], x + w + pad_margin_x)
            y2 = min(gray_frame.shape[0], y + h + pad_margin_y)

            pad_face_crop = gray_frame[y1:y2, x1:x2]

            # Draw face bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Show crop preview
            cv2.imshow("PAD Face Crop", pad_face_crop)

            # Show crop size for quick sanity check
            cv2.putText(
                frame,
                f"Face size: {w}x{h}",
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

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
