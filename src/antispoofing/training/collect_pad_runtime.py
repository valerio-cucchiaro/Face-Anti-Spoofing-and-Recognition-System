"""
Module: collect_pad_runtime_dataset

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

from src.detection.face_detector import load_face_detector
from src.preprocessing.frame_preprocessing import get_valid_face


# ----- CONSTANTS ----- #
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 6
MIN_SIZE = (30, 30)
THRESHOLD_AREA = 3000

MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "haarcascade_frontalface_default.xml"

RUNTIME_TEST_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data" / "pad_runtime_test"
LIVE_DATASET = RUNTIME_TEST_ROOT / "live"
SPOOF_DATASET = RUNTIME_TEST_ROOT / "spoof"

MAX_IMAGES_PER_CLASS = 30


def get_next_image_index(folder_path):
    """
    Return the next sequential image index for a dataset folder.

    Args:
        folder_path: Path of the dataset folder.

    Returns:
        int: next image index starting from 1.
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

    detector = load_face_detector(MODEL_PATH)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        valid_face = False
        face_crop = None

        # Detect and validate face
        face_rect, gray_frame, message = get_valid_face(
            frame,
            detector,
            SCALE_FACTOR,
            MIN_NEIGHBORS,
            MIN_SIZE,
            THRESHOLD_AREA
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

            # Raw grayscale face crop to be saved for PAD offline testing
            face_crop = gray_frame[y:y + h, x:x + w]

            # Draw face bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Show crop preview
            cv2.imshow("PAD Face Crop", face_crop)

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

        if key == ord("l") and valid_face and face_crop is not None:
            if live_count < MAX_IMAGES_PER_CLASS:
                save_path = LIVE_DATASET / f"{live_next_idx:03d}.png"
                cv2.imwrite(str(save_path), face_crop)
                print(f"[LIVE] Saved: {save_path.name}")
                live_count += 1
                live_next_idx += 1
            else:
                print("Maximum number of LIVE images reached")

        elif key == ord("s") and valid_face and face_crop is not None:
            if spoof_count < MAX_IMAGES_PER_CLASS:
                save_path = SPOOF_DATASET / f"{spoof_next_idx:03d}.png"
                cv2.imwrite(str(save_path), face_crop)
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