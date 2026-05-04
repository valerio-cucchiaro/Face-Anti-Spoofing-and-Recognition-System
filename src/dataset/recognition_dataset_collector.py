"""
Module: recognition_dataset_collector

Description:
This module is responsible for collecting face crops used to
build the custom recognition dataset.

The script captures frames from the webcam, detects and validates a
face with MediaPipe, and stores the cropped face region in the selected
``custom_raw`` split directory when the user requests a capture. The
actual recognition preprocessing is deferred to the loader and runtime
pipeline.

The custom dataset is organized as:
- ``gallery``
- ``probe_known``
- ``probe_unknown``

Images are saved sequentially and can be captured interactively
through keyboard input.

This script is a manual data-collection utility, not part of the
automated recognition training or evaluation flow.
"""

import cv2
import argparse
from pathlib import Path

from src.detection.face_detector import load_mediapipe_detector
from src.preprocessing.face_bbox_visualization import draw_visual_face_bbox
from src.preprocessing.face_preprocessing import crop_face
from src.preprocessing.frame_preprocessing import get_valid_face


# ----- CONSTANTS ----- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CUSTOM_RAW_ROOT = PROJECT_ROOT / "data" / "recognition" / "custom_raw"
THRESHOLD_AREA = 8000
PERSON = "matteo"
MODE = "probe_unknown"      # gallery, probe_known, probe_unknown
MAX_IMAGES_GALLERY = 30
MAX_IMAGES_PROBE = 10

VALID_SPLITS = {"gallery", "probe_known", "probe_unknown"}
MAX_IMAGES_BY_SPLIT = {
    "gallery": MAX_IMAGES_GALLERY,
    "probe_known": MAX_IMAGES_PROBE,
    "probe_unknown": MAX_IMAGES_PROBE,
}

def resolve_dataset_dir(split_name, person_name):
    """
    Resolve the raw custom recognition dataset directory for one split
    and one identity.
    """

    return CUSTOM_RAW_ROOT / split_name / person_name


def collect_dataset(person_name=PERSON, split_name=MODE):
    """
    Collect webcam face samples for the custom recognition dataset.

    The function detects the largest valid face in each frame through
    the shared MediaPipe-based validation helper and stores the raw
    frame in the selected ``custom_raw`` split when the user presses
    the capture key.
    """

    if split_name not in VALID_SPLITS:
        raise ValueError(
            f"Unsupported split: {split_name}. Use one of: {sorted(VALID_SPLITS)}"
        )

    dataset_dir = resolve_dataset_dir(split_name, person_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    existing_images = list(dataset_dir.glob("*.png"))
    image_count = len(existing_images)
    max_images = MAX_IMAGES_BY_SPLIT[split_name]

    cap = cv2.VideoCapture(0)
    detector = load_mediapipe_detector()

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        display_frame = frame.copy()
        valid_face = False
        face_crop = None

        face_rect, gray_frame, message = get_valid_face(
            frame,
            detector,
            THRESHOLD_AREA,
        )

        if message == "No face detected":
            cv2.putText(
                display_frame,
                message,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        elif message == "Move closer to the camera":
            cv2.putText(
                display_frame,
                message,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        elif message == "Face detected":
            x, y, w, h = face_rect

            valid_face = True
            face_crop = crop_face(frame, x, y, w, h, padding_ratio=0.10)
            draw_visual_face_bbox(display_frame, face_rect, (0, 255, 0), 2)

            cv2.putText(
                display_frame,
                message,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                f"Samples: {image_count} / {max_images}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Webcam", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and valid_face:
            if image_count < max_images:
                image_count += 1
                if face_crop is None:
                    print("Skipping save: invalid face crop")
                else:
                    cv2.imwrite(str(dataset_dir / f"{image_count:03d}.png"), face_crop)

            else:
                print("Maximum number of images reached")
                break
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect raw recognition dataset samples."
    )
    parser.add_argument("--person", default=PERSON)
    parser.add_argument(
        "--split",
        default=MODE,
        choices=sorted(VALID_SPLITS),
    )
    cli_args = parser.parse_args()
    collect_dataset(person_name=cli_args.person, split_name=cli_args.split)
