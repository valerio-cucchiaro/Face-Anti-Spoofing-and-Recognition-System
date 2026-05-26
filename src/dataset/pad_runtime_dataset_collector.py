"""
Module: pad_runtime_dataset_collector

Description:
This module collects custom PAD face samples for offline dataset
creation.

It captures frames from the webcam, detects and validates a face,
uses the same crop-quality checks as the runtime PAD flow, and saves
the cropped face region into one of two folders inside the selected custom
dataset split:

data/pad/pad_custom_raw/<split>/live
data/pad/pad_custom_raw/<split>/spoof

The saved images are not converted into PAD features here. They remain
cropped face images so that the offline PAD training and evaluation
tools can reapply their own normalization and feature extraction
pipeline later.

The same collector can target either the ``train`` or ``test`` split
of ``data/pad/pad_custom_raw`` through a configurable split name.
"""

import cv2
from pathlib import Path

from src.antispoofing.pad_crop_utils import (
    extract_pad_face_crop,
    extract_pad_face_crop_bgr,
)
from src.antispoofing.pad_frame_quality import (
    PAD_MIN_VARIANCE_LAPLACIAN,
    evaluate_bbox_stability,
    evaluate_pad_crop_quality,
)
from src.detection.face_detector import load_mediapipe_detector
from src.preprocessing.face_bbox_visualization import draw_visual_face_bbox
from src.preprocessing.frame_preprocessing import DEFAULT_MIN_FACE_AREA, get_valid_face


# ----- CONSTANTS ----- #
THRESHOLD_AREA = DEFAULT_MIN_FACE_AREA

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PAD_CUSTOM_RAW_ROOT = PROJECT_ROOT / "data" / "pad" / "pad_custom_raw"
DATASET_SPLIT = "test"     # train or test

MAX_IMAGES_LIVE = 120
MAX_IMAGES_SPOOF = 140


def get_next_image_index(folder_path):
    """
    Return the next sequential image index for a dataset folder.
    """
    existing_images = sorted(folder_path.glob("*.png"))
    return len(existing_images) + 1


def get_custom_split_paths(split_name):
    """
    Build the custom PAD dataset paths for the selected split.
    """

    if split_name not in {"train", "test"}:
        raise ValueError(
            "Unsupported split_name: "
            f"{split_name}. Use 'train' or 'test'."
        )

    split_root = PAD_CUSTOM_RAW_ROOT / split_name
    return split_root, split_root / "live", split_root / "spoof"


def collect_pad_custom_dataset(split_name=DATASET_SPLIT):
    """
    Collect live and spoof raw frames from the webcam for custom PAD
    dataset creation.

    The function reuses the shared face validation, PAD crop
    extraction, and PAD quality checks so that saved samples follow the
    same geometric assumptions as the runtime PAD pipeline.

    The selected split controls whether samples are saved under:
    - ``data/pad/pad_custom_raw/train/...``
    - ``data/pad/pad_custom_raw/test/...``

    Controls:
        l -> save current valid face crop in the live folder
        s -> save current valid face crop in the spoof folder
        q -> quit

    Notes:
        - live  = real face from webcam
        - spoof = face shown through phone/photo attack
    """

    _, live_dataset, spoof_dataset = get_custom_split_paths(split_name)

    # Create dataset directories if they do not exist
    live_dataset.mkdir(parents=True, exist_ok=True)
    spoof_dataset.mkdir(parents=True, exist_ok=True)

    live_count = len(list(live_dataset.glob("*.png")))
    spoof_count = len(list(spoof_dataset.glob("*.png")))

    live_next_idx = get_next_image_index(live_dataset)
    spoof_next_idx = get_next_image_index(spoof_dataset)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    detector = load_mediapipe_detector()
    prev_face_rect = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        display_frame = frame.copy()
        crop_source_frame = frame.copy()
        valid_face = False
        quality_text = "PAD frame: WAITING"
        quality_color = (200, 200, 200)
        pad_face_crop_bgr = None

        face_rect, gray_frame, message = get_valid_face(
            frame,
            detector,
            THRESHOLD_AREA,
        )

        # Default overlay info
        cv2.putText(display_frame, f"Split: {split_name.upper()}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(display_frame, f"LIVE samples: {live_count} / {MAX_IMAGES_LIVE}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(display_frame, f"SPOOF samples: {spoof_count} / {MAX_IMAGES_SPOOF}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(display_frame, "Press L = save live | S = save spoof | Q = quit", (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

        if message == "No face detected":
            prev_face_rect = None
            cv2.putText(display_frame, message, (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        elif message == "Move closer to the camera":
            prev_face_rect = None
            cv2.putText(display_frame, message, (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        elif message == "Face detected":
            x, y, w, h = face_rect
            current_face_rect = (x, y, w, h)

            # Apply the same bbox-stability policy used by the runtime
            # pipeline so the saved dataset reflects the real online
            # acquisition conditions.
            stability_result = evaluate_bbox_stability(
                current_face_rect,
                prev_face_rect,
            )
            bbox_stable = stability_result["is_stable"]
            prev_face_rect = current_face_rect

            # Build the grayscale PAD crop only as a quality gate. The
            # saved dataset image uses the same canonical PAD crop
            # geometry, but keeps the original BGR colors.
            pad_face_crop = extract_pad_face_crop(gray_frame, x, y, w, h)
            pad_face_crop_bgr = extract_pad_face_crop_bgr(crop_source_frame, x, y, w, h)
            quality_result = evaluate_pad_crop_quality(
                pad_face_crop,
                PAD_MIN_VARIANCE_LAPLACIAN,
            )
            laplacian_variance = quality_result["laplacian_variance"]

            # Only allow saving frames that pass the same structural and blur
            # checks used by the runtime PAD history update. Bbox
            # stability is treated here as an acquisition policy, not as
            # part of the PAD feature preprocessing path.
            valid_face = quality_result["is_crop_usable_for_history"] and bbox_stable

            # Draw face bounding box
            draw_visual_face_bbox(display_frame, face_rect, (0, 255, 0), 2)

            # Show a small crop preview so the user sees exactly what
            # will be stored when the current frame passes quality gates.
            if pad_face_crop_bgr is not None:
                cv2.imshow("PAD Face Crop", pad_face_crop_bgr)

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
            elif not bbox_stable:
                quality_text = "PAD frame: SKIPPED | unstable bbox"
                quality_color = (0, 165, 255)
            else:
                quality_text = (
                    f"PAD frame: READY | {quality_result['crop_width']}x"
                    f"{quality_result['crop_height']} | blur={laplacian_variance:.1f}"
                )
                quality_color = (0, 255, 0)

            # Show crop and detector size for quick sanity checks.
            cv2.putText(display_frame, f"Face size: {w}x{h}", (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(display_frame, quality_text, (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2, cv2.LINE_AA)

        cv2.imshow("Webcam", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("l") and valid_face:
            if live_count < MAX_IMAGES_LIVE:
                save_path = live_dataset / f"{live_next_idx:03d}.png"
                cv2.imwrite(str(save_path), pad_face_crop_bgr)
                print(f"[LIVE] Saved: {save_path.name}")
                live_count += 1
                live_next_idx += 1
            else:
                print("Maximum number of LIVE images reached")

        elif key == ord("s") and valid_face:
            if spoof_count < MAX_IMAGES_SPOOF:
                save_path = spoof_dataset / f"{spoof_next_idx:03d}.png"
                cv2.imwrite(str(save_path), pad_face_crop_bgr)
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
    collect_pad_custom_dataset()
