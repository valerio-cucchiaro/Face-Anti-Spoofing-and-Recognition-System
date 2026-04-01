"""
Module: run_pipeline_cv

Description:
This module runs the computer-vision runtime focused on face
anti-spoofing only.

The pipeline detects a valid face, extracts the PAD crop, applies the
runtime PAD classifier, and smooths the result with ``PadRuntimeState``.
No recognition step is executed here. This makes the runtime useful for
testing liveness behavior and PAD responsiveness in isolation.
"""

import cv2

from src.antispoofing.pad_crop_utils import extract_pad_face_crop
from src.antispoofing.pad_frame_quality import (
    PAD_MIN_VARIANCE_LAPLACIAN,
    evaluate_bbox_stability,
    evaluate_pad_crop_quality,
)
from src.antispoofing.pad_runtime_detector import AntiSpoofingDetector
from src.antispoofing.pad_runtime_state import PadRuntimeState
from src.detection.face_detector import load_mediapipe_detector
from src.preprocessing.frame_preprocessing import get_valid_face


# ----- FACE DETECTION AND VALIDATION ----- #
THRESHOLD_AREA = 8000

# ----- PAD TEMPORAL SMOOTHING ----- #
PAD_WINDOW_SIZE = 5
PAD_RUN_INTERVAL = 3

# ----- PAD HYSTERESIS THRESHOLDS ----- #
PAD_ENTER_LIVE_THRESHOLD = 0.70
PAD_EXIT_LIVE_THRESHOLD = 0.55
PAD_ENTER_SPOOF_THRESHOLD = 0.30
PAD_EXIT_SPOOF_THRESHOLD = 0.45
PAD_RESET_AFTER_MISSES = 15


def log_cv_event(message):
    """
    Print a concise PAD-runtime debug message.
    """

    print(f"[CV] {message}")


def create_pad_runtime_state():
    """
    Build the PAD temporal state used by the anti-spoofing-only
    runtime.
    """

    return PadRuntimeState(
        window_size=PAD_WINDOW_SIZE,
        enter_live_threshold=PAD_ENTER_LIVE_THRESHOLD,
        exit_live_threshold=PAD_EXIT_LIVE_THRESHOLD,
        enter_spoof_threshold=PAD_ENTER_SPOOF_THRESHOLD,
        exit_spoof_threshold=PAD_EXIT_SPOOF_THRESHOLD,
        reset_after_misses=PAD_RESET_AFTER_MISSES,
    )


def draw_pad_overlay(frame, face_rect, box_color, pad_text, details_text):
    """
    Draw a compact overlay for the PAD-only runtime.
    """

    x, y, w, h = face_rect
    top_line_y = max(25, y - 35)
    bottom_line_1_y = y + h + 25
    bottom_line_2_y = y + h + 55

    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
    cv2.putText(
        frame,
        "State: PAD ONLY",
        (x, top_line_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        pad_text,
        (x, bottom_line_1_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        box_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        details_text,
        (x, bottom_line_2_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )


def format_timing_ms(value):
    """
    Format a timing value in milliseconds for overlay and console logs.
    """

    if value is None:
        return "None"

    return f"{value:.1f}ms"


def get_pad_box_color(final_pad_label):
    """
    Map the current PAD state to a bounding-box color.
    """

    if final_pad_label == "spoof":
        return (0, 0, 255)

    if final_pad_label == "live":
        return (0, 255, 0)

    return (0, 255, 255)


def run_pipeline_cv():
    """
    Execute the anti-spoofing-only runtime pipeline.

    This entry point is meant for computer-vision experiments where the
    PAD module must be observed without the recognition stage.
    """

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    detector = load_mediapipe_detector()
    pad_detector = AntiSpoofingDetector()
    pad_runtime_state = create_pad_runtime_state()

    frame_counter = 0
    previous_face_rect = None
    last_timing_text = "features: waiting"
    log_cv_event(f"PAD run interval: {PAD_RUN_INTERVAL}")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        frame_counter += 1
        face_rect, gray_frame, message = get_valid_face(
            frame,
            detector,
            THRESHOLD_AREA,
        )

        if message != "Face detected":
            pad_runtime_state.update(face_missing=True)
            previous_face_rect = None
            display_message = (
                "Searching face" if message == "No face detected" else message
            )
            cv2.putText(
                frame,
                display_message,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("PAD Runtime", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            continue

        x, y, w, h = face_rect
        current_face_rect = (x, y, w, h)
        stability_result = evaluate_bbox_stability(
            current_face_rect,
            previous_face_rect,
        )
        previous_face_rect = current_face_rect

        pad_face_crop = extract_pad_face_crop(gray_frame, x, y, w, h)
        quality_result = evaluate_pad_crop_quality(
            pad_face_crop,
            PAD_MIN_VARIANCE_LAPLACIAN,
        )
        pad_frame_usable = (
            quality_result["is_frame_usable_for_history"]
            and stability_result["is_stable"]
        )

        should_run_pad = (frame_counter % PAD_RUN_INTERVAL == 0)
        if should_run_pad:
            if quality_result["is_valid_crop"]:
                pad_result = pad_detector.predict(pad_face_crop)
            else:
                pad_result = {
                    "status": "invalid_crop",
                    "label": None,
                    "score": None,
                }

            pad_state_result = pad_runtime_state.update(
                pad_result=pad_result,
                frame_usable=pad_frame_usable,
                face_missing=False,
            )
            score_text = (
                f"{pad_state_result['current_score']:.2f}"
                if pad_state_result["current_score"] is not None
                else "None"
            )
            avg_live_text = (
                f"{pad_state_result['mean_live_probability']:.2f}"
                if pad_state_result["mean_live_probability"] is not None
                else "None"
            )
            pad_timings = pad_result.get("timings_ms", {})
            preprocess_ms = pad_timings.get("pad_preprocess")
            original_total_ms = pad_timings.get("original_total")
            downsampled_total_ms = pad_timings.get("downsampled_total")
            tgsd_ms = pad_timings.get("tgsd")
            lftp_ms = pad_timings.get("lftp")
            total_predict_ms = pad_timings.get("total_predict")
            last_timing_text = (
                f"pre={format_timing_ms(preprocess_ms)} | "
                f"orig={format_timing_ms(original_total_ms)} | "
                f"down={format_timing_ms(downsampled_total_ms)} | "
                f"tot={format_timing_ms(total_predict_ms)}"
            )
            log_cv_event(
                "PAD check: "
                f"usable={pad_frame_usable} | "
                f"current={pad_state_result['current_label']} | "
                f"score={score_text} | "
                f"final={pad_state_result['final_label']} | "
                f"avg_live={avg_live_text} | "
                f"pre={format_timing_ms(preprocess_ms)} | "
                f"orig_lbp={format_timing_ms(pad_timings.get('original_lbp'))} | "
                f"orig_quality={format_timing_ms(pad_timings.get('original_quality'))} | "
                f"orig_lft={format_timing_ms(pad_timings.get('original_lft'))} | "
                f"down_lbp={format_timing_ms(pad_timings.get('downsampled_lbp'))} | "
                f"down_quality={format_timing_ms(pad_timings.get('downsampled_quality'))} | "
                f"down_lft={format_timing_ms(pad_timings.get('downsampled_lft'))} | "
                f"tgsd={format_timing_ms(tgsd_ms)} | "
                f"lftp={format_timing_ms(lftp_ms)} | "
                f"tot={format_timing_ms(total_predict_ms)}"
            )
        else:
            pad_state_result = pad_runtime_state.update(
                pad_result=None,
                frame_usable=False,
                face_missing=False,
            )

        final_pad_label = pad_state_result["final_label"]
        mean_live_probability = pad_state_result["mean_live_probability"]

        if mean_live_probability is not None:
            pad_text = (
                f"PAD: {final_pad_label.upper()} | "
                f"avg_live={mean_live_probability:.2f}"
            )
        else:
            pad_text = f"PAD: {final_pad_label.upper()}"

        details_text = (
            f"crop_valid={quality_result['is_valid_crop']} | "
            f"sharp={quality_result['is_sharp_enough']}"
        )
        if should_run_pad:
            details_text = f"{details_text} | {last_timing_text}"
        draw_pad_overlay(
            frame,
            face_rect,
            get_pad_box_color(final_pad_label),
            pad_text,
            details_text,
        )

        cv2.imshow("PAD Runtime", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
