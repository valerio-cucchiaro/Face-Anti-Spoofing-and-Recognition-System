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
import time

from src.antispoofing.pad_crop_utils import extract_pad_face_crop
from src.antispoofing.pad_feature_extractor import get_pad_feature_modes
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

# ----- PAD DISPLAY STATE MACHINE ----- #
PAD_STARTUP_DELAY_SECONDS = 2.0
DISPLAY_LIVE_ENTRY_CONFIRMATIONS = 2
DISPLAY_SPOOF_ENTRY_CONFIRMATIONS = 3
DISPLAY_LIVE_EXIT_INCOHERENT = 3
DISPLAY_SPOOF_EXIT_INCOHERENT = 2
DISPLAY_HOLD_SECONDS = 1.0
DISPLAY_PAD_MIN_ANALYZING_SECONDS = 1.0


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


class PadDisplayState:
    """
    Slow visual state machine layered on top of the PAD runtime state.

    The PAD classifier may start with harsh early predictions. This
    state manager keeps the first seconds in an explicit analyzing phase
    and only promotes stable live/spoof decisions to the visible
    overlay.
    """

    SEARCHING = "SEARCHING"
    ANALYZING = "ANALYZING"
    SHOW_LIVE = "SHOW_LIVE"
    SHOW_SPOOF = "SHOW_SPOOF"

    def __init__(self):
        self.state = self.SEARCHING
        self.state_since = None
        self.incoherent_updates = 0
        self.live_entry_counter = 0
        self.spoof_entry_counter = 0

    def _transition(self, new_state, now, logger, reason):
        if self.state != new_state:
            self.state = new_state
            self.state_since = now
            self.incoherent_updates = 0
            self.live_entry_counter = 0
            self.spoof_entry_counter = 0
            logger(
                "Overlay transition: "
                f"state={new_state} | text={self.get_pad_text()} | reason={reason}"
            )

    def _hold_active(self, now):
        if self.state_since is None:
            return False
        return (now - self.state_since) < DISPLAY_HOLD_SECONDS

    def _analyzing_min_time_active(self, now):
        if self.state_since is None:
            return False
        return (now - self.state_since) < DISPLAY_PAD_MIN_ANALYZING_SECONDS

    def update(self, face_present, pad_state_result, pad_frame_usable, now, logger):
        """
        Update the visible PAD overlay state.
        """

        if not face_present:
            self._transition(self.SEARCHING, now, logger, reason="face_missing")
            return

        final_label = pad_state_result["final_label"]

        if self.state == self.SEARCHING:
            self._transition(self.ANALYZING, now, logger, reason="face_detected")

        if self.state == self.ANALYZING:
            if self._analyzing_min_time_active(now):
                self.live_entry_counter = 0
                self.spoof_entry_counter = 0
                return

            if not pad_frame_usable:
                self.live_entry_counter = 0
                self.spoof_entry_counter = 0
                return

            if final_label == "live":
                self.live_entry_counter += 1
                self.spoof_entry_counter = 0
                if self.live_entry_counter >= DISPLAY_LIVE_ENTRY_CONFIRMATIONS:
                    self._transition(
                        self.SHOW_LIVE,
                        now,
                        logger,
                        reason="live_confirmed",
                    )
                return

            if final_label == "spoof":
                self.spoof_entry_counter += 1
                self.live_entry_counter = 0
                if self.spoof_entry_counter >= DISPLAY_SPOOF_ENTRY_CONFIRMATIONS:
                    self._transition(
                        self.SHOW_SPOOF,
                        now,
                        logger,
                        reason="spoof_confirmed",
                    )
                return

            self.live_entry_counter = 0
            self.spoof_entry_counter = 0
            return

        if self.state == self.SHOW_LIVE:
            if final_label == "live" and pad_frame_usable:
                self.incoherent_updates = 0
                return

            if self._hold_active(now):
                return

            if not pad_frame_usable or final_label in ("unknown", "spoof"):
                self.incoherent_updates += 1
                if self.incoherent_updates >= DISPLAY_LIVE_EXIT_INCOHERENT:
                    self._transition(self.ANALYZING, now, logger, reason="live_decay")
            return

        if self.state == self.SHOW_SPOOF:
            if final_label == "spoof" and pad_frame_usable:
                self.incoherent_updates = 0
                return

            if self._hold_active(now):
                return

            if not pad_frame_usable or final_label in ("unknown", "live"):
                self.incoherent_updates += 1
                if self.incoherent_updates >= DISPLAY_SPOOF_EXIT_INCOHERENT:
                    self._transition(self.ANALYZING, now, logger, reason="spoof_decay")

    def get_pad_text(self):
        """
        Return the user-facing PAD line for the current overlay state.
        """

        if self.state == self.SEARCHING:
            return "PAD: Searching face..."
        if self.state == self.ANALYZING:
            return "PAD: Analyzing..."
        if self.state == self.SHOW_LIVE:
            return "PAD: LIVE"
        return "PAD: SPOOF"

    def get_box_color(self):
        """
        Return the overlay color for the current PAD state.
        """

        if self.state == self.SHOW_LIVE:
            return (0, 255, 0)
        if self.state == self.SHOW_SPOOF:
            return (0, 0, 255)
        return (0, 255, 255)


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


def get_active_pad_feature_set_name(pad_config):
    """
    Build a human-readable description of the active PAD feature set.
    """

    debug_mode = pad_config["pad_debug_feature_mode"]
    base_name = "LBP + quality + GGD(LFT)_light (two-scale)"

    if debug_mode == "baseline":
        return base_name
    if debug_mode == "tgsd":
        return f"{base_name} + TGSD"

    return f"{debug_mode} + ggd_light"


def build_pad_timing_summary(pad_timings):
    """
    Format only the timing blocks that are actually active.
    """

    timing_parts = [
        f"pre={format_timing_ms(pad_timings.get('pad_preprocess'))}",
        f"orig={format_timing_ms(pad_timings.get('original_total'))}",
        f"down={format_timing_ms(pad_timings.get('downsampled_total'))}",
    ]

    if "tgsd" in pad_timings:
        timing_parts.append(f"tgsd={format_timing_ms(pad_timings.get('tgsd'))}")

    timing_parts.append(f"tot={format_timing_ms(pad_timings.get('total_predict'))}")
    return " | ".join(timing_parts)


def build_pad_log_signature(pad_state_result, pad_frame_usable, pad_timings):
    """
    Build a compact signature used to suppress repeated PAD log lines.
    """

    mean_live_probability = pad_state_result["mean_live_probability"]
    if mean_live_probability is not None:
        mean_live_probability = round(mean_live_probability, 1)

    return (
        pad_frame_usable,
        pad_state_result["current_label"],
        pad_state_result["final_label"],
        mean_live_probability,
    )


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
    display_state = PadDisplayState()
    active_feature_set = get_active_pad_feature_set_name(pad_detector.pad_config)
    configured_modes = get_pad_feature_modes()

    frame_counter = 0
    previous_face_rect = None
    last_timing_text = "features: waiting"
    last_logged_signature = None
    last_pad_state_result = {
        "final_label": "unknown",
        "mean_live_probability": None,
        "current_status": None,
        "current_label": None,
        "current_score": None,
        "missing_face_counter": 0,
    }
    startup_started_at = time.monotonic()
    log_cv_event(f"PAD run interval: {PAD_RUN_INTERVAL}")
    log_cv_event(
        "PAD runtime feature set: "
        f"{active_feature_set} | dim={pad_detector.pad_config['feature_dim']}"
    )
    log_cv_event(
        "PAD runtime modes: "
        f"debug={configured_modes['pad_debug_feature_mode']} | "
        f"lft={configured_modes['pad_lft_mode']} | "
        f"tgsd_tau={configured_modes.get('tgsd_tau')} | "
        f"tgsd_top_percent={configured_modes.get('tgsd_top_percent')} | "
        f"tgsd_grad={configured_modes.get('tgsd_gradient_operator')}"
    )

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        frame_counter += 1
        now = time.monotonic()

        if (now - startup_started_at) < PAD_STARTUP_DELAY_SECONDS:
            cv2.imshow("PAD Runtime", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            continue

        face_rect, gray_frame, message = get_valid_face(
            frame,
            detector,
            THRESHOLD_AREA,
        )

        if message != "Face detected":
            pad_runtime_state.update(face_missing=True)
            display_state.update(
                face_present=False,
                pad_state_result={
                    "final_label": "unknown",
                    "mean_live_probability": None,
                },
                pad_frame_usable=False,
                now=now,
                logger=log_cv_event,
            )
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
            last_pad_state_result = pad_state_result
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
            last_timing_text = build_pad_timing_summary(pad_timings)
            current_signature = build_pad_log_signature(
                pad_state_result,
                pad_frame_usable,
                pad_timings,
            )
            if current_signature != last_logged_signature:
                log_cv_event(
                    "PAD check: "
                    f"feature_set={active_feature_set} | "
                    f"usable={pad_frame_usable} | "
                    f"current={pad_state_result['current_label']} | "
                    f"score={score_text} | "
                    f"final={pad_state_result['final_label']} | "
                    f"avg_live={avg_live_text} | "
                    f"{last_timing_text}"
                )
                last_logged_signature = current_signature
        else:
            pad_state_result = last_pad_state_result

        final_pad_label = pad_state_result["final_label"]
        mean_live_probability = pad_state_result["mean_live_probability"]
        display_state.update(
            face_present=True,
            pad_state_result=pad_state_result,
            pad_frame_usable=pad_frame_usable,
            now=now,
            logger=log_cv_event,
        )

        pad_text = display_state.get_pad_text()
        if display_state.state in (PadDisplayState.SHOW_LIVE, PadDisplayState.SHOW_SPOOF):
            if mean_live_probability is not None:
                pad_text = f"{pad_text} | avg_live={mean_live_probability:.2f}"

        details_text = (
            f"crop_valid={quality_result['is_valid_crop']} | "
            f"sharp={quality_result['is_sharp_enough']}"
        )
        if should_run_pad:
            details_text = f"{details_text} | {last_timing_text}"
        draw_pad_overlay(
            frame,
            face_rect,
            display_state.get_box_color(),
            pad_text,
            details_text,
        )

        cv2.imshow("PAD Runtime", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
