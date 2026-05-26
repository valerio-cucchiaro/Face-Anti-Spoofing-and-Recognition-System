"""
Module: run_pipeline_full

Description:
This module implements the complete real-time runtime pipeline of the
biometric face-recognition system.

The runtime is organized as a lightweight state machine:
- DETECT: search for a valid face
- PAD_CHECK: accumulate liveness evidence through PAD smoothing
- RECOGNITION_LOCKED: keep a stable live lock and run recognition
- RESET: clear the session and return to DETECT

The PAD temporal smoothing remains delegated to ``PadRuntimeState``,
while the recognition temporal smoothing remains delegated to
``RecognitionRuntimeState``. This module orchestrates when those two
subsystems should be used.
"""

import cv2
import time

from src.antispoofing.pad_crop_utils import extract_pad_face_crop
from src.antispoofing.pad_frame_quality import (
    PAD_MIN_VARIANCE_LAPLACIAN,
    evaluate_bbox_stability,
    evaluate_pad_crop_quality,
)
from src.antispoofing.pad_runtime_detector import AntiSpoofingDetector
from src.antispoofing.pad_runtime_state import PadRuntimeState
from src.detection.face_detector import load_mediapipe_detector
from src.preprocessing.face_bbox_visualization import draw_visual_face_bbox
from src.preprocessing.frame_preprocessing import DEFAULT_MIN_FACE_AREA, get_valid_face
from src.preprocessing.face_preprocessing import preprocess_face_for_recognition
from src.pipeline.runtime_session_state import RuntimeSessionState
from src.recognition.face_recognizer import load_recognizer, predict_identity_safe
from src.recognition.recognition_runtime_state import RecognitionRuntimeState


# ----- FACE DETECTION AND VALIDATION ----- #
THRESHOLD_AREA = DEFAULT_MIN_FACE_AREA
FACE_TOO_CLOSE_MAX_AREA = 80000

# ----- PAD TEMPORAL SMOOTHING ----- #
PAD_WINDOW_SIZE = 5
PAD_RUN_INTERVAL = 2
PAD_LIVE_LOCK_CONFIRMATIONS = 3
PAD_SPOOF_UNLOCK_CONFIRMATIONS = 3

# ----- PAD HYSTERESIS THRESHOLDS ----- #
PAD_ENTER_LIVE_THRESHOLD = 0.65
PAD_EXIT_LIVE_THRESHOLD = 0.55
PAD_ENTER_SPOOF_THRESHOLD = 0.30
PAD_EXIT_SPOOF_THRESHOLD = 0.45
PAD_STARTUP_DELAY_SECONDS = 2.0
DISPLAY_LIVE_ENTRY_CONFIRMATIONS = 2
DISPLAY_SPOOF_ENTRY_CONFIRMATIONS = 3
DISPLAY_LIVE_EXIT_INCOHERENT = 3
DISPLAY_SPOOF_EXIT_INCOHERENT = 2
DISPLAY_PAD_HOLD_SECONDS = 1.0
DISPLAY_PAD_MIN_ANALYZING_SECONDS = 1.0
FACE_MESSAGE_HOLD_SECONDS = 2.0
FACE_REACQUIRE_CONFIRMATIONS = 5
FACE_ACQUISITION_MESSAGE = "Move closer and center your face"
FACE_TOO_CLOSE_MESSAGE = "Move back and center your face"

# ----- PAD STATE RESET ----- #
PAD_RESET_AFTER_MISSES = 15

# ----- RECOGNITION TEMPORAL SMOOTHING ----- #
RECOGNITION_WINDOW_SIZE = 5
RECOGNITION_MIN_CONFIRMATIONS = 3
RECOGNITION_RESET_AFTER_MISSES = 10
RECOGNITION_RUN_INTERVAL = 3
DISPLAY_KNOWN_ENTRY_VOTES = 3
DISPLAY_UNKNOWN_ENTRY_VOTES = 3
DISPLAY_KNOWN_EXIT_INCOHERENT = 4
DISPLAY_UNKNOWN_EXIT_INCOHERENT = 3
DISPLAY_RECOGNITION_HOLD_SECONDS = 1.0
DISPLAY_KNOWN_CONFIRM_TIMEOUT_SECONDS = 1.0
RECOGNITION_START_DELAY_AFTER_LIVE_SECONDS = 1.0

# ----- RECOGNITION THRESHOLDS ----- #
# Live webcam threshold chosen explicitly for runtime/demo usage.
RUNTIME_RECOGNITION_THRESHOLD = 44.0

# ----- SESSION RESET RULES ----- #
SESSION_RESET_AFTER_MISSES = 10
LOCK_RESET_AFTER_MISSING_FACE = 2
LOCK_RESET_AFTER_MOVEMENT = 1
LOCK_RESET_AFTER_INVALID_CROP = 1
LOCK_MOVEMENT_MAX_DELTA_X = 18
LOCK_MOVEMENT_MAX_DELTA_Y = 18
LOCK_MOVEMENT_MAX_DELTA_W = 24
LOCK_MOVEMENT_MAX_DELTA_H = 24

# ----- CAMERA READ RECOVERY ----- #
CAMERA_READ_MAX_FAILURES = 10

# ----- RUNTIME DEBUG ----- #
FULL_RUNTIME_VERBOSE_LOGS = False


def create_pad_runtime_state():
    """
    Build the PAD temporal state used by the runtime pipeline.
    """

    return PadRuntimeState(
        window_size=PAD_WINDOW_SIZE,
        enter_live_threshold=PAD_ENTER_LIVE_THRESHOLD,
        exit_live_threshold=PAD_EXIT_LIVE_THRESHOLD,
        enter_spoof_threshold=PAD_ENTER_SPOOF_THRESHOLD,
        exit_spoof_threshold=PAD_EXIT_SPOOF_THRESHOLD,
        reset_after_misses=PAD_RESET_AFTER_MISSES,
    )


def format_identity_for_display(name):
    """
    Make runtime identity labels a bit more readable on screen.
    """

    if name is None:
        return None

    return name.replace("custom:", "custom: ")


class PadDisplayState:
    """
    Slow visual PAD state layered on top of the PAD runtime smoothing.
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
                "PAD overlay: "
                f"state={new_state} | text={self.get_pad_text()} | reason={reason}"
            )

    def _hold_active(self, now):
        if self.state_since is None:
            return False
        return (now - self.state_since) < DISPLAY_PAD_HOLD_SECONDS

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
                    self._transition(self.SHOW_LIVE, now, logger, reason="live_confirmed")
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


def create_recognition_runtime_state():
    """
    Build the recognition temporal state used by the runtime pipeline.
    """

    return RecognitionRuntimeState(
        window_size=RECOGNITION_WINDOW_SIZE,
        min_confirmations=RECOGNITION_MIN_CONFIRMATIONS,
        reset_after_misses=RECOGNITION_RESET_AFTER_MISSES,
    )


class RecognitionDisplayState:
    """
    Slow visual recognition state layered on top of recognition votes.
    """

    SEARCHING = "SEARCHING"
    ANALYZING = "ANALYZING"
    SHOW_KNOWN = "SHOW_KNOWN"
    SHOW_UNKNOWN = "SHOW_UNKNOWN"

    def __init__(self):
        self.state = self.SEARCHING
        self.display_name = None
        self.state_since = None
        self.last_known_reinforcement = None
        self.incoherent_updates = 0

    def _transition(self, new_state, now, logger, reason, display_name=None):
        state_changed = self.state != new_state
        name_changed = self.display_name != display_name

        self.state = new_state
        self.display_name = display_name
        self.state_since = now
        self.incoherent_updates = 0

        if new_state == self.SHOW_KNOWN:
            self.last_known_reinforcement = now
        elif new_state != self.SHOW_KNOWN:
            self.last_known_reinforcement = None

        if state_changed or name_changed:
            logger(
                "Recognition overlay: "
                f"state={new_state} | text={self.get_identity_text()} | reason={reason}"
            )

    def _hold_active(self, now):
        if self.state_since is None:
            return False
        return (now - self.state_since) < DISPLAY_RECOGNITION_HOLD_SECONDS

    def update(self, face_present, recognition_result, now, logger):
        """
        Update the visible recognition overlay state.
        """

        if not face_present:
            self._transition(
                self.SEARCHING,
                now,
                logger,
                reason="face_missing",
                display_name=None,
            )
            return

        final_name = recognition_result["final_name"]
        current_name = recognition_result["current_name"]
        candidate_name = recognition_result.get("candidate_name")
        candidate_votes = recognition_result.get("candidate_votes", 0)
        unknown_votes = recognition_result.get("unknown_votes", 0)
        history_size = recognition_result["history_size"]

        if self.state == self.SEARCHING:
            self._transition(
                self.ANALYZING,
                now,
                logger,
                reason="face_detected",
                display_name=None,
            )

        if self.state == self.ANALYZING:
            if final_name != "Unknown" and candidate_votes >= DISPLAY_KNOWN_ENTRY_VOTES:
                self._transition(
                    self.SHOW_KNOWN,
                    now,
                    logger,
                    reason=f"known_confirmed:{final_name}",
                    display_name=final_name,
                )
                return

            if (
                candidate_name == "Unknown"
                and unknown_votes >= DISPLAY_UNKNOWN_ENTRY_VOTES
            ):
                self._transition(
                    self.SHOW_UNKNOWN,
                    now,
                    logger,
                    reason="unknown_confirmed",
                    display_name=None,
                )
                return

            if history_size < DISPLAY_KNOWN_ENTRY_VOTES:
                return
            return

        if self.state == self.SHOW_KNOWN:
            shown_name = self.display_name
            same_name_confirmed = final_name == shown_name
            same_name_pending = current_name == shown_name or candidate_name == shown_name

            if same_name_confirmed:
                self.last_known_reinforcement = now
                self.incoherent_updates = 0
                return

            if self._hold_active(now):
                return

            if same_name_pending:
                self.incoherent_updates = 0
                return

            self.incoherent_updates += 1

            confirmation_timed_out = (
                self.last_known_reinforcement is not None
                and (now - self.last_known_reinforcement)
                >= DISPLAY_KNOWN_CONFIRM_TIMEOUT_SECONDS
            )
            unknown_is_persistent = (
                candidate_name == "Unknown"
                and unknown_votes >= DISPLAY_UNKNOWN_ENTRY_VOTES
            )

            if (
                self.incoherent_updates >= DISPLAY_KNOWN_EXIT_INCOHERENT
                or confirmation_timed_out
                or unknown_is_persistent
            ):
                self._transition(
                    self.ANALYZING,
                    now,
                    logger,
                    reason="known_decay",
                    display_name=None,
                )
            return

        if self.state == self.SHOW_UNKNOWN:
            unknown_still_confirmed = (
                candidate_name == "Unknown"
                and unknown_votes >= DISPLAY_UNKNOWN_ENTRY_VOTES
            )
            known_is_confirmed = (
                final_name != "Unknown"
                and candidate_votes >= DISPLAY_KNOWN_ENTRY_VOTES
            )

            if unknown_still_confirmed:
                self.incoherent_updates = 0
                return

            if self._hold_active(now):
                return

            if known_is_confirmed:
                self._transition(
                    self.ANALYZING,
                    now,
                    logger,
                    reason=f"known_candidate_after_unknown:{final_name}",
                    display_name=None,
                )
                return

            self.incoherent_updates += 1
            if self.incoherent_updates >= DISPLAY_UNKNOWN_EXIT_INCOHERENT:
                self._transition(
                    self.ANALYZING,
                    now,
                    logger,
                    reason="unknown_released",
                    display_name=None,
                )

    def get_identity_text(self):
        """
        Return the user-facing recognition line for the current overlay.
        """

        if self.state == self.SEARCHING:
            return "Identity: pending"
        if self.state == self.ANALYZING:
            return "Identity: pending"
        if self.state == self.SHOW_UNKNOWN:
            return "Identity: Unknown"
        return f"Identity: {format_identity_for_display(self.display_name)}"

    def get_identity_color(self):
        """
        Return the overlay color for the recognition identity line.
        """

        if self.state == self.SHOW_KNOWN:
            return (0, 255, 0)
        if self.state == self.SHOW_UNKNOWN:
            return (0, 0, 255)
        return (255, 255, 0)


def log_runtime_event(message):
    """
    Print a concise runtime event to the console.
    """

    print(f"[APP] {message}")


def log_runtime_verbose(message):
    """
    Print a detailed runtime debug message only when verbose logging is
    enabled for the full pipeline.
    """

    if FULL_RUNTIME_VERBOSE_LOGS:
        print(f"[RUNTIME] {message}")


class FaceAcquisitionDisplayState:
    """
    Stabilize the visual feedback shown before the face overlay appears.
    """

    def __init__(self):
        self.warning_text = None
        self.warning_started_at = None
        self.reacquire_counter = 0

    def register_warning(self, message, now, logger=None, reason="face_lost"):
        if self.warning_text != message:
            self.warning_text = message
            self.warning_started_at = now
            if logger is not None:
                logger(f"Face acquisition: {reason} | text={message}")

        self.reacquire_counter = 0
        return self.warning_text

    def should_show_face_overlay(self, now, logger=None):
        if self.warning_text is None:
            return True

        if (
            self.warning_started_at is not None
            and (now - self.warning_started_at) < FACE_MESSAGE_HOLD_SECONDS
        ):
            return False

        self.reacquire_counter += 1
        if self.reacquire_counter >= FACE_REACQUIRE_CONFIRMATIONS:
            self.warning_text = None
            self.warning_started_at = None
            self.reacquire_counter = 0
            if logger is not None:
                logger("Face acquisition: overlay resumed")
            return True

        return False

    def get_warning_text(self):
        return self.warning_text or FACE_ACQUISITION_MESSAGE


def reset_runtime_session(session_state):
    """
    Reset the runtime session and recreate the temporal PAD and
    recognition states.
    """

    session_state.reset_session()
    return (
        create_pad_runtime_state(),
        create_recognition_runtime_state(),
        PadDisplayState(),
        RecognitionDisplayState(),
    )


def draw_face_acquisition_message(frame, message):
    """
    Draw the pre-overlay face acquisition warning.
    """

    cv2.putText(
        frame,
        message,
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def draw_face_overlay(
    frame,
    face_rect,
    box_color,
    phase_text,
    pad_text,
    recognition_text,
    identity_text=None,
    identity_color=(0, 255, 0),
):
    """
    Draw a compact overlay around the current face box.

    The runtime keeps the on-screen overlay intentionally small and
    stable, while more detailed events are emitted to the console.
    """

    x, y, w, h = face_rect
    frame_width = frame.shape[1]
    box_center_x = x + (w // 2)
    top_line_y = max(25, y - 72)
    second_top_line_y = max(50, y - 44)
    bottom_line_1_y = y + h + 25
    bottom_line_2_y = y + h + 55

    def centered_text_origin(text, font_scale, thickness):
        text_size, _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        text_width = text_size[0]
        text_x = int(box_center_x - (text_width / 2))
        return max(5, min(text_x, frame_width - text_width - 5))

    draw_visual_face_bbox(frame, face_rect, box_color, 2)
    cv2.putText(
        frame,
        phase_text,
        (centered_text_origin(phase_text, 0.6, 2), top_line_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        pad_text,
        (centered_text_origin(pad_text, 0.7, 2), second_top_line_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        box_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        recognition_text,
        (centered_text_origin(recognition_text, 0.7, 2), bottom_line_1_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if identity_text is not None:
        cv2.putText(
            frame,
            identity_text,
            (centered_text_origin(identity_text, 0.7, 2), bottom_line_2_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            identity_color,
            2,
            cv2.LINE_AA,
        )


def get_box_color(session_state, final_pad_label):
    """
    Select the bounding-box color from the runtime phase and PAD state.
    """

    if session_state.current_phase == RuntimeSessionState.RECOGNITION_LOCKED:
        return (0, 255, 0)

    if final_pad_label == "spoof":
        return (0, 0, 255)

    if final_pad_label == "live":
        return (0, 255, 255)

    return (0, 255, 255)


def run_pipeline_full():
    """
    Execute the full real-time biometric runtime pipeline.

    The runtime first confirms liveness through PAD. Only after a stable
    live lock is reached does it open a dedicated recognition phase,
    where recognition can accumulate evidence without being disrupted by
    continuous heavy PAD reevaluation.
    """

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    try:
        detector = load_mediapipe_detector()
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize the MediaPipe face detector. "
            "Check the MediaPipe dependency and runtime environment."
        ) from exc
    try:
        pad_detector = AntiSpoofingDetector()
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize PAD runtime detector. "
            "Check PAD model, scaler, classifier, and config artifacts."
        ) from exc

    try:
        recognizer, label_map = load_recognizer()
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize recognition runtime model. "
            "Check LBPH model, label_map.pkl, and recognizer_config.json."
        ) from exc
    runtime_threshold = RUNTIME_RECOGNITION_THRESHOLD

    session_state = RuntimeSessionState()
    pad_runtime_state = create_pad_runtime_state()
    recognition_runtime_state = create_recognition_runtime_state()
    pad_display_state = PadDisplayState()
    recognition_display_state = RecognitionDisplayState()
    face_acquisition_display_state = FaceAcquisitionDisplayState()

    frame_counter = 0
    camera_read_failures = 0
    startup_started_at = time.monotonic()
    log_runtime_event(
        f"Recognition runtime threshold: {runtime_threshold:.2f}"
    )
    log_runtime_event("Runtime initialized: waiting for a centered face")

    while True:
        ret, frame = cap.read()

        if not ret:
            camera_read_failures += 1
            log_runtime_event(
                "Webcam frame read failed: "
                f"{camera_read_failures}/{CAMERA_READ_MAX_FAILURES}"
            )
            if camera_read_failures >= CAMERA_READ_MAX_FAILURES:
                raise RuntimeError(
                    "Webcam read failed repeatedly; stopping runtime session"
                )
            continue

        camera_read_failures = 0

        frame_counter += 1
        now = time.monotonic()

        if (now - startup_started_at) < PAD_STARTUP_DELAY_SECONDS:
            cv2.imshow("Webcam", frame)

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
            session_state.register_missing_face()
            session_state.previous_face_rect = None
            session_state.last_pad_result = pad_runtime_state.update(face_missing=True)
            session_state.store_recognition_result(
                recognition_runtime_state.update(face_missing=True)
            )
            pad_display_state.update(
                face_present=False,
                pad_state_result={
                    "final_label": "unknown",
                    "mean_live_probability": None,
                },
                pad_frame_usable=False,
                now=now,
                logger=log_runtime_event,
            )
            recognition_display_state.update(
                face_present=False,
                recognition_result=session_state.last_recognition_result,
                now=now,
                logger=log_runtime_event,
            )

            reset_after_misses = SESSION_RESET_AFTER_MISSES
            if session_state.current_phase == RuntimeSessionState.RECOGNITION_LOCKED:
                reset_after_misses = LOCK_RESET_AFTER_MISSING_FACE

            if session_state.missing_face_counter >= reset_after_misses:
                session_state.transition_to(RuntimeSessionState.RESET)
                (
                    pad_runtime_state,
                    recognition_runtime_state,
                    pad_display_state,
                    recognition_display_state,
                ) = reset_runtime_session(session_state)

            display_message = face_acquisition_display_state.register_warning(
                FACE_ACQUISITION_MESSAGE,
                now,
                logger=log_runtime_event,
                reason="face_missing",
            )
            draw_face_acquisition_message(frame, display_message)
            cv2.imshow("Webcam", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            continue

        x, y, w, h = face_rect
        if (w * h) > FACE_TOO_CLOSE_MAX_AREA:
            # "Too close" is a demo guidance condition: it asks the user to
            # move back so the crop remains usable, but it is not treated as
            # spoof evidence.
            session_state.register_missing_face()
            session_state.previous_face_rect = None
            session_state.last_pad_result = pad_runtime_state.update(face_missing=True)
            session_state.store_recognition_result(
                recognition_runtime_state.update(face_missing=True)
            )
            pad_display_state.update(
                face_present=False,
                pad_state_result={
                    "final_label": "unknown",
                    "mean_live_probability": None,
                },
                pad_frame_usable=False,
                now=now,
                logger=log_runtime_event,
            )
            recognition_display_state.update(
                face_present=False,
                recognition_result=session_state.last_recognition_result,
                now=now,
                logger=log_runtime_event,
            )

            reset_after_misses = SESSION_RESET_AFTER_MISSES
            if session_state.current_phase == RuntimeSessionState.RECOGNITION_LOCKED:
                reset_after_misses = LOCK_RESET_AFTER_MISSING_FACE

            if session_state.missing_face_counter >= reset_after_misses:
                session_state.transition_to(RuntimeSessionState.RESET)
                (
                    pad_runtime_state,
                    recognition_runtime_state,
                    pad_display_state,
                    recognition_display_state,
                ) = reset_runtime_session(session_state)

            display_message = face_acquisition_display_state.register_warning(
                FACE_TOO_CLOSE_MESSAGE,
                now,
                logger=log_runtime_event,
                reason="too_close",
            )
            draw_face_acquisition_message(frame, display_message)
            cv2.imshow("Webcam", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            continue

        if not face_acquisition_display_state.should_show_face_overlay(
            now,
            logger=log_runtime_event,
        ):
            draw_face_acquisition_message(
                frame,
                face_acquisition_display_state.get_warning_text(),
            )
            cv2.imshow("Webcam", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            continue

        session_state.clear_missing_face_counter()

        current_face_rect = (x, y, w, h)
        stability_result = evaluate_bbox_stability(
            current_face_rect,
            session_state.previous_face_rect,
        )
        bbox_stable = stability_result["is_stable"]
        movement_result = evaluate_bbox_stability(
            current_face_rect,
            session_state.previous_face_rect,
            max_delta_x=LOCK_MOVEMENT_MAX_DELTA_X,
            max_delta_y=LOCK_MOVEMENT_MAX_DELTA_Y,
            max_delta_w=LOCK_MOVEMENT_MAX_DELTA_W,
            max_delta_h=LOCK_MOVEMENT_MAX_DELTA_H,
        )
        movement_detected = not movement_result["is_stable"]
        session_state.store_face_rect(current_face_rect)

        if session_state.current_phase == RuntimeSessionState.DETECT:
            if session_state.transition_to(RuntimeSessionState.PAD_CHECK):
                log_runtime_event("Entering PAD_CHECK")

        pad_face_crop = extract_pad_face_crop(gray_frame, x, y, w, h)
        quality_result = evaluate_pad_crop_quality(
            pad_face_crop,
            PAD_MIN_VARIANCE_LAPLACIAN,
        )
        pad_frame_usable = (
            quality_result["is_crop_usable_for_history"] and bbox_stable
        )

        if session_state.current_phase == RuntimeSessionState.PAD_CHECK:
            should_run_pad = (frame_counter % PAD_RUN_INTERVAL == 0)

            if should_run_pad:
                if quality_result["is_valid_crop"]:
                    pad_result = pad_detector.predict_safe(pad_face_crop)
                    if pad_result["status"] == "predict_failed":
                        log_runtime_event(
                            "PAD prediction failed: "
                            f"error={pad_result.get('error')}"
                        )
                else:
                    pad_result = {
                        "status": "invalid_crop",
                        "label": None,
                        "score": None,
                    }
            else:
                pad_result = None

            if should_run_pad:
                pad_state_result = pad_runtime_state.update(
                    pad_result=pad_result,
                    frame_usable=pad_frame_usable,
                    face_missing=False,
                )
                session_state.store_pad_result(pad_state_result)
            else:
                pad_state_result = session_state.last_pad_result

            pad_display_state.update(
                face_present=True,
                pad_state_result=session_state.last_pad_result,
                pad_frame_usable=pad_frame_usable,
                now=now,
                logger=log_runtime_event,
            )

            if should_run_pad:
                if (
                    pad_display_state.state == PadDisplayState.SHOW_LIVE
                    and session_state.last_pad_result["final_label"] == "live"
                    and pad_frame_usable
                    and bbox_stable
                ):
                    live_count = session_state.register_pad_live_confirmation()
                else:
                    session_state.reset_pad_live_confirmation()
                    live_count = 0

                if live_count >= PAD_LIVE_LOCK_CONFIRMATIONS:
                    session_state.live_lock_started_at = now
                    session_state.transition_to(
                        RuntimeSessionState.RECOGNITION_LOCKED
                    )
                    log_runtime_event("PAD locked: LIVE")
                    log_runtime_event("Entering RECOGNITION_LOCKED")

            pad_text = pad_display_state.get_pad_text()

            recognition_text = "Checking liveness..."
            draw_face_overlay(
                frame,
                face_rect,
                pad_display_state.get_box_color(),
                "State: CHECKING LIVENESS",
                pad_text,
                recognition_text,
            )

        elif session_state.current_phase == RuntimeSessionState.RECOGNITION_LOCKED:
            session_state.register_locked_movement(movement_detected)
            session_state.register_locked_crop_validity(quality_result["is_valid_crop"])

            if (
                session_state.locked_movement_counter
                >= LOCK_RESET_AFTER_MOVEMENT
            ):
                if session_state.transition_to(RuntimeSessionState.RESET):
                    log_runtime_event("Reset: face moved")
            elif (
                session_state.locked_invalid_crop_counter
                >= LOCK_RESET_AFTER_INVALID_CROP
            ):
                if session_state.transition_to(RuntimeSessionState.RESET):
                    log_runtime_event("Reset: invalid crop")

            if session_state.current_phase == RuntimeSessionState.RESET:
                (
                    pad_runtime_state,
                    recognition_runtime_state,
                    pad_display_state,
                    recognition_display_state,
                ) = reset_runtime_session(session_state)
            else:
                should_run_locked_pad = (frame_counter % PAD_RUN_INTERVAL == 0)
                if should_run_locked_pad:
                    if quality_result["is_valid_crop"]:
                        locked_pad_result = pad_detector.predict_safe(pad_face_crop)
                        if locked_pad_result["status"] == "predict_failed":
                            log_runtime_event(
                                "Locked PAD re-check failed: "
                                f"error={locked_pad_result.get('error')}"
                            )
                    else:
                        locked_pad_result = {
                            "status": "invalid_crop",
                            "label": None,
                            "score": None,
                        }

                    locked_pad_state_result = pad_runtime_state.update(
                        pad_result=locked_pad_result,
                        frame_usable=pad_frame_usable,
                        face_missing=False,
                    )
                    session_state.store_pad_result(locked_pad_state_result)

                    # Leaving the live lock for spoof requires stronger
                    # evidence than merely keeping the session active:
                    # the PAD result must point to spoof and the frame must
                    # still be fully usable for temporal PAD updates.
                    spoof_evidence = (
                        locked_pad_state_result["final_label"] == "spoof"
                        and pad_frame_usable
                        and bbox_stable
                    )
                    spoof_count = session_state.register_locked_spoof_evidence(
                        spoof_evidence
                    )

                    log_runtime_verbose(
                        "Locked PAD re-check: "
                        f"usable={pad_frame_usable} | "
                        f"final={locked_pad_state_result['final_label']} | "
                        f"mean_live="
                        f"{locked_pad_state_result['mean_live_probability']} | "
                        f"spoof_votes={spoof_count}/"
                        f"{PAD_SPOOF_UNLOCK_CONFIRMATIONS}"
                    )

                    if spoof_count >= PAD_SPOOF_UNLOCK_CONFIRMATIONS:
                        if session_state.transition_to(RuntimeSessionState.RESET):
                            log_runtime_event(
                                "Reset: locked PAD re-check confirmed SPOOF"
                            )

                if session_state.current_phase == RuntimeSessionState.RESET:
                    (
                        pad_runtime_state,
                        recognition_runtime_state,
                        pad_display_state,
                        recognition_display_state,
                    ) = reset_runtime_session(session_state)
                    cv2.imshow("Webcam", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    continue

                recognition_delay_active = (
                    session_state.live_lock_started_at is not None
                    and (now - session_state.live_lock_started_at)
                    < RECOGNITION_START_DELAY_AFTER_LIVE_SECONDS
                )

                if recognition_delay_active:
                    recognition_text = "Recognition: waiting..."
                    identity_text = "Identity: pending"
                    identity_color = (255, 255, 0)
                else:
                    should_run_recognition = (
                        frame_counter % RECOGNITION_RUN_INTERVAL == 0
                    )
                    recognition_frame_usable = (
                        bbox_stable
                        and quality_result["is_valid_crop"]
                        and quality_result["is_sharp_enough"]
                    )
                    if should_run_recognition:
                        face = None
                        if recognition_frame_usable:
                            face = preprocess_face_for_recognition(gray_frame, x, y, w, h)

                        if face is None:
                            recognition_result = recognition_runtime_state.update(
                                frame_usable=False,
                                recognition_allowed=True,
                                face_missing=False,
                            )
                        else:
                            prediction = predict_identity_safe(
                                face,
                                recognizer,
                                label_map,
                                threshold=runtime_threshold,
                            )
                            recognition_result = recognition_runtime_state.update(
                                predicted_name=prediction["predicted_name"],
                                confidence=prediction["confidence"],
                                frame_usable=True,
                                recognition_allowed=True,
                                face_missing=False,
                            )
                            if prediction["failure_status"] is not None:
                                log_runtime_event(
                                    "Recognition prediction failed: "
                                    f"status={prediction['failure_status']} | "
                                    f"error={prediction['error']}"
                                )

                        log_runtime_verbose(
                            "Recognition check: "
                            f"usable={recognition_frame_usable} | "
                            f"threshold={runtime_threshold:.2f} | "
                            f"current={recognition_result['current_name']} | "
                            f"distance={recognition_result['current_confidence']} | "
                            f"final={recognition_result['final_name']} | "
                            f"votes={recognition_result['stable_count']}/"
                            f"{recognition_result['history_size']}"
                        )

                        session_state.store_recognition_result(recognition_result)

                    recognition_result = session_state.last_recognition_result
                    recognition_display_state.update(
                        face_present=True,
                        recognition_result=recognition_result,
                        now=now,
                        logger=log_runtime_event,
                    )
                    recognition_text = "Recognition active"
                    identity_text = recognition_display_state.get_identity_text()
                    identity_color = recognition_display_state.get_identity_color()

                draw_face_overlay(
                    frame,
                    face_rect,
                    get_box_color(session_state, "live"),
                    "State: LIVE LOCKED",
                    "PAD: LOCKED LIVE",
                    recognition_text,
                    identity_text=identity_text,
                    identity_color=identity_color,
                )

        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    log_runtime_event("Runtime stopped")
