"""
Module: run_pipeline_bs

Description:
This module runs the biometric-systems runtime focused on face
recognition only.

The pipeline keeps the webcam loop intentionally simple:
- detect a valid face
- preprocess it for LBPH recognition
- apply a short temporal smoothing on recognition outputs
- display the current and stable identity on screen

No PAD gate is used here. This runtime is useful when recognition must
be tested in isolation from anti-spoofing.
"""

import cv2
import time

from src.antispoofing.pad_crop_utils import extract_pad_face_crop
from src.antispoofing.pad_frame_quality import (
    PAD_MIN_VARIANCE_LAPLACIAN,
    evaluate_bbox_stability,
    evaluate_pad_crop_quality,
)
from src.detection.face_detector import load_mediapipe_detector
from src.preprocessing.face_preprocessing import preprocess_face_for_recognition
from src.preprocessing.frame_preprocessing import get_valid_face
from src.recognition.face_recognizer import load_recognizer, predict_identity
from src.recognition.recognition_runtime_state import RecognitionRuntimeState


# ----- FACE DETECTION AND VALIDATION ----- #
THRESHOLD_AREA = 8000

# ----- RECOGNITION RUNTIME SETTINGS ----- #
RECOGNITION_WINDOW_SIZE = 5
RECOGNITION_MIN_CONFIRMATIONS = 3
RECOGNITION_RESET_AFTER_MISSES = 10
RECOGNITION_RUN_INTERVAL = 3
RUNTIME_RECOGNITION_THRESHOLD = 45.0
DISPLAY_KNOWN_ENTRY_VOTES = 3
DISPLAY_UNKNOWN_ENTRY_VOTES = 3
DISPLAY_KNOWN_EXIT_INCOHERENT = 4
DISPLAY_UNKNOWN_EXIT_INCOHERENT = 3
DISPLAY_HOLD_SECONDS = 1.0
DISPLAY_KNOWN_CONFIRM_TIMEOUT_SECONDS = 1.0


def format_identity_for_display(name):
    """
    Make runtime identity labels a bit more readable on screen.
    """

    if name is None:
        return None

    return name.replace("custom:", "custom: ")


class RecognitionDisplayState:
    """
    Slow visual state machine layered on top of the recognition votes.

    The webcam overlay must look calm even when frame-level predictions
    fluctuate. This state manager keeps transient uncertainty internal
    and only promotes stable labels to the visible overlay.
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
        self.uncertain_reason = "bootstrap"

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
            shown_text = self.get_identity_text()
            logger(
                "Overlay transition: "
                f"state={new_state} | text={shown_text} | reason={reason}"
            )

    def _hold_active(self, now):
        if self.state_since is None:
            return False
        return (now - self.state_since) < DISPLAY_HOLD_SECONDS

    def _set_uncertain_reason(self, reason):
        self.uncertain_reason = reason

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
                self._set_uncertain_reason("bootstrap")
            elif candidate_name not in (None, "Unknown"):
                self._set_uncertain_reason("conflict")
            else:
                self._set_uncertain_reason("unknown_pending")
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
            self._set_uncertain_reason("decay")

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
            self._set_uncertain_reason("conflict")
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
        Return the user-facing identity line for the current overlay.
        """

        if self.state == self.SEARCHING:
            return "Searching face..."
        if self.state == self.ANALYZING:
            return "Analyzing..."
        if self.state == self.SHOW_UNKNOWN:
            return "Identity: Unknown"
        return f"Identity: {format_identity_for_display(self.display_name)}"

    def get_identity_color(self):
        """
        Return the overlay color for the current identity line.
        """

        if self.state == self.SHOW_KNOWN:
            return (0, 255, 0)
        if self.state == self.SHOW_UNKNOWN:
            return (0, 0, 255)
        return (255, 255, 0)


def log_bs_event(message):
    """
    Print a concise recognition-runtime debug message.
    """

    print(f"[BS] {message}")


def build_recognition_log_signature(recognition_result):
    """
    Build the compact signature used to suppress repeated debug lines.
    """

    return (
        recognition_result["current_name"],
        recognition_result["candidate_name"],
        recognition_result["candidate_votes"],
        recognition_result["unknown_votes"],
        recognition_result["final_name"],
        recognition_result["stable_count"],
        recognition_result["history_size"],
    )


def create_recognition_runtime_state():
    """
    Build the temporal recognition state used by the recognition-only
    runtime.
    """

    return RecognitionRuntimeState(
        window_size=RECOGNITION_WINDOW_SIZE,
        min_confirmations=RECOGNITION_MIN_CONFIRMATIONS,
        reset_after_misses=RECOGNITION_RESET_AFTER_MISSES,
    )


def draw_recognition_overlay(
    frame,
    face_rect,
    recognition_text,
    identity_text,
    identity_color,
):
    """
    Draw a compact overlay for the recognition-only runtime.
    """

    x, y, w, h = face_rect
    top_line_y = max(25, y - 35)
    bottom_line_1_y = y + h + 25
    bottom_line_2_y = y + h + 55

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        frame,
        "State: RECOGNITION",
        (x, top_line_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        recognition_text,
        (x, bottom_line_1_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        identity_text,
        (x, bottom_line_2_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        identity_color,
        2,
        cv2.LINE_AA,
    )


def run_pipeline_bs():
    """
    Execute the recognition-only runtime pipeline.

    This entry point is meant for biometric-system experiments where
    the recognition module must be observed without interference from
    PAD logic.
    """

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    detector = load_mediapipe_detector()
    recognizer, label_map = load_recognizer()
    recognition_state = create_recognition_runtime_state()
    display_state = RecognitionDisplayState()
    last_recognition_result = {
        "final_name": "Unknown",
        "stable_confidence": None,
        "stable_count": 0,
        "history_size": 0,
        "candidate_name": None,
        "candidate_votes": 0,
        "unknown_votes": 0,
        "current_name": None,
        "current_confidence": None,
        "missing_face_counter": 0,
    }
    last_logged_signature = None
    previous_face_rect = None

    frame_counter = 0
    log_bs_event(
        f"Recognition runtime threshold: {RUNTIME_RECOGNITION_THRESHOLD:.2f}"
    )

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
            previous_face_rect = None
            recognition_result = recognition_state.update(face_missing=True)
            last_recognition_result = recognition_result
            display_state.update(
                face_present=False,
                recognition_result=recognition_result,
                now=time.monotonic(),
                logger=log_bs_event,
            )
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

            if recognition_result["final_name"] != "Unknown":
                cv2.putText(
                    frame,
                    f"Last identity: {recognition_result['final_name']}",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Recognition Runtime", frame)

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
        bbox_stable = stability_result["is_stable"]
        previous_face_rect = current_face_rect
        recognition_face_crop = extract_pad_face_crop(gray_frame, x, y, w, h)
        quality_result = evaluate_pad_crop_quality(
            recognition_face_crop,
            PAD_MIN_VARIANCE_LAPLACIAN,
        )
        should_run_recognition = (frame_counter % RECOGNITION_RUN_INTERVAL == 0)

        if should_run_recognition:
            recognition_frame_usable = (
                bbox_stable
                and quality_result["is_valid_crop"]
                and quality_result["is_sharp_enough"]
            )
            face = None
            if recognition_frame_usable:
                face = preprocess_face_for_recognition(gray_frame, x, y, w, h)

            if face is None:
                recognition_result = recognition_state.update(
                    frame_usable=False,
                    recognition_allowed=True,
                    face_missing=False,
                )
            else:
                predicted_name, confidence = predict_identity(
                    face,
                    recognizer,
                    label_map,
                    threshold=RUNTIME_RECOGNITION_THRESHOLD,
                )
                recognition_result = recognition_state.update(
                    predicted_name=predicted_name,
                    confidence=confidence,
                    frame_usable=True,
                    recognition_allowed=True,
                    face_missing=False,
                )
                confidence_text = (
                    f"{recognition_result['current_confidence']:.2f}"
                    if recognition_result["current_confidence"] is not None
                    else "None"
                )
                current_signature = build_recognition_log_signature(recognition_result)
                if current_signature != last_logged_signature:
                    log_bs_event(
                        "Recognition check: "
                        f"usable={recognition_frame_usable} | "
                        f"current={recognition_result['current_name']} | "
                        f"confidence={confidence_text} | "
                        f"candidate={recognition_result['candidate_name']} | "
                        f"candidate_votes={recognition_result['candidate_votes']} | "
                        f"unknown_votes={recognition_result['unknown_votes']} | "
                        f"final={recognition_result['final_name']} | "
                        f"votes={recognition_result['stable_count']}/"
                        f"{recognition_result['history_size']}"
                    )
                    last_logged_signature = current_signature
            last_recognition_result = recognition_result
        else:
            recognition_result = last_recognition_result

        display_state.update(
            face_present=True,
            recognition_result=recognition_result,
            now=time.monotonic(),
            logger=log_bs_event,
        )

        recognition_text = "Recognition active"
        identity_text = display_state.get_identity_text()
        identity_color = display_state.get_identity_color()

        draw_recognition_overlay(
            frame,
            face_rect,
            recognition_text,
            identity_text,
            identity_color,
        )

        cv2.imshow("Recognition Runtime", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
