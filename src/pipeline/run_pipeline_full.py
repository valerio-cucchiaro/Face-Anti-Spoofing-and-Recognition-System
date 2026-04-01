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
from src.preprocessing.face_preprocessing import preprocess_face_for_recognition
from src.pipeline.runtime_session_state import RuntimeSessionState
from src.recognition.face_recognizer import load_recognizer, predict_identity
from src.recognition.recognition_runtime_state import RecognitionRuntimeState


# ----- FACE DETECTION AND VALIDATION ----- #
THRESHOLD_AREA = 8000

# ----- PAD TEMPORAL SMOOTHING ----- #
PAD_WINDOW_SIZE = 5
PAD_RUN_INTERVAL = 3
PAD_LIVE_LOCK_CONFIRMATIONS = 3

# ----- PAD HYSTERESIS THRESHOLDS ----- #
PAD_ENTER_LIVE_THRESHOLD = 0.70
PAD_EXIT_LIVE_THRESHOLD = 0.55
PAD_ENTER_SPOOF_THRESHOLD = 0.30
PAD_EXIT_SPOOF_THRESHOLD = 0.45

# ----- PAD STATE RESET ----- #
PAD_RESET_AFTER_MISSES = 15

# ----- RECOGNITION TEMPORAL SMOOTHING ----- #
RECOGNITION_WINDOW_SIZE = 5
RECOGNITION_MIN_CONFIRMATIONS = 3
RECOGNITION_RESET_AFTER_MISSES = 10
RECOGNITION_RUN_INTERVAL = 3

# ----- RECOGNITION THRESHOLDS ----- #
# Live webcam threshold chosen explicitly for runtime/demo usage.
RUNTIME_RECOGNITION_THRESHOLD = 45.0

# ----- SESSION RESET RULES ----- #
SESSION_RESET_AFTER_MISSES = 10
LOCK_RESET_AFTER_UNSTABLE_BBOX = 3
LOCK_RESET_AFTER_INVALID_CROP = 3

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


def create_recognition_runtime_state():
    """
    Build the recognition temporal state used by the runtime pipeline.
    """

    return RecognitionRuntimeState(
        window_size=RECOGNITION_WINDOW_SIZE,
        min_confirmations=RECOGNITION_MIN_CONFIRMATIONS,
        reset_after_misses=RECOGNITION_RESET_AFTER_MISSES,
    )


def log_runtime_event(message):
    """
    Print a concise runtime event to the console.
    """

    print(f"[RUNTIME] {message}")


def log_runtime_verbose(message):
    """
    Print a detailed runtime debug message only when verbose logging is
    enabled for the full pipeline.
    """

    if FULL_RUNTIME_VERBOSE_LOGS:
        print(f"[RUNTIME] {message}")


def reset_runtime_session(session_state):
    """
    Reset the runtime session and recreate the temporal PAD and
    recognition states.
    """

    session_state.reset_session()
    return create_pad_runtime_state(), create_recognition_runtime_state()


def draw_face_overlay(
    frame,
    face_rect,
    box_color,
    phase_text,
    pad_text,
    recognition_text,
    identity_text=None,
):
    """
    Draw a compact overlay around the current face box.

    The runtime keeps the on-screen overlay intentionally small and
    stable, while more detailed events are emitted to the console.
    """

    x, y, w, h = face_rect
    top_line_y = max(25, y - 55)
    second_top_line_y = max(50, y - 30)
    bottom_line_1_y = y + h + 25
    bottom_line_2_y = y + h + 55

    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
    cv2.putText(
        frame,
        phase_text,
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
        (x, second_top_line_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        box_color,
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

    if identity_text is not None:
        cv2.putText(
            frame,
            identity_text,
            (x, bottom_line_2_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
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

    detector = load_mediapipe_detector()
    pad_detector = AntiSpoofingDetector()
    recognizer, label_map = load_recognizer()
    runtime_threshold = RUNTIME_RECOGNITION_THRESHOLD

    session_state = RuntimeSessionState()
    pad_runtime_state = create_pad_runtime_state()
    recognition_runtime_state = create_recognition_runtime_state()

    frame_counter = 0
    log_runtime_event(
        f"Recognition runtime threshold: {runtime_threshold:.2f}"
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
            session_state.register_missing_face()
            session_state.previous_face_rect = None
            session_state.last_pad_result = pad_runtime_state.update(face_missing=True)
            session_state.store_recognition_result(
                recognition_runtime_state.update(face_missing=True)
            )

            if session_state.missing_face_counter >= SESSION_RESET_AFTER_MISSES:
                if session_state.transition_to(RuntimeSessionState.RESET):
                    log_runtime_event("Reset: face lost")
                pad_runtime_state, recognition_runtime_state = reset_runtime_session(
                    session_state
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
            cv2.imshow("Webcam", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            continue

        session_state.clear_missing_face_counter()

        x, y, w, h = face_rect
        current_face_rect = (x, y, w, h)
        stability_result = evaluate_bbox_stability(
            current_face_rect,
            session_state.previous_face_rect,
        )
        bbox_stable = stability_result["is_stable"]
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
            quality_result["is_frame_usable_for_history"] and bbox_stable
        )

        if session_state.current_phase == RuntimeSessionState.PAD_CHECK:
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
                session_state.store_pad_result(pad_state_result)

                if (
                    pad_state_result["final_label"] == "live"
                    and pad_frame_usable
                    and bbox_stable
                ):
                    live_count = session_state.register_pad_live_confirmation()
                else:
                    session_state.reset_pad_live_confirmation()
                    live_count = 0

                if live_count >= PAD_LIVE_LOCK_CONFIRMATIONS:
                    session_state.live_lock_active = True
                    session_state.transition_to(
                        RuntimeSessionState.RECOGNITION_LOCKED
                    )
                    log_runtime_event("PAD locked: LIVE")
                    log_runtime_event("Entering RECOGNITION_LOCKED")

            final_pad_label = session_state.last_pad_result["final_label"]
            mean_live_probability = session_state.last_pad_result[
                "mean_live_probability"
            ]
            if mean_live_probability is not None:
                pad_text = (
                    f"PAD: {final_pad_label.upper()} | avg_live="
                    f"{mean_live_probability:.2f}"
                )
            else:
                pad_text = f"PAD: {final_pad_label.upper()}"

            recognition_text = (
                "Checking liveness..."
                f" ({session_state.pad_live_confirmation_count}/"
                f"{PAD_LIVE_LOCK_CONFIRMATIONS})"
            )
            box_color = get_box_color(session_state, final_pad_label)
            draw_face_overlay(
                frame,
                face_rect,
                box_color,
                "State: CHECKING LIVENESS",
                pad_text,
                recognition_text,
            )

        elif session_state.current_phase == RuntimeSessionState.RECOGNITION_LOCKED:
            session_state.register_locked_bbox_stability(bbox_stable)
            session_state.register_locked_crop_validity(
                quality_result["is_valid_crop"]
            )

            if (
                session_state.locked_unstable_counter >= LOCK_RESET_AFTER_UNSTABLE_BBOX
            ):
                session_state.transition_to(RuntimeSessionState.RESET)
                log_runtime_event("Reset: abrupt movement")
            elif (
                session_state.locked_invalid_crop_counter
                >= LOCK_RESET_AFTER_INVALID_CROP
            ):
                session_state.transition_to(RuntimeSessionState.RESET)
                log_runtime_event("Reset: face crop lost")

            if session_state.current_phase == RuntimeSessionState.RESET:
                pad_runtime_state, recognition_runtime_state = reset_runtime_session(
                    session_state
                )
                draw_face_overlay(
                    frame,
                    face_rect,
                    (0, 165, 255),
                    "State: RESET",
                    "PAD: unlocked",
                    "Resetting session",
                )
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
                        predicted_name, confidence = predict_identity(
                            face,
                            recognizer,
                            label_map,
                            threshold=runtime_threshold,
                        )
                        recognition_result = recognition_runtime_state.update(
                            predicted_name=predicted_name,
                            confidence=confidence,
                            frame_usable=True,
                            recognition_allowed=True,
                            face_missing=False,
                        )

                    current_name = recognition_result["current_name"]
                    current_confidence = recognition_result["current_confidence"]
                    final_name = recognition_result["final_name"]
                    stable_count = recognition_result["stable_count"]
                    history_size = recognition_result["history_size"]
                    confidence_text = (
                        f"{current_confidence:.2f}"
                        if current_confidence is not None
                        else "None"
                    )
                    log_runtime_verbose(
                        "Recognition check: "
                        f"usable={recognition_frame_usable} | "
                        f"threshold={runtime_threshold:.2f} | "
                        f"current={current_name} | "
                        f"confidence={confidence_text} | "
                        f"final={final_name} | "
                        f"votes={stable_count}/{history_size}"
                    )

                    previous_identity = session_state.last_stable_identity
                    session_state.store_recognition_result(recognition_result)
                    current_identity = session_state.last_stable_identity
                    if (
                        current_identity != "Unknown"
                        and current_identity != previous_identity
                    ):
                        log_runtime_event(
                            f"Recognition result: {current_identity}"
                        )

                recognition_result = session_state.last_recognition_result
                final_identity = recognition_result["final_name"]
                if final_identity == "Unknown":
                    identity_text = "Identity: Unknown"
                else:
                    identity_text = f"Identity: {final_identity}"

                recognition_text = "Recognition active"

                draw_face_overlay(
                    frame,
                    face_rect,
                    get_box_color(session_state, "live"),
                    "State: LIVE LOCKED",
                    "PAD: LOCKED LIVE",
                    recognition_text,
                    identity_text=identity_text,
                )

        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
