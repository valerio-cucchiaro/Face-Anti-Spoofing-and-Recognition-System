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


def log_bs_event(message):
    """
    Print a concise recognition-runtime debug message.
    """

    print(f"[BS] {message}")


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
        (0, 255, 0),
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
    last_recognition_result = {
        "final_name": "Unknown",
        "stable_confidence": None,
        "stable_count": 0,
        "history_size": 0,
        "current_name": None,
        "current_confidence": None,
        "missing_face_counter": 0,
    }

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
            recognition_result = recognition_state.update(face_missing=True)
            last_recognition_result = recognition_result
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
        should_run_recognition = (frame_counter % RECOGNITION_RUN_INTERVAL == 0)

        if should_run_recognition:
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
                log_bs_event(
                    "Recognition check: "
                    f"current={recognition_result['current_name']} | "
                    f"confidence={confidence_text} | "
                    f"final={recognition_result['final_name']} | "
                    f"votes={recognition_result['stable_count']}/"
                    f"{recognition_result['history_size']}"
                )
            last_recognition_result = recognition_result
        else:
            recognition_result = last_recognition_result

        final_name = recognition_result["final_name"]
        recognition_text = "Recognition active"

        if final_name == "Unknown":
            identity_text = "Identity: Unknown"
        else:
            identity_text = f"Identity: {final_name}"

        draw_recognition_overlay(
            frame,
            face_rect,
            recognition_text,
            identity_text,
        )

        cv2.imshow("Recognition Runtime", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
