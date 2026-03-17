"""
Module: run_pipeline_full

Description:
This module implements the complete runtime pipeline of the biometric
face recognition system. It orchestrates the main processing flow of
the application using the system webcam.

The pipeline performs the following stages:

webcam acquisition
→ face detection
→ face validation
→ face preprocessing
→ face antispoofing
→ face recognition
→ visualization of results

The module integrates the detection, preprocessing, and recognition
components and displays the predicted identity and confidence score
in real time.
"""

import cv2
from pathlib import Path
from collections import deque

from src.detection.face_detector import load_face_detector
from src.preprocessing.face_preprocessing import preprocess_face_for_recognition
from src.preprocessing.frame_preprocessing import get_valid_face
from src.recognition.face_recognizer import load_recognizer, predict_identity
from src.antispoofing.spoof_detector import AntiSpoofingDetector


# ----- CONSTANTS ----- #
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "haarcascade_frontalface_default.xml"
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 6
MIN_SIZE = (30, 30)
THRESHOLD_AREA = 3000   # minimum area of the detected face to be considered valid
PAD_WINDOW_SIZE = 5
PAD_MIN_LIVE = 4
PAD_MIN_SPOOF = 4


def run_pipeline_full():
    """
    Execute the full real-time processing pipeline.

    The function captures frames from the webcam, detects and validates
    faces, applies preprocessing, performs identity recognition using
    the trained LBPH model, and displays the results on screen.
    """

    # Create a VideoCapture object to access the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    # Load face detector
    detector = load_face_detector(MODEL_PATH)

    # Load anti-spoofing detector
    pad_detector = AntiSpoofingDetector()

    # Store last PAD predictions for temporal smoothing
    pad_history = deque(maxlen=PAD_WINDOW_SIZE)

    # Load the face recognizer and label map
    recognizer, label_map = load_recognizer()

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            print("Failed to grab frame")
            break

        # Get valid face from frame
        face_rect, gray_frame, message = get_valid_face(
            frame,
            detector,
            SCALE_FACTOR,
            MIN_NEIGHBORS,
            MIN_SIZE,
            THRESHOLD_AREA
        )

        if message == "No face detected":
            pad_history.clear()

            cv2.putText(
                frame,
                message,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        elif message == "Move closer to the camera":
            pad_history.clear()

            cv2.putText(
                frame,
                message,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        elif message == "Face detected":
            x, y, w, h = face_rect

            # PAD crop aligned with training flow: crop from grayscale frame
            pad_face_crop = gray_frame[y:y + h, x:x + w]

            cv2.imshow("PAD crop", pad_face_crop)
            # print("PAD crop shape:", pad_face_crop.shape)

            # Run PAD prediction on current frame
            pad_result = pad_detector.predict(pad_face_crop)

            current_pad_status = pad_result["status"]
            current_pad_label = pad_result["label"]
            current_pad_score = pad_result["score"]

            # Default final PAD decision
            final_pad_label = "unknown"

            # Update temporal history only with valid live/spoof predictions
            if current_pad_status == "ok" and current_pad_label in {"live", "spoof"}:
                pad_history.append(current_pad_label)

            # Temporal decision on last valid frames
            live_count = pad_history.count("live")
            spoof_count = pad_history.count("spoof")

            if live_count >= PAD_MIN_LIVE:
                final_pad_label = "live"
            elif spoof_count >= PAD_MIN_SPOOF:
                final_pad_label = "spoof"
            else:
                final_pad_label = "unknown"

            # Choose box color from final smoothed decision
            if final_pad_label == "live":
                box_color = (0, 255, 0)
            elif final_pad_label == "spoof":
                box_color = (0, 0, 255)
            else:
                box_color = (0, 255, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            # Display PAD result: show both current-frame result and final smoothed result
            cv2.putText(
                frame,
                f"PAD now: {current_pad_label.upper()} ({current_pad_score:.2f})",
                (x, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"PAD final: {final_pad_label.upper()}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_color,
                2,
                cv2.LINE_AA
            )

            # Only perform recognition if final PAD decision is live
            if final_pad_label == "live":
                face = preprocess_face_for_recognition(gray_frame, x, y, w, h)

                predicted_name, confidence = predict_identity(face, recognizer, label_map)

                cv2.imshow("Face", face)

                cv2.putText(
                    frame,
                    f"Identity: {predicted_name}",
                    (x, y + h + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    frame,
                    f"Confidence: {confidence:.2f}",
                    (x, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA
                )

            else:
                cv2.putText(
                    frame,
                    "Recognition blocked",
                    (x, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

        # Show webcam
        cv2.imshow("Webcam", frame)

        # Exit when 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
