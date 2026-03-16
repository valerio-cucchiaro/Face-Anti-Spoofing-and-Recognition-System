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
→ anti-spoofing analysis
→ face recognition
→ visualization of results

The module integrates the detection, preprocessing, and recognition
components and displays the predicted identity and confidence score
in real time.
"""

import cv2
from pathlib import Path

from src.detection.face_detector import load_face_detector
from src.preprocessing.face_preprocessing import preprocess_face
from src.pipeline.frame_preprocessing import get_valid_face
from src.recognition.face_recognizer import load_recognizer, predict_identity
from src.antispoofing.blink_detector import BlinkDetector, create_face_landmarker
from src.antispoofing.spoof_detector import AntiSpoofingDetector


# ----- CONSTANTS ----- #
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "haarcascade_frontalface_default.xml"
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 6
MIN_SIZE = (30, 30)
THRESHOLD_AREA = 3000   # minimum area of the detected face to be considered valid


def run_pipeline_full():
    """
    Execute the full real-time processing pipeline.

    The function captures frames from the webcam, detects and validates
    faces, applies preprocessing, performs anti-spoofing analysis,
    performs identity recognition using the trained LBPH model,
    and displays the results on screen.
    """

    # Create a VideoCapture object to access the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    detector = load_face_detector(MODEL_PATH)

    # Load the face recognizer and label map
    recognizer, label_map = load_recognizer()

    # Create the face landmarker for blink detection
    face_landmarker = create_face_landmarker()

    # Initialize the blink detector
    blink_detector = BlinkDetector()

    # Initialize the anti-spoofing detector
    anti_spoof_detector = AntiSpoofingDetector()

    # Manual frame counter used to generate a stable timestamp in milliseconds
    frame_index = 0

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            print("Failed to grab frame")
            break

        # Increment frame counter
        frame_index += 1

        # Approximate timestamp in milliseconds for MediaPipe VIDEO mode
        timestamp_ms = frame_index * 33

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
            blink_detector.reset_state()
            anti_spoof_detector.reset_state()

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
            blink_detector.reset_state()
            anti_spoof_detector.reset_state()

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

            # Preprocess the detected face
            face = preprocess_face(gray_frame, x, y, w, h)

            # Run robust blink detection
            blink_result = blink_detector.process_frame(frame, face_landmarker, timestamp_ms)

            # Safety check: if blink landmarks are lost even though Haar face exists,
            # reset state to avoid carrying stale temporal information
            if not blink_result.face_detected:
                blink_detector.reset_state()
                anti_spoof_detector.reset_state()

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                cv2.putText(
                    frame,
                    "Checking liveness...",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA
                )

            else:
                # Perform anti-spoofing analysis using texture and robust blink cues
                spoof_result = anti_spoof_detector.check_spoof(
                    face=face,
                    blink_detected=blink_result.blink_detected,
                    blink_count=blink_result.blink_count,
                    eye_closed=blink_result.eye_closed,
                    ear=blink_result.ear_smoothed
                )

                ear_text = "None" if blink_result.ear_smoothed is None else f"{blink_result.ear_smoothed:.2f}"

                cv2.putText(
                    frame,
                    f"EAR: {ear_text}",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    frame,
                    f"Blink: {spoof_result['blink_count']}",
                    (30, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    frame,
                    f"Texture: {spoof_result['texture_ok']}",
                    (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    frame,
                    f"Liveness: {spoof_result['liveness_confirmed']}",
                    (30, 155),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    frame,
                    f"Status: {spoof_result['status']}",
                    (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                if spoof_result["status"] == "spoof":
                    # Draw bounding box and display spoofing warning
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        "Spoofing detected!",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                    )

                elif spoof_result["status"] == "checking":
                    # Draw bounding box and display liveness checking message
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                    cv2.putText(
                        frame,
                        "Checking liveness...",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA
                    )

                elif spoof_result["status"] == "real":
                    # Predict the identity of the detected face
                    predicted_name, confidence = predict_identity(face, recognizer, label_map)

                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Show processed face
                    cv2.imshow("Face", face)

                    # Display status message
                    cv2.putText(
                        frame,
                        "Real face detected",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                    # Display predicted identity and confidence
                    cv2.putText(
                        frame,
                        f"Identity: {predicted_name}",
                        (x, y - 10),
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

        # Show webcam
        cv2.imshow("Webcam", frame)

        # Exit when 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()