"""
Module: run_pipeline_full

Description:
This module implements the complete runtime pipeline of the biometric
face recognition system. It orchestrates the main processing flow of
the application using the system webcam.

The pipeline performs the following stages:

webcam acquisition -> face detection -> face validation -> face
preprocessing -> antispoofing (PAD) -> face recognition ->
visualization of results

The module integrates MediaPipe face detection, preprocessing,
anti-spoofing, and recognition. It applies temporal smoothing to PAD
decisions and displays the final biometric output in real time.
"""

import cv2

from src.antispoofing.pad_crop_utils import extract_pad_face_crop
from src.antispoofing.pad_frame_quality import evaluate_pad_crop_quality
from src.antispoofing.pad_runtime_state import PadRuntimeState
from src.detection.face_detector import load_mediapipe_detector
from src.preprocessing.face_preprocessing import preprocess_face_for_recognition
from src.preprocessing.frame_preprocessing import get_valid_face
from src.recognition.face_recognizer import load_recognizer, predict_identity
from src.antispoofing.pad_runtime_detector import AntiSpoofingDetector


# ----- FACE DETECTION AND VALIDATION ----- #
THRESHOLD_AREA = 8000   # minimum area of the detected face to be considered valid

# ----- PAD TEMPORAL SMOOTHING ----- #
PAD_WINDOW_SIZE = 5

# ----- PAD HYSTERESIS THRESHOLDS ----- #
PAD_ENTER_LIVE_THRESHOLD = 0.70
PAD_EXIT_LIVE_THRESHOLD = 0.55
PAD_ENTER_SPOOF_THRESHOLD = 0.30
PAD_EXIT_SPOOF_THRESHOLD = 0.45

# ----- PAD FRAME QUALITY FILTERING ----- #
PAD_MIN_VARIANCE_LAPLACIAN = 80.0

# ----- PAD STATE RESET ----- #
PAD_RESET_AFTER_MISSES = 15


def run_pipeline_full():
    """
    Execute the full real-time processing pipeline.

    The function captures frames from the webcam, detects and validates
    faces, applies runtime PAD decision smoothing, performs identity
    recognition only when the final PAD decision is live, and displays
    the results on screen.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    detector = load_mediapipe_detector()
    pad_detector = AntiSpoofingDetector()

    pad_runtime_state = PadRuntimeState(
        window_size=PAD_WINDOW_SIZE,
        enter_live_threshold=PAD_ENTER_LIVE_THRESHOLD,
        exit_live_threshold=PAD_EXIT_LIVE_THRESHOLD,
        enter_spoof_threshold=PAD_ENTER_SPOOF_THRESHOLD,
        exit_spoof_threshold=PAD_EXIT_SPOOF_THRESHOLD,
        reset_after_misses=PAD_RESET_AFTER_MISSES,
    )

    recognizer, label_map = load_recognizer()

    prev_face_rect = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break
        
        face_rect, gray_frame, message = get_valid_face(
            frame,
            detector,
            THRESHOLD_AREA,
        )

        if message == "No face detected" or message == "No valid face detected":
            state_result = pad_runtime_state.update(face_missing=True)
            cv2.putText(frame, message, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        elif message == "Move closer to the camera":
            state_result = pad_runtime_state.update(face_missing=True)
            cv2.putText(frame, message, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        elif message == "Face detected":
            x, y, w, h = face_rect

            # --- BBOX STABILITY CHECK --- #
            bbox_stable = True

            if prev_face_rect is not None:
                prev_x, prev_y, prev_w, prev_h = prev_face_rect

                dx = abs(x - prev_x)
                dy = abs(y - prev_y)
                dw = abs(w - prev_w)
                dh = abs(h - prev_h)

                # Thresholds
                if dx > 30 or dy > 30 or dw > 35 or dh > 35:
                    bbox_stable = False

            # Update the reference box used for the next stability check.
            prev_face_rect = (x, y, w, h)

            # Build the PAD crop through the shared helper used by the
            # collector and offline PAD utilities.
            pad_face_crop = extract_pad_face_crop(gray_frame, x, y, w, h)

            # Evaluate the PAD crop quality before updating the temporal state.
            quality_result = evaluate_pad_crop_quality(pad_face_crop, PAD_MIN_VARIANCE_LAPLACIAN)
            laplacian_variance = quality_result["laplacian_variance"]
            update_pad_history = quality_result["is_frame_usable_for_history"] and bbox_stable

            # Run PAD inference only on valid crops, otherwise mark the frame as invalid.
            if quality_result["is_valid_crop"]:
                pad_result = pad_detector.predict(pad_face_crop)
            else:
                pad_result = {"status": "invalid_crop", "label": None, "score": None}

            state_result = pad_runtime_state.update(
                pad_result=pad_result,
                frame_usable=update_pad_history,
                face_missing=False,
            )
            current_pad_status = state_result["current_status"]
            current_pad_label = state_result["current_label"]
            current_pad_score = state_result["current_score"]
            mean_live_probability = state_result["mean_live_probability"]
            final_pad_label = state_result["final_label"]

            # Choose box color from final smoothed decision
            if final_pad_label == "live":
                box_color = (0, 255, 0)
            elif final_pad_label == "spoof":
                box_color = (0, 0, 255)
            else:
                box_color = (0, 255, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            # Build the current PAD text safely even when inference fails.
            if current_pad_status == "ok" and current_pad_label is not None and current_pad_score is not None:
                current_pad_text = f"PAD now: {current_pad_label.upper()} ({current_pad_score:.2f})"
            else:
                current_pad_text = "PAD now: UNKNOWN"

            # Show the current-frame PAD prediction and its score.
            cv2.putText(frame, current_pad_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            if mean_live_probability is not None:
                final_pad_text = f"PAD final: {final_pad_label.upper()} | avg_live={mean_live_probability:.2f}"
            else:
                final_pad_text = f"PAD final: {final_pad_label.upper()}"

            # Show the final smoothed PAD decision derived from recent frames.
            cv2.putText(frame, final_pad_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2, cv2.LINE_AA)

            # --- DEBUG --- #
            if not quality_result["is_valid_crop"]:
                if not quality_result["is_large_enough"]:
                    quality_text = f"PAD frame: SKIPPED | too small ({quality_result['crop_width']}x{quality_result['crop_height']})"
                elif not quality_result["has_plausible_aspect_ratio"]:
                    quality_text = f"PAD frame: SKIPPED | bad ratio ({quality_result['aspect_ratio']:.2f})"
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
                quality_text = f"PAD frame: ACCEPTED | {quality_result['crop_width']}x{quality_result['crop_height']} | blur={laplacian_variance:.1f}"
                quality_color = (0, 255, 0)

            cv2.putText(frame, quality_text, (x, y + h + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2, cv2.LINE_AA)

            # Only perform recognition if final PAD decision is live
            if final_pad_label == "live":
                face = preprocess_face_for_recognition(gray_frame, x, y, w, h)
                predicted_name, confidence = predict_identity(face, recognizer, label_map)
                cv2.imshow("Face", face)
                cv2.putText(frame, f"Identity: {predicted_name}", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (x, y + h + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Recognition blocked", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
