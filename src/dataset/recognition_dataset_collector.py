"""
Module: recognition_dataset_collector

Description:
This module is responsible for collecting face images used to build
the gallery or probe partitions of the custom recognition dataset.

The script captures frames from the webcam, detects and validates a
face with MediaPipe, applies the same preprocessing pipeline used in
the runtime system, and stores the processed face images in the
dataset directory.

Images are saved sequentially and can be captured interactively
through keyboard input.

This script is a manual data-collection utility, not part of the
automated recognition training or evaluation flow.
"""

import cv2
from pathlib import Path

from src.detection.face_detector import load_mediapipe_detector
from src.preprocessing.face_preprocessing import preprocess_face_for_recognition
from src.preprocessing.frame_preprocessing import get_valid_face


# ----- CONSTANTS ----- #
THRESHOLD_AREA = 8000
PERSON = "francesca"
MODE = "probe"      # gallery or probe
MAX_IMAGES_GALLERY = 30
MAX_IMAGES_PROBE = 10

if MODE not in {"gallery", "probe"}:
    raise ValueError("MODE must be either 'gallery' or 'probe'")

DATASET = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "recognition"
    / "custom"
    / MODE
    / PERSON
)
MAX_IMAGES = MAX_IMAGES_GALLERY if MODE == "gallery" else MAX_IMAGES_PROBE


def collect_dataset():
    """
    Collect webcam face samples for the custom recognition dataset.

    The function detects the largest valid face in each frame through
    the shared MediaPipe-based validation helper, applies the standard
    recognition preprocessing pipeline, and saves the resulting face
    images to the selected gallery or probe directory when the user
    presses the capture key.
    """

    DATASET.mkdir(parents=True, exist_ok=True)  

    existing_images = list(DATASET.glob("*.png"))
    image_count = len(existing_images)

    cap = cv2.VideoCapture(0)
    detector = load_mediapipe_detector()

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    while True:
        ret, frame = cap.read()  

        if not ret:  
            print("Failed to grab frame")
            break

        valid_face = False

        face_rect, gray_frame, message = get_valid_face(
            frame,
            detector,
            THRESHOLD_AREA,
        )

        if message == "No face detected":
            cv2.putText(frame, message, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        elif message == "Move closer to the camera":
            cv2.putText(frame, message, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        elif message == "Face detected":
            x, y, w, h = face_rect
            
            face = preprocess_face_for_recognition(gray_frame, x, y, w, h)
            if face is None:
                cv2.putText(frame, "Invalid face crop", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2, cv2.LINE_AA)
            else:
                valid_face = True

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.imshow('Face', face)

                cv2.putText(frame, message, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Samples: {image_count} / {MAX_IMAGES}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Webcam', frame)

        key = cv2.waitKey(1) & 0xFF   
        if key == ord('s') and valid_face:
            if image_count < MAX_IMAGES:
                image_count += 1
                cv2.imwrite(str(DATASET / f"{image_count:03d}.png"), face)
                
            else:
                print("Maximum number of images reached")
                break
        if key == ord('q'):    
            break

    cap.release()  
    cv2.destroyAllWindows() 


if __name__ == "__main__":
    collect_dataset()
