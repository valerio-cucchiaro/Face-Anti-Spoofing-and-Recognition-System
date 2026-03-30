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
    Collect face samples from the webcam to build the recognition
    custom dataset.

    The function detects the largest valid face in each frame through
    the shared MediaPipe-based validation helper, applies the standard
    recognition preprocessing pipeline, and saves the resulting face
    images to the selected gallery or probe directory when the user
    presses the capture key.
    """

    # create the dataset directory if it doesn't exist
    DATASET.mkdir(parents=True, exist_ok=True)  

    # get the list of existing images in the dataset
    existing_images = list(DATASET.glob("*.png"))
    image_count = len(existing_images)

    # create a VideoCapture object to access the webcam
    cap = cv2.VideoCapture(0)
    detector = load_mediapipe_detector()

    # check if the webcam is opened successfully: NO --> RunTimeError
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    while True:
        # read a frame from the webcam
        ret, frame = cap.read()  

        # check if the frame was read successfully: NO --> break the loop
        if not ret:  
            print("Failed to grab frame")
            break

        # flag to indicate if a valid face is detected
        valid_face = False

        # Select the largest valid face from the current webcam frame.
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
            valid_face = True
            
            # preprocess the detected face
            face = preprocess_face_for_recognition(gray_frame, x, y, w, h)

            # draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show processed face
            cv2.imshow('Face', face)

            # display message
            cv2.putText(frame, message, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Samples: {image_count} / {MAX_IMAGES}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        
        # show webcam 
        cv2.imshow('Webcam', frame)

        # store the key pressed and exit when 'q' is pressed 
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

    # release the webcam and close all windows
    cap.release()  
    cv2.destroyAllWindows() 


if __name__ == "__main__":
    collect_dataset()
