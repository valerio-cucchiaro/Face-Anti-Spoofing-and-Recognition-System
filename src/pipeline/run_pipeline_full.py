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
    faces, applies preprocessing, performs identity recognition using
    the trained LBPH model, and displays the results on screen.
    """

    # create a VideoCapture object to access the webcam
    cap = cv2.VideoCapture(0)  
    detector = load_face_detector(MODEL_PATH)

    # load the face recognizer and label map
    recognizer, label_map = load_recognizer()

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

        # get valid face from frame
        face_rect, gray_frame, message = get_valid_face(frame, detector, SCALE_FACTOR, MIN_NEIGHBORS, MIN_SIZE, THRESHOLD_AREA)

        if message == "No face detected":
            cv2.putText(frame, message, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        elif message == "Move closer to the camera":
            cv2.putText(frame, message, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        elif message == "Face detected":
            x, y, w, h = face_rect
            
            # preprocess the detected face
            face = preprocess_face(gray_frame, x, y, w, h)

            # predict the identity of the detected face
            predicted_name, confidence = predict_identity(face, recognizer, label_map)

            # draw boundig box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show processed face
            cv2.imshow('Face', face)

            # display status message
            cv2.putText(frame, message, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # display predicted identity and confidence
            cv2.putText(frame, f"Identity: {predicted_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        
        # show webcam 
        cv2.imshow('Webcam', frame)

        # store the key pressed and exit when 'q' is pressed 
        key = cv2.waitKey(1) & 0xFF    
        if key == ord('q'):    
            break

    # release the webcam and close all windows
    cap.release()  
    cv2.destroyAllWindows() 