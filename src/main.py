# import OpenCV library for image processing
import cv2
from pathlib import Path  

from detection.face_detector import detect_faces, load_face_detector
from preprocessing.face_preprocessing import face_histogram_equalization, to_grayscale, crop_face, resize_face

# ----- CONSTANTS ----- #
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "haarcascade_frontalface_default.xml"
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 6   
MIN_SIZE = (30, 30)
THRESHOLD_AREA = 3000   # minimum area of the detected face to be considered valid 

def main():
    # create a VideoCapture object to access the webcam
    cap = cv2.VideoCapture(0)  
    detector = load_face_detector(MODEL_PATH)

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

        gray_frame = to_grayscale(frame)
        faces = detect_faces(detector, gray_frame, SCALE_FACTOR, MIN_NEIGHBORS, MIN_SIZE)
        # if no faces are detected display a message
        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        else:
            # select the largest face 
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])  
            x, y, w, h = largest_face

            # if the detected face is too small display a message
            if w * h < THRESHOLD_AREA:
                cv2.putText(frame, "Move closer to the camera", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            else:
                # draw a rectangle around the detected face in the original frame and display a message
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Face detected", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # # preprocess the detected face
                face = crop_face(gray_frame, x, y, w, h)
                face = resize_face(face)
                face = face_histogram_equalization(face)
                cv2.imshow('Face', face)

        # display the frame in a window 
        cv2.imshow('Webcam', frame) 
    
        # store the key pressed and exit when 'q' is pressed 
        key = cv2.waitKey(1) & 0xFF    
        if key == ord('q'):    
            break

    # release the webcam and close all windows
    cap.release()  
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()