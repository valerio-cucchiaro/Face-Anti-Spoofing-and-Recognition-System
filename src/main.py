# import OpenCV library for image processing
import cv2  

from detection.face_detector import detect_faces, load_face_detector
from preprocessing.face_preprocessing import to_grayscale, crop_face, resize_face

# ----- CONSTANTS ----- #
MODEL_PATH = "models/haarcascade_frontalface_default.xml"

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
        faces = detect_faces(detector, gray_frame)

        # draw rectangles around the detected faces in the original frame
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face = crop_face(gray_frame, x, y, w, h)
            face = resize_face(face)
            cv2.imshow('Face', face)

        # display the frame in a window 
        cv2.imshow('Webcam', frame) 
        # cv2.imshow('Gray Webcam', gray_frame)
    
        # store the key pressed and exit when 'q' is pressed 
        key = cv2.waitKey(1) & 0xFF    
        if key == ord('q'):    
            break

    # release the webcam and close all windows
    cap.release()  
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()