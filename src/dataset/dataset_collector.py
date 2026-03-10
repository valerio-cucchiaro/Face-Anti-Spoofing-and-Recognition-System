import cv2
from pathlib import Path

from src.detection.face_detector import detect_faces, load_face_detector
from src.preprocessing.face_preprocessing import face_histogram_equalization, to_grayscale, crop_face, resize_face


# ----- CONSTANTS ----- #
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 6   
MIN_SIZE = (30, 30)
THRESHOLD_AREA = 3000   # minimum area of the detected face to be considered valid 
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "haarcascade_frontalface_default.xml"
PERSON = "valerio"
DATASET = Path(__file__).resolve().parent.parent.parent / "data" / "dataset" / PERSON
MAX_IMAGES = 30 

# create the dataset directory if it doesn't exist
DATASET.mkdir(parents=True, exist_ok=True)  

# get the list of existing images in the dataset
existing_images = list(DATASET.glob("*.png"))
image_count = len(existing_images)

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

    # initialize the face variable and a flag to check if a valid face is detected
    face = None
    valid_face = False

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
            cv2.putText(frame, f"Samples: {image_count} / {MAX_IMAGES}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

            # # preprocess the detected face
            face = crop_face(gray_frame, x, y, w, h)
            face = resize_face(face)
            face = face_histogram_equalization(face)
            valid_face = True

    # display the frame in a window 
    cv2.imshow('Webcam', frame) 
    if valid_face:
        cv2.imshow('Face', face)

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


