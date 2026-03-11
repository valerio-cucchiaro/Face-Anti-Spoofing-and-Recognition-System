# import OpenCV library for image processing
import cv2  

def to_grayscale(frame):
    # convert the frame to grayscale
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def crop_face(gray_frame, x, y, w, h):
    # Face Crop: extract the face region from the gray_frame
    return gray_frame[y:y+h, x:x+w]

def resize_face(face, size=(128, 128)):
    # resize the face region to a fixed size
    return cv2.resize(face, size)

def face_histogram_equalization(face):
    # apply histogram equalization to the face region
    return cv2.equalizeHist(face)

def preprocess_face(gray_frame, x, y, w, h):
    # preprocess the detected face: crop, resize, and apply histogram equalization
    face = crop_face(gray_frame, x, y, w, h)
    face = resize_face(face)
    face = face_histogram_equalization(face)
    return face