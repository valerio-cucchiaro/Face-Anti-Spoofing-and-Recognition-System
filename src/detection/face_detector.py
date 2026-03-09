# import OpenCV library for image processing
import cv2  

def load_face_detector(model_path: str):
    # upload classifier
    detector = cv2.CascadeClassifier(model_path)

    if detector.empty():
        raise RuntimeError("Failed to load Haar Cascade classifier")
    
    return detector

def detect_faces(detector, gray_frame):
    # detect faces in the grayscale frame using the Haar Cascade classifier
    return detector.detectMultiScale(
        gray_frame, 
        scaleFactor=1.1, 
        minNeighbors=5,
        minSize=(30, 30)
    )