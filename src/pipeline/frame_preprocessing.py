from src.preprocessing.face_preprocessing import to_grayscale
from src.detection.face_detector import detect_faces

def get_valid_face(frame, detector, scale_factor, min_neighbors, min_size, threshold_area):
    # convert the frame to grayscale 
    gray_frame = to_grayscale(frame)

    # detect faces
    faces = detect_faces(detector, gray_frame, scale_factor, min_neighbors, min_size)

    # no faces detected
    if len(faces) == 0:
        return None, None, "No face detected"

    # select the largest face
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])  
    x, y, w, h = largest_face

    # if the detected face is too small display a message
    if w * h < threshold_area:
        return None, None, "Move closer to the camera"

    return (x, y, w, h), gray_frame, "Face detected"