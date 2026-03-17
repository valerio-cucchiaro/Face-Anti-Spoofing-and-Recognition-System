import cv2
from src.preprocessing.face_preprocessing import preprocess_face_for_pad
from src.antispoofing.feature_extractor import extract_pad_features

def extract_pad_features_from_crop(face_crop):

    if face_crop is None or face_crop.size == 0:
        return None

    if len(face_crop.shape) == 3:
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray_face = face_crop

    face = preprocess_face_for_pad(
        gray_face,
        0,
        0,
        gray_face.shape[1],
        gray_face.shape[0]
    )

    if face is None:
        return None

    return extract_pad_features(face)