"""
Module: test_model

Description:
This module evaluates the trained LBPH face recognition model using a
separate test dataset. It loads the trained model and label mapping,
runs predictions on test images, and reports recognition accuracy.

The module also collects confidence scores for genuine and impostor
predictions to support empirical threshold selection for the runtime
recognition system.
"""

import cv2
import pickle
from pathlib import Path


# ----- CONSTANTS ----- #
TEST_DIR = Path("data/test")
MODEL_PATH = Path("models/lbph_model.xml")
LABEL_MAP_PATH = Path("models/label_map.pkl")


def test_model():
    """
    Evaluate the trained recognition model on a test dataset.

    The function runs predictions on labeled test images, measures the
    recognition accuracy, and analyzes confidence scores for genuine and
    impostor matches in order to suggest a suitable recognition threshold.
    """

    print("Loading trained model...")

    # load the trained LBPH model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))

    print("Loading label map...")

    # load the label map
    with open(LABEL_MAP_PATH, "rb") as f:
        label_map = pickle.load(f)

    print("Testing images...\n")

    # lists to store genuine and impostor scores for analysis and threshold selection
    genuine_scores = [] 
    impostor_scores = []

    # counters for correct predictions and total predictions
    correct = 0
    total = 0

    # iterate over each person folder
    for person_dir in TEST_DIR.iterdir():

        # skip if it's not a directory
        if not person_dir.is_dir():
            continue
        
        # get the name of the person from the folder name
        true_name = person_dir.name

        # iterate over test images
        for img_path in person_dir.iterdir():

            # skip non-image files
            if img_path.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
                continue

            # read the image in grayscale
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            # predict the label and confidence
            label, confidence = recognizer.predict(img)

            # get the predicted name from the label map
            predicted_name = label_map[label]

            print(f"{img_path.name}")
            print(f"True: {true_name}")
            print(f"Predicted: {predicted_name}")
            print(f"Confidence: {confidence:.2f}")
            print("------")

            # check if the prediction is correct and store the confidence score 
            if predicted_name == true_name:
                genuine_scores.append(confidence)
                correct += 1
            else:
                impostor_scores.append(confidence)

            total += 1

    print("\nRESULTS")
    print("Correct:", correct)
    print("Total:", total)
    print("Accuracy:", correct / total)

    print("\nGenuine scores:")
    print(genuine_scores)

    print("\nImpostor scores:")
    print(impostor_scores)

    # threshold suggestion
    if genuine_scores:
        threshold = max(genuine_scores) + 5
        print("\nSuggested threshold:", threshold)

if __name__ == "__main__":
    test_model()