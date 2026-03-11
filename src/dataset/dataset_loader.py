import cv2
from pathlib import Path
import numpy as np


# ----- CONSTANTS ----- #
DATASET = Path(__file__).resolve().parent.parent.parent / "data" / "dataset"

def load_dataset():

    # check if the dataset directory exists: NO --> FileNotFoundError
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET}")

    faces = []      # list to store the detected faces
    labels = []     # list to store the corresponding labels of the detected faces
    label_map = {}  # dictionary to map person names to integer labels

    # scan each person directory inside the dataset
    for person_dir in DATASET.iterdir():
        if person_dir.is_dir():
            # get the person name from the folder name 
            person_name = person_dir.name

            # assign a numeric label to the current person
            current_label = len(label_map)

            # save the mapping label -> person name
            label_map[current_label] = person_name     

            # load all images of the current person
            for image_path in person_dir.glob("*.png"):
                # read the image in grayscale 
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)  

                # check if the image was read successfully: NO --> print a warning and skip it
                if image is None:
                    print(f"Warning: Could not read image {image_path}")
                    continue

                # append the detected face and its label to the corresponding lists
                faces.append(image)
                labels.append(current_label)
    
    return faces, np.array(labels), label_map