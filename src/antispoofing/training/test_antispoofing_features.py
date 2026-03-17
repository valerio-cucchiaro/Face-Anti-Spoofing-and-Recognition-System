import cv2
import random
from pathlib import Path
from src.antispoofing.texture_features import extract_lbp_features
from src.antispoofing.quality_features import extract_quality_features
from src.antispoofing.feature_extractor import extract_pad_features


def test_texture_features():
    DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "casia_pad" / "train" / "live"

    #select a random image from the dataset path
    image_files = list(DATASET_PATH.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"No images found in dataset path: {DATASET_PATH}")
    image_path = random.choice(image_files)

    # load image with cv2
    image = cv2.imread(str(image_path))

    features = extract_lbp_features(image)
    # print featuers vector shape
    print("LBP feature vector shape:", features.shape)
    #print features values
    print("LBP feature vector values:", features)
    # print the sum of the histogram values to check if it is normalized
    print("Histogram sum:", features.sum())


def test_quality_features():

    DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "casia_pad" / "train" / "live"

    #select a random image from the dataset path
    image_files = list(DATASET_PATH.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"No images found in dataset path: {DATASET_PATH}")
    image_path = random.choice(image_files)

    # load image with cv2
    image = cv2.imread(str(image_path))

    features = extract_quality_features(image)
    # print featuers vector shape
    print("Quality feature vector shape:", features.shape)
    #print features values
    print("Quality feature vector values:", features)


def test_combined_features():
    DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "casia_pad" / "train" / "live"

    #select a random image from the dataset path
    image_files = list(DATASET_PATH.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"No images found in dataset path: {DATASET_PATH}")
    image_path = random.choice(image_files)

    # load image with cv2
    image = cv2.imread(str(image_path))

    features = extract_pad_features(image)
    # print featuers vector shape
    print("Combined feature vector shape:", features.shape)
    #print features values
    print("Combined feature vector values:", features)

    







if __name__ == "__main__":
    print("Testing texture features...")
    test_texture_features()
    print("\nTesting quality features...")
    test_quality_features()
    print("\nTesting combined features...")
    test_combined_features()