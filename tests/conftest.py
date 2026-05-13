import sys
from pathlib import Path

import numpy as np
import pytest

from src.preprocessing.face_preprocessing import configure_recognition_preprocessing


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def restore_recognition_preprocessing():
    """
    Keep tests isolated from recognition preprocessing global state.
    """
    configure_recognition_preprocessing(crop_padding_ratio=0.10)
    yield
    configure_recognition_preprocessing(crop_padding_ratio=0.10)


@pytest.fixture
def sharp_gray_face():
    """Synthetic grayscale face-like crop with enough local detail."""
    grid_y, grid_x = np.indices((128, 128))
    pattern = ((grid_x // 8 + grid_y // 8) % 2) * 255
    return pattern.astype(np.uint8)


@pytest.fixture
def blurry_gray_face():
    """Synthetic grayscale crop with almost no detail."""
    return np.full((128, 128), 127, dtype=np.uint8)


@pytest.fixture
def empty_crop():
    """Empty grayscale crop used for defensive checks."""
    return np.array([], dtype=np.uint8)
