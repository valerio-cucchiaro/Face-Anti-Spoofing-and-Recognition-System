"""
Module: blink_detector

Description:
This module implements robust blink-based liveness detection using the
Eye Aspect Ratio (EAR) extracted from MediaPipe facial landmarks.

The module does not rely on a single-frame EAR decision only.
Instead, it applies:

- EAR smoothing with a small temporal buffer
- hysteresis with separate thresholds for eye closing and reopening
- minimum closed-eye duration validation

This makes blink detection more stable and reduces false positives
caused by head motion, landmark noise, or small pose variations.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ----- CONSTANTS ----- #
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "face_landmarker.task"

# Hysteresis thresholds:
# - below CLOSED_THRESHOLD => eye becomes closed
# - above OPEN_THRESHOLD   => eye becomes open again
EAR_CLOSED_THRESHOLD = 0.20
EAR_OPEN_THRESHOLD = 0.23

# Temporal filtering
EAR_SMOOTHING_WINDOW = 5
MIN_CLOSED_EYE_FRAMES = 2


def create_face_landmarker():
    """
    Create and return a MediaPipe Face Landmarker detector.
    """

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )

    face_landmarker = vision.FaceLandmarker.create_from_options(options)
    return face_landmarker


def compute_ear_for_eye(eye):
    """
    Compute the Eye Aspect Ratio (EAR) for a single eye.
    """

    p1, p2, p3, p4, p5, p6 = eye

    vertical1 = np.linalg.norm(p2 - p6)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0:
        return 0.0

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return float(ear)


def compute_ear(landmarks):
    """
    Compute the average Eye Aspect Ratio (EAR) for both eyes.
    """

    left_eye = np.array([landmarks[i] for i in LEFT_EYE_INDICES])
    right_eye = np.array([landmarks[i] for i in RIGHT_EYE_INDICES])

    ear_left = compute_ear_for_eye(left_eye)
    ear_right = compute_ear_for_eye(right_eye)

    ear = (ear_left + ear_right) / 2.0
    return float(ear)


@dataclass
class BlinkDetectorResult:
    """
    Container for blink detection results on a single frame.
    """

    face_detected: bool
    ear_raw: float | None
    ear_smoothed: float | None
    eye_closed: bool
    blink_detected: bool
    blink_count: int


class BlinkDetector:
    """
    Stateful robust blink detector.

    This class keeps temporal state across frames in order to detect
    valid blinks more reliably than a simple frame-by-frame threshold.
    """

    def __init__(
        self,
        closed_threshold=EAR_CLOSED_THRESHOLD,
        open_threshold=EAR_OPEN_THRESHOLD,
        smoothing_window=EAR_SMOOTHING_WINDOW,
        min_closed_frames=MIN_CLOSED_EYE_FRAMES
    ):
        """
        Initialize the blink detector.

        The detector uses:
        - a temporal EAR buffer for smoothing
        - hysteresis thresholds for stable eye state transitions
        - a minimum closed-eye duration to validate blinks
        """

        if closed_threshold >= open_threshold:
            raise ValueError(
                "closed_threshold must be smaller than open_threshold "
                "to implement hysteresis correctly."
            )

        if smoothing_window < 1:
            raise ValueError("smoothing_window must be at least 1.")

        if min_closed_frames < 1:
            raise ValueError("min_closed_frames must be at least 1.")

        self.closed_threshold = closed_threshold
        self.open_threshold = open_threshold
        self.min_closed_frames = min_closed_frames

        self.ear_buffer = deque(maxlen=smoothing_window)

        # Current eye state inferred through hysteresis
        self.eye_closed = False

        # Number of consecutive frames in which the eyes stayed closed
        self.closed_eye_frames = 0

        # Total valid blink count for the current tracked subject/state
        self.blink_count = 0

    def reset_state(self):
        """
        Reset the internal temporal state of the blink detector.

        This should be used when the face disappears or tracking is lost,
        so that stale temporal information is not reused.

        Note:
        blink_count is also reset, because it refers to the current
        tracked face sequence rather than to a global session counter.
        """

        self.ear_buffer.clear()
        self.eye_closed = False
        self.closed_eye_frames = 0
        self.blink_count = 0

    def _smooth_ear(self, ear):
        """
        Update the EAR buffer and return the smoothed EAR value.
        """

        self.ear_buffer.append(ear)
        smoothed_ear = float(np.mean(self.ear_buffer))
        return smoothed_ear

    def _update_eye_state(self, ear_smoothed):
        """
        Update eye open/closed state using hysteresis thresholds.

        Returns:
            bool: current eye_closed state after update
        """

        # If the eye is currently open, use the lower threshold to close it
        if not self.eye_closed and ear_smoothed < self.closed_threshold:
            self.eye_closed = True

        # If the eye is currently closed, use the higher threshold to reopen it
        elif self.eye_closed and ear_smoothed > self.open_threshold:
            self.eye_closed = False

        return self.eye_closed

    def _update_blink_state(self, eye_closed):
        """
        Update blink temporal logic.

        A valid blink is detected only when:
        - the eye remained closed for at least min_closed_frames
        - then the eye reopens

        Returns:
            bool: True if a valid blink is detected in the current frame
        """

        blink_detected = False

        if eye_closed:
            self.closed_eye_frames += 1
        else:
            if self.closed_eye_frames >= self.min_closed_frames:
                blink_detected = True
                self.blink_count += 1

            self.closed_eye_frames = 0

        return blink_detected

    def process_frame(self, frame, face_landmarker, timestamp_ms):
        """
        Process one frame and return robust blink detection information.

        Steps:
        1. Extract face landmarks with MediaPipe
        2. Compute raw EAR
        3. Apply temporal smoothing
        4. Update eye open/closed state using hysteresis
        5. Validate blink using minimum closed duration

        If no face landmarks are detected, the function returns
        face_detected=False but does not reset the internal state.
        State reset must be explicitly handled by the caller when needed.
        """

        # Convert the frame to RGB format for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a MediaPipe Image object from the RGB frame
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Run the face landmarker
        results = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        if not results.face_landmarks:
            return BlinkDetectorResult(
                face_detected=False,
                ear_raw=None,
                ear_smoothed=None,
                eye_closed=False,
                blink_detected=False,
                blink_count=self.blink_count
            )

        face_landmarks = results.face_landmarks[0]
        height, width, _ = frame.shape

        # Convert normalized landmark coordinates to pixel coordinates
        landmarks = []
        for landmark in face_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmarks.append([x, y])

        landmarks = np.array(landmarks)

        # Compute EAR
        ear_raw = compute_ear(landmarks)
        ear_smoothed = self._smooth_ear(ear_raw)

        # Update robust eye state and blink state
        eye_closed = self._update_eye_state(ear_smoothed)
        blink_detected = self._update_blink_state(eye_closed)

        return BlinkDetectorResult(
            face_detected=True,
            ear_raw=ear_raw,
            ear_smoothed=ear_smoothed,
            eye_closed=eye_closed,
            blink_detected=blink_detected,
            blink_count=self.blink_count
        )