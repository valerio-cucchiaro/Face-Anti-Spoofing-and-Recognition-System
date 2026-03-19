"""
Module: extract_replayattack_frames

Description:
This module extracts image frames from the Replay-Attack video dataset
and stores them as standalone image files for later PAD experiments.

The script scans the input dataset directory recursively, identifies
video files, assigns each video to the `live` or `spoof` class based
on its path, and saves one frame every fixed number of frames into the
corresponding output folder.
"""

import cv2
from pathlib import Path
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

# ----- CONSTANTS ----- #
REPLAY_PATH = Path("data/replay_attack_raw")
OUTPUT_PATH = Path("data/replay_attack_pad")
FRAME_STEP = 10
VIDEO_EXTENSIONS = {".mov", ".avi", ".mp4"}


def get_output_split(video_path: Path) -> str:
    """
    Map Replay-Attack raw split to PAD split.

    raw/train + raw/devel -> train
    raw/test              -> test
    """
    parts = {part.lower() for part in video_path.parts}

    if "test" in parts:
        return "test"

    if "train" in parts or "devel" in parts:
        return "train"

    raise ValueError(f"Cannot infer split from path: {video_path}")


def get_output_label(video_path: Path) -> str:
    """
    Map Replay-Attack class folders to PAD labels.

    real   -> live
    attack -> spoof
    """
    parts = {part.lower() for part in video_path.parts}

    if "real" in parts:
        return "live"

    if "attack" in parts:
        return "spoof"

    raise ValueError(f"Cannot infer label from path: {video_path}")


def build_output_stem(video_path: Path) -> str:
    """
    Build a unique filename stem from the video path to avoid collisions.
    """
    relative_path = video_path.relative_to(REPLAY_PATH).with_suffix("")
    # Keep split and class information in the filename so frames from
    # different videos can never overwrite each other.
    safe_parts = [part.replace(" ", "_").replace("-", "_") for part in relative_path.parts]
    return "__".join(safe_parts)


def extract_frames(video_path: Path, output_dir: Path, step: int = 10):
    """
    Extract and save frames from a video at a fixed sampling interval.
    """

    # Create the output directory if it does not already exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open the input video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[SKIP] Cannot open video: {video_path}")
        return

    frame_idx = 0
    saved_idx = 0
    output_stem = build_output_stem(video_path)

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            break

        if frame_idx % step == 0:
            out_name = output_dir / f"{output_stem}__frame_{saved_idx:04d}.jpg"
            cv2.imwrite(str(out_name), frame)
            saved_idx += 1

        frame_idx += 1

    # Release the video handle when extraction is complete
    cap.release()


def main():
    """
    Extract Replay-Attack frames and organize them in PAD format.
    """
    if not REPLAY_PATH.exists():
        raise FileNotFoundError(f"Missing dataset folder: {REPLAY_PATH}")

    for video_path in REPLAY_PATH.rglob("*"):
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        # Convert the original Replay-Attack directory layout into the
        # simpler PAD split/label structure used by the training scripts.
        split = get_output_split(video_path)
        label = get_output_label(video_path)

        output_dir = OUTPUT_PATH / split / label
        extract_frames(video_path, output_dir, step=FRAME_STEP)


if __name__ == "__main__":
    main()
