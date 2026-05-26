"""
Module: replayattack_frame_extractor

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
import argparse
from contextlib import contextmanager

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"


def silence_opencv_logs():
    """
    Silence OpenCV/FFmpeg logs when the installed cv2 build exposes a
    compatible logging API.
    """
    if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        return

    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(3)


silence_opencv_logs()

# ----- CONSTANTS ----- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPLAY_PATH = PROJECT_ROOT / "data" / "pad" / "replay_attack_raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "pad" / "replay_attack_pad"
FRAME_STEP = 10
VIDEO_EXTENSIONS = {".mov", ".avi", ".mp4"}


@contextmanager
def suppress_native_stderr():
    """
    Temporarily silence native FFmpeg/OpenCV stderr messages.

    Some Replay-Attack videos cannot be decoded by the local OpenCV
    build because its FFmpeg lacks zlib support. The extractor reports
    those files as skipped, so the low-level repeated header errors are
    hidden to keep the split log readable.
    """
    stderr_fd = 2
    saved_stderr_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


def get_output_split(video_path: Path, split_policy="legacy") -> str:
    """
    Map Replay-Attack raw split to PAD split.

    legacy:
      raw/train + raw/devel -> train
      raw/test              -> test

    validation:
      raw/train -> train
      raw/devel -> validation
      raw/test  -> test

    The current PAD training/evaluation stack supports the explicit
    ``validation`` layout. The ``legacy`` option is kept only for
    backward-compatible dataset regeneration.
    """
    if split_policy not in {"legacy", "validation"}:
        raise ValueError(
            f"Unsupported split policy: {split_policy}. Use 'legacy' or 'validation'."
        )

    parts = {part.lower() for part in video_path.parts}

    if "test" in parts:
        return "test"

    if "devel" in parts:
        return "validation" if split_policy == "validation" else "train"

    if "train" in parts:
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

    # Open the input video file. Native FFmpeg errors are suppressed here
    # and summarized through the explicit skipped-video count in main().
    with suppress_native_stderr():
        cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[SKIP] Cannot open video: {video_path}")
        return 0

    frame_idx = 0
    saved_idx = 0
    output_stem = build_output_stem(video_path)

    while True:
        with suppress_native_stderr():
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
    return saved_idx


def build_arg_parser():
    """
    Build the command-line interface for Replay-Attack extraction.
    """
    parser = argparse.ArgumentParser(
        description="Extract Replay-Attack frames into the PAD dataset layout."
    )
    parser.add_argument(
        "--split-policy",
        choices=["legacy", "validation"],
        default="legacy",
        help=(
            "Use 'legacy' to merge raw/devel into train, or 'validation' "
            "to write raw/devel into validation. The current PAD stack "
            "supports the explicit validation layout; 'legacy' is kept "
            "for backward-compatible regeneration."
        ),
    )
    return parser


def main(split_policy="legacy"):
    """
    Extract Replay-Attack frames and organize them in PAD format.
    """
    print("====== START REPLAY-ATTACK FRAME EXTRACTION ======")
    print("Replay-Attack split policy:", split_policy)
    if not REPLAY_PATH.exists():
        raise FileNotFoundError(f"Missing dataset folder: {REPLAY_PATH}")

    processed_videos = 0
    skipped_videos = 0
    saved_frames = 0

    for video_path in REPLAY_PATH.rglob("*"):
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        # Convert the original Replay-Attack directory layout into the
        # simpler PAD split/label structure used by the training scripts.
        split = get_output_split(video_path, split_policy=split_policy)
        label = get_output_label(video_path)

        output_dir = OUTPUT_PATH / split / label
        extracted_count = extract_frames(video_path, output_dir, step=FRAME_STEP)
        if extracted_count == 0:
            skipped_videos += 1
        else:
            processed_videos += 1
            saved_frames += extracted_count

    print("Replay-Attack extraction completed")
    print(f"Processed videos: {processed_videos}")
    print(f"Skipped videos: {skipped_videos}")
    print(f"Saved frames: {saved_frames}")
    print("====== END REPLAY-ATTACK FRAME EXTRACTION ======")


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    main(split_policy=cli_args.split_policy)
