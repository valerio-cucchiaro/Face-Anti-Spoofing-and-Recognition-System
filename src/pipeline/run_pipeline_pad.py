"""
Module: run_pipeline_pad

Description:
This module runs the PAD-only runtime focused on face
anti-spoofing only.

The pipeline detects a valid face, extracts the PAD crop, applies the
runtime PAD classifier, and smooths the result with ``PadRuntimeState``.
No recognition step is executed here. This makes the runtime useful for
testing liveness behavior and PAD responsiveness in isolation.
"""

import cv2
import time

from src.antispoofing.pad_crop_utils import extract_pad_face_crop
from src.antispoofing.pad_frame_quality import (
    PAD_MIN_VARIANCE_LAPLACIAN,
    evaluate_bbox_stability,
    evaluate_pad_crop_quality,
)
from src.detection.face_landmarks import (
    extract_face_mesh_landmarks,
    get_reference_points,
    load_mediapipe_face_mesh,
)
from src.antispoofing.pad_runtime_detector import AntiSpoofingDetector
from src.antispoofing.pad_runtime_state import PadRuntimeState
from src.detection.face_detector import load_mediapipe_detector
from src.preprocessing.face_bbox_visualization import draw_visual_face_bbox
from src.preprocessing.face_pose import (
    compute_pose_metrics_from_landmarks,
    format_pose_metrics_for_log,
)
from src.preprocessing.frame_preprocessing import DEFAULT_MIN_FACE_AREA, get_valid_face


# ----- FACE DETECTION AND VALIDATION ----- #
THRESHOLD_AREA = DEFAULT_MIN_FACE_AREA

# ----- PAD TEMPORAL SMOOTHING ----- #
PAD_WINDOW_SIZE = 5
PAD_RUN_INTERVAL = 2

# ----- PAD HYSTERESIS THRESHOLDS ----- #
PAD_ENTER_LIVE_THRESHOLD = 0.65
PAD_EXIT_LIVE_THRESHOLD = 0.55
PAD_ENTER_SPOOF_THRESHOLD = 0.30
PAD_EXIT_SPOOF_THRESHOLD = 0.45
PAD_RESET_AFTER_MISSES = 15

# ----- PAD DISPLAY STATE MACHINE ----- #
PAD_STARTUP_DELAY_SECONDS = 2.0
DISPLAY_LIVE_ENTRY_CONFIRMATIONS = 2
DISPLAY_SPOOF_ENTRY_CONFIRMATIONS = 3
DISPLAY_LIVE_EXIT_INCOHERENT = 3
DISPLAY_SPOOF_EXIT_INCOHERENT = 2
DISPLAY_HOLD_SECONDS = 1.0
DISPLAY_PAD_MIN_ANALYZING_SECONDS = 1.0

# ----- DISTANCE BANDS (FACE AREA RATIO) ----- #
PAD_DISTANCE_FAR_MAX_AREA_RATIO = 0.08
PAD_DISTANCE_MID_MAX_AREA_RATIO = 0.16

# ----- CAMERA READ RECOVERY ----- #
CAMERA_READ_MAX_FAILURES = 10


def log_pad_event(message):
    """
    Print a concise PAD-runtime debug message.
    """

    print(f"[PAD] {message}")


def build_normal_pad_check_log(
    distance_band,
    pad_frame_usable,
    pad_state_result,
    score_text,
    live_score_text,
    avg_live_text,
    quality_result,
    pad_result,
):
    """
    Build a compact PAD-check log line for normal runtime use.
    """

    return (
        "PAD check: "
        f"band={distance_band} | "
        f"usable={pad_frame_usable} | "
        f"instant={pad_state_result['current_label']} | "
        f"score={score_text} | "
        f"prob_label={pad_result.get('probability_label')} | "
        f"live_score={live_score_text} | "
        f"final={pad_state_result['final_label']} | "
        f"avg_live={avg_live_text} | "
        f"crop_valid={quality_result['is_valid_crop']} | "
        f"sharp={quality_result['is_sharp_enough']}"
    )


def build_verbose_pad_check_log(
    active_feature_set,
    distance_band,
    pad_frame_usable,
    pad_state_result,
    score_text,
    live_score_text,
    avg_live_text,
    history_snapshot,
    pose_text,
    spatial_text,
    light_text,
    last_timing_text,
    pad_result,
    probability_mismatch_count,
    probability_check_count,
    probability_usable_mismatch_count,
):
    """
    Build the full PAD-check diagnostic log line for verbose mode.
    """

    return (
        "PAD check: "
        f"feature_set={active_feature_set} | "
        f"distance_band={distance_band} | "
        f"usable={pad_frame_usable} | "
        f"current={pad_state_result['current_label']} | "
        f"score={score_text} | "
        f"prob_label={pad_result.get('probability_label')} | "
        f"live_score={live_score_text} | "
        f"prob_mismatch={pad_result.get('prediction_probability_mismatch')} | "
        f"prob_mismatch_total={probability_mismatch_count}/{probability_check_count} | "
        f"prob_mismatch_usable={probability_usable_mismatch_count} | "
        f"final={pad_state_result['final_label']} | "
        f"avg_live={avg_live_text} | "
        f"history_len={len(history_snapshot)} | "
        f"history={history_snapshot} | "
        f"pose={pose_text} | "
        f"{spatial_text} | "
        f"{light_text} | "
        f"{last_timing_text}"
    )


def create_pad_runtime_state():
    """
    Build the PAD temporal state used by the anti-spoofing-only
    runtime.
    """

    return PadRuntimeState(
        window_size=PAD_WINDOW_SIZE,
        enter_live_threshold=PAD_ENTER_LIVE_THRESHOLD,
        exit_live_threshold=PAD_EXIT_LIVE_THRESHOLD,
        enter_spoof_threshold=PAD_ENTER_SPOOF_THRESHOLD,
        exit_spoof_threshold=PAD_EXIT_SPOOF_THRESHOLD,
        reset_after_misses=PAD_RESET_AFTER_MISSES,
    )


class PadDisplayState:
    """
    Slow visual state machine layered on top of the PAD runtime state.

    The PAD classifier may start with harsh early predictions. This
    state manager keeps the first seconds in an explicit analyzing phase
    and only promotes stable live/spoof decisions to the visible
    overlay.
    """

    SEARCHING = "SEARCHING"
    ANALYZING = "ANALYZING"
    SHOW_LIVE = "SHOW_LIVE"
    SHOW_SPOOF = "SHOW_SPOOF"

    def __init__(self):
        self.state = self.SEARCHING
        self.state_since = None
        self.incoherent_updates = 0
        self.live_entry_counter = 0
        self.spoof_entry_counter = 0

    def _transition(self, new_state, now, logger, reason):
        if self.state != new_state:
            self.state = new_state
            self.state_since = now
            self.incoherent_updates = 0
            self.live_entry_counter = 0
            self.spoof_entry_counter = 0
            logger(
                "Overlay transition: "
                f"state={new_state} | text={self.get_pad_text()} | reason={reason}"
            )

    def _hold_active(self, now):
        if self.state_since is None:
            return False
        return (now - self.state_since) < DISPLAY_HOLD_SECONDS

    def _analyzing_min_time_active(self, now):
        if self.state_since is None:
            return False
        return (now - self.state_since) < DISPLAY_PAD_MIN_ANALYZING_SECONDS

    def update(self, face_present, pad_state_result, pad_frame_usable, now, logger):
        """
        Update the visible PAD overlay state.
        """

        if not face_present:
            self._transition(self.SEARCHING, now, logger, reason="face_missing")
            return

        final_label = pad_state_result["final_label"]

        if self.state == self.SEARCHING:
            self._transition(self.ANALYZING, now, logger, reason="face_detected")

        if self.state == self.ANALYZING:
            if self._analyzing_min_time_active(now):
                self.live_entry_counter = 0
                self.spoof_entry_counter = 0
                return

            if not pad_frame_usable:
                self.live_entry_counter = 0
                self.spoof_entry_counter = 0
                return

            if final_label == "live":
                self.live_entry_counter += 1
                self.spoof_entry_counter = 0
                if self.live_entry_counter >= DISPLAY_LIVE_ENTRY_CONFIRMATIONS:
                    self._transition(
                        self.SHOW_LIVE,
                        now,
                        logger,
                        reason="live_confirmed",
                    )
                return

            if final_label == "spoof":
                self.spoof_entry_counter += 1
                self.live_entry_counter = 0
                if self.spoof_entry_counter >= DISPLAY_SPOOF_ENTRY_CONFIRMATIONS:
                    self._transition(
                        self.SHOW_SPOOF,
                        now,
                        logger,
                        reason="spoof_confirmed",
                    )
                return

            self.live_entry_counter = 0
            self.spoof_entry_counter = 0
            return

        if self.state == self.SHOW_LIVE:
            if final_label == "live" and pad_frame_usable:
                self.incoherent_updates = 0
                return

            if self._hold_active(now):
                return

            if not pad_frame_usable or final_label in ("unknown", "spoof"):
                self.incoherent_updates += 1
                if self.incoherent_updates >= DISPLAY_LIVE_EXIT_INCOHERENT:
                    self._transition(self.ANALYZING, now, logger, reason="live_decay")
            return

        if self.state == self.SHOW_SPOOF:
            if final_label == "spoof" and pad_frame_usable:
                self.incoherent_updates = 0
                return

            if self._hold_active(now):
                return

            if not pad_frame_usable or final_label in ("unknown", "live"):
                self.incoherent_updates += 1
                if self.incoherent_updates >= DISPLAY_SPOOF_EXIT_INCOHERENT:
                    self._transition(self.ANALYZING, now, logger, reason="spoof_decay")

    def get_pad_text(self):
        """
        Return the user-facing PAD line for the current overlay state.
        """

        if self.state == self.SEARCHING:
            return "PAD: Searching face..."
        if self.state == self.ANALYZING:
            return "PAD: Analyzing..."
        if self.state == self.SHOW_LIVE:
            return "PAD: LIVE"
        return "PAD: SPOOF"

    def get_box_color(self):
        """
        Return the overlay color for the current PAD state.
        """

        if self.state == self.SHOW_LIVE:
            return (0, 255, 0)
        if self.state == self.SHOW_SPOOF:
            return (0, 0, 255)
        return (0, 255, 255)


def draw_pad_overlay(frame, face_rect, box_color, pad_text, details_text):
    """
    Draw a compact overlay for the PAD-only runtime.
    """

    _x, _y, _w, _h = face_rect
    header_line_1_y = 35
    header_line_2_y = 65

    header_line_3_y = 95

    draw_visual_face_bbox(frame, face_rect, box_color, 2)
    cv2.putText(
        frame,
        "State: PAD ONLY",
        (20, header_line_1_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        pad_text,
        (20, header_line_2_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        box_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        details_text,
        (20, header_line_3_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

def format_timing_ms(value):
    """
    Format a timing value in milliseconds for overlay and console logs.
    """

    if value is None:
        return "None"

    return f"{value:.1f}ms"


def get_active_pad_feature_set_name(pad_config):
    """
    Build a human-readable description of the active PAD feature set.
    """

    debug_mode = pad_config["pad_debug_feature_mode"]
    base_name = "LBP + quality + GGD(LFT)_light (two-scale)"

    if debug_mode == "baseline":
        return base_name
    if debug_mode == "tgsd":
        return f"{base_name} + TGSD"

    return f"{debug_mode} + ggd_light"


def build_pad_timing_summary(pad_timings):
    """
    Format only the timing blocks that are actually active.
    """

    timing_parts = [
        f"pre={format_timing_ms(pad_timings.get('pad_preprocess'))}",
        f"orig={format_timing_ms(pad_timings.get('original_total'))}",
        f"down={format_timing_ms(pad_timings.get('downsampled_total'))}",
    ]

    if "tgsd" in pad_timings:
        timing_parts.append(f"tgsd={format_timing_ms(pad_timings.get('tgsd'))}")

    timing_parts.append(f"tot={format_timing_ms(pad_timings.get('total_predict'))}")
    return " | ".join(timing_parts)


def build_pad_spatial_summary(frame_shape, face_rect):
    """
    Summarize face position and size to help reason about distance and pose.
    """

    frame_h, frame_w = frame_shape[:2]
    x, y, w, h = face_rect
    frame_area = float(frame_w * frame_h) if frame_w and frame_h else 0.0
    face_area = float(w * h)
    face_area_ratio = face_area / frame_area if frame_area > 0 else 0.0

    face_center_x = x + (w / 2.0)
    face_center_y = y + (h / 2.0)
    frame_center_x = frame_w / 2.0 if frame_w else 0.0
    frame_center_y = frame_h / 2.0 if frame_h else 0.0

    offset_x = (
        (face_center_x - frame_center_x) / frame_center_x if frame_center_x else 0.0
    )
    offset_y = (
        (face_center_y - frame_center_y) / frame_center_y if frame_center_y else 0.0
    )

    return (
        f"bbox_area={face_area_ratio:.3f} | "
        f"center_offset_x={offset_x:.3f} | "
        f"center_offset_y={offset_y:.3f}"
    )


def build_pad_light_summary(gray_crop):
    """
    Summarize crop brightness, contrast and simple asymmetry proxies.
    """

    if gray_crop is None or gray_crop.size == 0:
        return "crop_light=None"

    crop_mean = float(gray_crop.mean())
    crop_std = float(gray_crop.std())
    crop_lap = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
    dark_ratio = float((gray_crop < 80).mean())
    bright_ratio = float((gray_crop > 200).mean())

    mid_x = gray_crop.shape[1] // 2
    mid_y = gray_crop.shape[0] // 2
    left_mean = float(gray_crop[:, :mid_x].mean()) if mid_x > 0 else crop_mean
    right_mean = float(gray_crop[:, mid_x:].mean()) if mid_x > 0 else crop_mean
    top_mean = float(gray_crop[:mid_y, :].mean()) if mid_y > 0 else crop_mean
    bottom_mean = float(gray_crop[mid_y:, :].mean()) if mid_y > 0 else crop_mean

    lr_delta = right_mean - left_mean
    tb_delta = bottom_mean - top_mean

    return (
        f"mean={crop_mean:.1f} | "
        f"std={crop_std:.1f} | "
        f"lap={crop_lap:.1f} | "
        f"dark={dark_ratio:.2f} | "
        f"bright={bright_ratio:.2f} | "
        f"lr={lr_delta:.1f} | "
        f"tb={tb_delta:.1f}"
    )


def classify_pad_distance_band(face_area_ratio):
    """
    Classify a face crop into a coarse distance band.
    """

    if face_area_ratio >= PAD_DISTANCE_MID_MAX_AREA_RATIO:
        return "near"
    if face_area_ratio >= PAD_DISTANCE_FAR_MAX_AREA_RATIO:
        return "mid"
    return "far"


def run_pipeline_pad(
    pad_artifact_profile="runtime",
    pad_artifact_tag=None,
    log_level="normal",
):
    """
    Execute the anti-spoofing-only runtime pipeline.

    This entry point is meant for runtime experiments where the PAD
    module must be observed without the recognition stage.
    """

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    detector = load_mediapipe_detector()
    face_mesh = load_mediapipe_face_mesh()
    try:
        pad_detector = AntiSpoofingDetector(
            artifact_profile=pad_artifact_profile,
            artifact_tag=pad_artifact_tag,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize PAD runtime detector. "
            "Check PAD model, scaler, classifier, and config artifacts."
        ) from exc
    pad_runtime_state = create_pad_runtime_state()
    display_state = PadDisplayState()
    active_feature_set = get_active_pad_feature_set_name(pad_detector.pad_config)

    frame_counter = 0
    camera_read_failures = 0
    previous_face_rect = None
    last_timing_text = "features: waiting"
    last_pad_state_result = {
        "final_label": "unknown",
        "mean_live_probability": None,
        "current_status": None,
        "current_label": None,
        "current_score": None,
        "missing_face_counter": 0,
    }
    distance_band_counts = {
        "near": {"live": 0, "spoof": 0, "unknown": 0},
        "mid": {"live": 0, "spoof": 0, "unknown": 0},
        "far": {"live": 0, "spoof": 0, "unknown": 0},
    }
    probability_check_count = 0
    probability_mismatch_count = 0
    probability_usable_mismatch_count = 0
    last_logged_distance_band = None
    last_logged_final_label = None
    startup_started_at = time.monotonic()
    # Diagnostic startup logs for quick runtime checks. They report the
    # active feature family, exported feature dimension, and runtime PAD
    # smoothing thresholds loaded for the current session.
    log_pad_event(f"PAD run interval: {PAD_RUN_INTERVAL}")
    log_pad_event(
        "PAD runtime artifact selection: "
        f"profile={pad_artifact_profile} | tag={pad_artifact_tag or 'canonical'}"
    )
    log_pad_event(
        "PAD runtime feature set: "
        f"{active_feature_set} | dim={pad_detector.pad_config['feature_dim']}"
    )
    log_pad_event(
        "PAD runtime modes: "
        f"feature_mode={pad_detector.pad_config['pad_debug_feature_mode']} | "
        f"lft={pad_detector.pad_config['pad_lft_mode']} | "
        f"tgsd_tau={pad_detector.pad_config.get('tgsd_tau')} | "
        f"tgsd_top_percent={pad_detector.pad_config.get('tgsd_top_percent')} | "
        f"tgsd_grad={pad_detector.pad_config.get('tgsd_gradient_operator')}"
    )
    log_pad_event(
        "PAD runtime thresholds: "
        f"window={pad_runtime_state.window_size} | "
        f"enter_live={pad_runtime_state.enter_live_threshold:.2f} | "
        f"exit_live={pad_runtime_state.exit_live_threshold:.2f} | "
        f"enter_spoof={pad_runtime_state.enter_spoof_threshold:.2f} | "
        f"exit_spoof={pad_runtime_state.exit_spoof_threshold:.2f} | "
        f"reset_after_misses={pad_runtime_state.reset_after_misses}"
    )
    log_pad_event(f"PAD runtime log level: {log_level}")

    while True:
        ret, frame = cap.read()

        if not ret:
            camera_read_failures += 1
            log_pad_event(
                "Webcam frame read failed: "
                f"{camera_read_failures}/{CAMERA_READ_MAX_FAILURES}"
            )
            if camera_read_failures >= CAMERA_READ_MAX_FAILURES:
                raise RuntimeError(
                    "Webcam read failed repeatedly; stopping PAD runtime session"
                )
            continue

        camera_read_failures = 0

        frame_counter += 1
        now = time.monotonic()

        if (now - startup_started_at) < PAD_STARTUP_DELAY_SECONDS:
            cv2.imshow("PAD Runtime", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            continue

        face_rect, gray_frame, message = get_valid_face(
            frame,
            detector,
            THRESHOLD_AREA,
        )

        if message != "Face detected":
            pad_runtime_state.update(face_missing=True)
            display_state.update(
                face_present=False,
                pad_state_result={
                    "final_label": "unknown",
                    "mean_live_probability": None,
                },
                pad_frame_usable=False,
                now=now,
                logger=log_pad_event,
            )
            previous_face_rect = None
            display_message = (
                "Searching face" if message == "No face detected" else message
            )
            cv2.putText(
                frame,
                display_message,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("PAD Runtime", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            continue

        x, y, w, h = face_rect
        current_face_rect = (x, y, w, h)
        frame_area = float(frame.shape[0] * frame.shape[1])
        face_area_ratio = (float(w * h) / frame_area) if frame_area > 0 else 0.0
        distance_band = classify_pad_distance_band(face_area_ratio)
        landmarks_result = extract_face_mesh_landmarks(face_mesh, frame)
        pose_metrics = None
        if landmarks_result is not None:
            reference_points = get_reference_points(landmarks_result.landmarks)
            pose_metrics = compute_pose_metrics_from_landmarks(reference_points)
        stability_result = evaluate_bbox_stability(
            current_face_rect,
            previous_face_rect,
        )
        previous_face_rect = current_face_rect

        pad_face_crop = extract_pad_face_crop(gray_frame, x, y, w, h)
        quality_result = evaluate_pad_crop_quality(
            pad_face_crop,
            PAD_MIN_VARIANCE_LAPLACIAN,
        )
        spatial_text = build_pad_spatial_summary(frame.shape, face_rect)
        light_text = build_pad_light_summary(pad_face_crop)
        pad_frame_usable = (
            quality_result["is_crop_usable_for_history"]
            and stability_result["is_stable"]
        )

        should_run_pad = (frame_counter % PAD_RUN_INTERVAL == 0)
        if should_run_pad:
            if quality_result["is_valid_crop"]:
                pad_result = pad_detector.predict_safe(pad_face_crop)
                if pad_result["status"] == "predict_failed":
                    log_pad_event(
                        "PAD prediction failed: "
                        f"error={pad_result.get('error')}"
                    )
                elif pad_result["status"] == "ok":
                    probability_check_count += 1
                    if pad_result.get("prediction_probability_mismatch"):
                        probability_mismatch_count += 1
                        if pad_frame_usable:
                            probability_usable_mismatch_count += 1
            else:
                pad_result = {
                    "status": "invalid_crop",
                    "label": None,
                    "score": None,
                }

            pad_state_result = pad_runtime_state.update(
                pad_result=pad_result,
                frame_usable=pad_frame_usable,
                face_missing=False,
            )
            last_pad_state_result = pad_state_result
            score_text = (
                f"{pad_state_result['current_score']:.2f}"
                if pad_state_result["current_score"] is not None
                else "None"
            )
            avg_live_text = (
                f"{pad_state_result['mean_live_probability']:.2f}"
                if pad_state_result["mean_live_probability"] is not None
                else "None"
            )
            history_snapshot = pad_runtime_state.get_history_snapshot()
            pad_timings = pad_result.get("timings_ms", {})
            last_timing_text = build_pad_timing_summary(pad_timings)
            final_label = pad_state_result["final_label"]
            if final_label not in distance_band_counts[distance_band]:
                final_label = "unknown"
            distance_band_counts[distance_band][final_label] += 1
            should_log_details = (
                distance_band != last_logged_distance_band
                or (
                    final_label in ("live", "spoof")
                    and final_label != last_logged_final_label
                )
            )
            if should_log_details:
                pose_text = format_pose_metrics_for_log(pose_metrics)
                if pose_text == "pose=None":
                    pose_text = "None"
                live_score = pad_result.get("live_score")
                live_score_text = (
                    f"{live_score:.2f}" if live_score is not None else "None"
                )
                if log_level == "verbose":
                    log_message = build_verbose_pad_check_log(
                        active_feature_set=active_feature_set,
                        distance_band=distance_band,
                        pad_frame_usable=pad_frame_usable,
                        pad_state_result=pad_state_result,
                        score_text=score_text,
                        live_score_text=live_score_text,
                        avg_live_text=avg_live_text,
                        history_snapshot=history_snapshot,
                        pose_text=pose_text,
                        spatial_text=spatial_text,
                        light_text=light_text,
                        last_timing_text=last_timing_text,
                        pad_result=pad_result,
                        probability_mismatch_count=probability_mismatch_count,
                        probability_check_count=probability_check_count,
                        probability_usable_mismatch_count=probability_usable_mismatch_count,
                    )
                else:
                    log_message = build_normal_pad_check_log(
                        distance_band=distance_band,
                        pad_frame_usable=pad_frame_usable,
                        pad_state_result=pad_state_result,
                        score_text=score_text,
                        live_score_text=live_score_text,
                        avg_live_text=avg_live_text,
                        quality_result=quality_result,
                        pad_result=pad_result,
                    )
                log_pad_event(log_message)
                last_logged_distance_band = distance_band
                if final_label in ("live", "spoof"):
                    last_logged_final_label = final_label
        else:
            pad_state_result = last_pad_state_result

        mean_live_probability = pad_state_result["mean_live_probability"]
        display_state.update(
            face_present=True,
            pad_state_result=pad_state_result,
            pad_frame_usable=pad_frame_usable,
            now=now,
            logger=log_pad_event,
        )

        pad_text = display_state.get_pad_text()
        if display_state.state in (PadDisplayState.SHOW_LIVE, PadDisplayState.SHOW_SPOOF):
            if mean_live_probability is not None:
                pad_text = f"{pad_text} | avg_live={mean_live_probability:.2f}"

        details_text = (
            f"crop_valid={quality_result['is_valid_crop']} | "
            f"sharp={quality_result['is_sharp_enough']}"
        )
        draw_pad_overlay(
            frame,
            face_rect,
            display_state.get_box_color(),
            pad_text,
            details_text,
        )

        cv2.imshow("PAD Runtime", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Final diagnostic summary reporting how often the PAD runtime ended
    # in live/spoof/unknown across near, mid, and far face-size bands.
    log_pad_event("Distance band summary (PAD checks):")
    for band in ("near", "mid", "far"):
        band_counts = distance_band_counts[band]
        total = band_counts["live"] + band_counts["spoof"] + band_counts["unknown"]
        log_pad_event(
            f"{band}: total={total} | "
            f"live={band_counts['live']} | "
            f"spoof={band_counts['spoof']} | "
            f"unknown={band_counts['unknown']}"
        )
    if hasattr(face_mesh, "close"):
        face_mesh.close()
