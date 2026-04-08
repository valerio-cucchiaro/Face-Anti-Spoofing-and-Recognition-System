"""
Module: run_pipeline_bs

Description:
This module runs the biometric-systems runtime focused on face
recognition only.

The pipeline keeps the webcam loop intentionally simple:
- detect a valid face
- preprocess it for LBPH recognition
- apply a short temporal smoothing on recognition outputs
- display the current and stable identity on screen

No PAD gate is used here. This runtime is useful when recognition must
be tested in isolation from anti-spoofing.
"""

import cv2
import time
import numpy as np

from src.antispoofing.pad_crop_utils import extract_pad_face_crop
from src.antispoofing.pad_frame_quality import (
    PAD_MIN_VARIANCE_LAPLACIAN,
    evaluate_bbox_stability,
    evaluate_pad_crop_quality,
)
from src.detection.face_detector import load_mediapipe_detector
from src.preprocessing.face_preprocessing import preprocess_face_for_recognition
from src.preprocessing.frame_preprocessing import get_valid_face
from src.recognition.face_recognizer import load_recognizer, predict_identity
from src.recognition.recognition_runtime_state import RecognitionRuntimeState


# ----- FACE DETECTION AND VALIDATION ----- #
THRESHOLD_AREA = 8000

# ----- RECOGNITION RUNTIME SETTINGS ----- #
RECOGNITION_WINDOW_SIZE = 5
RECOGNITION_MIN_CONFIRMATIONS = 3
RECOGNITION_RESET_AFTER_MISSES = 10
RECOGNITION_RUN_INTERVAL = 3
RUNTIME_RECOGNITION_THRESHOLD = 43.0
DISPLAY_KNOWN_ENTRY_VOTES = 3
DISPLAY_UNKNOWN_ENTRY_VOTES = 3
DISPLAY_KNOWN_EXIT_INCOHERENT = 4
DISPLAY_UNKNOWN_EXIT_INCOHERENT = 3
DISPLAY_HOLD_SECONDS = 1.0
DISPLAY_KNOWN_CONFIRM_TIMEOUT_SECONDS = 1.0
RECOGNITION_SUMMARY_INTERVAL_SECONDS = 15.0

# ----- LIGHT DIAGNOSTICS ----- #
LIGHT_DARK_PIXEL_THRESHOLD = 40.0
LIGHT_BRIGHT_PIXEL_THRESHOLD = 215.0
LIGHT_GOOD_MEAN_MIN = 70.0
LIGHT_GOOD_MEAN_MAX = 180.0
LIGHT_MID_MEAN_MIN = 45.0
LIGHT_MID_MEAN_MAX = 220.0
LIGHT_GOOD_DARK_RATIO_MAX = 0.25
LIGHT_MID_DARK_RATIO_MAX = 0.60
LIGHT_GOOD_BRIGHT_RATIO_MAX = 0.15
LIGHT_MID_BRIGHT_RATIO_MAX = 0.45
LIGHT_GOOD_ASYMMETRY_MAX = 18.0
LIGHT_MID_ASYMMETRY_MAX = 55.0


def format_identity_for_display(name):
    """
    Make runtime identity labels a bit more readable on screen.
    """

    if name is None:
        return None

    return name.replace("custom:", "custom: ")


class RecognitionDisplayState:
    """
    Slow visual state machine layered on top of the recognition votes.

    The webcam overlay must look calm even when frame-level predictions
    fluctuate. This state manager keeps transient uncertainty internal
    and only promotes stable labels to the visible overlay.
    """

    SEARCHING = "SEARCHING"
    ANALYZING = "ANALYZING"
    SHOW_KNOWN = "SHOW_KNOWN"
    SHOW_UNKNOWN = "SHOW_UNKNOWN"

    def __init__(self):
        self.state = self.SEARCHING
        self.display_name = None
        self.state_since = None
        self.last_known_reinforcement = None
        self.incoherent_updates = 0
        self.uncertain_reason = "bootstrap"
        self.transition_counts = {
            self.SEARCHING: 0,
            self.ANALYZING: 0,
            self.SHOW_KNOWN: 0,
            self.SHOW_UNKNOWN: 0,
        }

    def _transition(self, new_state, now, logger, reason, display_name=None):
        state_changed = self.state != new_state
        name_changed = self.display_name != display_name

        self.state = new_state
        self.display_name = display_name
        self.state_since = now
        self.incoherent_updates = 0

        if new_state == self.SHOW_KNOWN:
            self.last_known_reinforcement = now
        elif new_state != self.SHOW_KNOWN:
            self.last_known_reinforcement = None

        if state_changed or name_changed:
            self.transition_counts[new_state] = self.transition_counts.get(new_state, 0) + 1
            shown_text = self.get_identity_text()
            logger(
                "Overlay transition: "
                f"state={new_state} | text={shown_text} | reason={reason}"
            )

    def _hold_active(self, now):
        if self.state_since is None:
            return False
        return (now - self.state_since) < DISPLAY_HOLD_SECONDS

    def _set_uncertain_reason(self, reason):
        self.uncertain_reason = reason

    def update(self, face_present, recognition_result, now, logger):
        """
        Update the visible recognition overlay state.
        """

        if not face_present:
            self._transition(
                self.SEARCHING,
                now,
                logger,
                reason="face_missing",
                display_name=None,
            )
            return

        final_name = recognition_result["final_name"]
        current_name = recognition_result["current_name"]
        candidate_name = recognition_result.get("candidate_name")
        candidate_votes = recognition_result.get("candidate_votes", 0)
        unknown_votes = recognition_result.get("unknown_votes", 0)
        history_size = recognition_result["history_size"]

        if self.state == self.SEARCHING:
            self._transition(
                self.ANALYZING,
                now,
                logger,
                reason="face_detected",
                display_name=None,
            )

        if self.state == self.ANALYZING:
            if final_name != "Unknown" and candidate_votes >= DISPLAY_KNOWN_ENTRY_VOTES:
                self._transition(
                    self.SHOW_KNOWN,
                    now,
                    logger,
                    reason=f"known_confirmed:{final_name}",
                    display_name=final_name,
                )
                return

            if (
                candidate_name == "Unknown"
                and unknown_votes >= DISPLAY_UNKNOWN_ENTRY_VOTES
            ):
                self._transition(
                    self.SHOW_UNKNOWN,
                    now,
                    logger,
                    reason="unknown_confirmed",
                    display_name=None,
                )
                return

            if history_size < DISPLAY_KNOWN_ENTRY_VOTES:
                self._set_uncertain_reason("bootstrap")
            elif candidate_name not in (None, "Unknown"):
                self._set_uncertain_reason("conflict")
            else:
                self._set_uncertain_reason("unknown_pending")
            return

        if self.state == self.SHOW_KNOWN:
            shown_name = self.display_name
            same_name_confirmed = final_name == shown_name
            same_name_pending = current_name == shown_name or candidate_name == shown_name

            if same_name_confirmed:
                self.last_known_reinforcement = now
                self.incoherent_updates = 0
                return

            if self._hold_active(now):
                return

            if same_name_pending:
                self.incoherent_updates = 0
                return

            self.incoherent_updates += 1
            self._set_uncertain_reason("decay")

            confirmation_timed_out = (
                self.last_known_reinforcement is not None
                and (now - self.last_known_reinforcement)
                >= DISPLAY_KNOWN_CONFIRM_TIMEOUT_SECONDS
            )
            unknown_is_persistent = (
                candidate_name == "Unknown"
                and unknown_votes >= DISPLAY_UNKNOWN_ENTRY_VOTES
            )

            if (
                self.incoherent_updates >= DISPLAY_KNOWN_EXIT_INCOHERENT
                or confirmation_timed_out
                or unknown_is_persistent
            ):
                self._transition(
                    self.ANALYZING,
                    now,
                    logger,
                    reason="known_decay",
                    display_name=None,
                )
            return

        if self.state == self.SHOW_UNKNOWN:
            unknown_still_confirmed = (
                candidate_name == "Unknown"
                and unknown_votes >= DISPLAY_UNKNOWN_ENTRY_VOTES
            )
            known_is_confirmed = (
                final_name != "Unknown"
                and candidate_votes >= DISPLAY_KNOWN_ENTRY_VOTES
            )

            if unknown_still_confirmed:
                self.incoherent_updates = 0
                return

            if self._hold_active(now):
                return

            if known_is_confirmed:
                self._transition(
                    self.ANALYZING,
                    now,
                    logger,
                    reason=f"known_candidate_after_unknown:{final_name}",
                    display_name=None,
                )
                return

            self.incoherent_updates += 1
            self._set_uncertain_reason("conflict")
            if self.incoherent_updates >= DISPLAY_UNKNOWN_EXIT_INCOHERENT:
                self._transition(
                    self.ANALYZING,
                    now,
                    logger,
                    reason="unknown_released",
                    display_name=None,
                )

    def get_identity_text(self):
        """
        Return the user-facing identity line for the current overlay.
        """

        if self.state == self.SEARCHING:
            return "Searching face..."
        if self.state == self.ANALYZING:
            return "Analyzing..."
        if self.state == self.SHOW_UNKNOWN:
            return "Identity: Unknown"
        return f"Identity: {format_identity_for_display(self.display_name)}"

    def get_identity_color(self):
        """
        Return the overlay color for the current identity line.
        """

        if self.state == self.SHOW_KNOWN:
            return (0, 255, 0)
        if self.state == self.SHOW_UNKNOWN:
            return (0, 0, 255)
        return (255, 255, 0)


def log_bs_event(message):
    """
    Print a concise recognition-runtime debug message.
    """

    print(f"[BS] {message}")


def classify_light_band(mean_luma, dark_ratio, bright_ratio, asymmetry):
    """
    Classify the crop lighting into a compact diagnostic band.
    """

    if (
        mean_luma < LIGHT_MID_MEAN_MIN
        or mean_luma > LIGHT_MID_MEAN_MAX
        or dark_ratio > LIGHT_MID_DARK_RATIO_MAX
        or bright_ratio > LIGHT_MID_BRIGHT_RATIO_MAX
        or asymmetry > LIGHT_MID_ASYMMETRY_MAX
    ):
        return "bad_light"

    if (
        mean_luma < LIGHT_GOOD_MEAN_MIN
        or mean_luma > LIGHT_GOOD_MEAN_MAX
        or dark_ratio > LIGHT_GOOD_DARK_RATIO_MAX
        or bright_ratio > LIGHT_GOOD_BRIGHT_RATIO_MAX
        or asymmetry > LIGHT_GOOD_ASYMMETRY_MAX
    ):
        return "mid_light"

    return "good_light"


def compute_light_diagnostics(face_crop):
    """
    Compute compact, runtime-only diagnostics for the lighting on the crop.
    """

    if face_crop is None or face_crop.size == 0:
        return None

    crop = face_crop.astype(np.float32)
    mean_luma = float(np.mean(crop))
    std_luma = float(np.std(crop))
    dark_ratio = float(np.mean(crop < LIGHT_DARK_PIXEL_THRESHOLD))
    bright_ratio = float(np.mean(crop > LIGHT_BRIGHT_PIXEL_THRESHOLD))

    asymmetry = 0.0
    if crop.shape[1] >= 2:
        mid = crop.shape[1] // 2
        left_half = crop[:, :mid]
        right_half = crop[:, mid:]
        if left_half.size > 0 and right_half.size > 0:
            asymmetry = abs(
                float(np.mean(left_half)) - float(np.mean(right_half))
            )

    light_band = classify_light_band(
        mean_luma,
        dark_ratio,
        bright_ratio,
        asymmetry,
    )

    return {
        "band": light_band,
        "mean": mean_luma,
        "std": std_luma,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "asymmetry": asymmetry,
    }


def compute_bbox_geometry(current_rect, previous_rect):
    """
    Compute compact geometric diagnostics for a face bounding box.
    """

    x, y, w, h = current_rect
    area = w * h
    aspect_ratio = (w / h) if h > 0 else 0.0

    delta_x = delta_y = delta_w = delta_h = 0
    if previous_rect is not None:
        prev_x, prev_y, prev_w, prev_h = previous_rect
        delta_x = x - prev_x
        delta_y = y - prev_y
        delta_w = w - prev_w
        delta_h = h - prev_h

    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "area": area,
        "aspect_ratio": aspect_ratio,
        "delta_x": delta_x,
        "delta_y": delta_y,
        "delta_w": delta_w,
        "delta_h": delta_h,
    }


class RecognitionRuntimeLogStats:
    """
    Track compact runtime recognition statistics for periodic and final
    summaries.
    """

    def __init__(self):
        self.start_time = time.monotonic()
        self.last_summary_time = self.start_time
        self.prediction_checks = 0
        self.usable_checks = 0
        self.accepted_checks = 0
        self.rejected_checks = 0
        self.accepted_confidences = []
        self.rejected_confidences = []
        self.light_band_counts = {
            "good_light": 0,
            "mid_light": 0,
            "bad_light": 0,
            "unknown": 0,
        }
        self.light_means = []
        self.light_stds = []
        self.light_dark_ratios = []
        self.light_bright_ratios = []
        self.light_asymmetries = []
        self.overlay_known_transitions = 0
        self.overlay_unknown_transitions = 0
        self.last_final_name = "Unknown"

    def register_prediction_check(
        self,
        usable,
        accepted,
        final_name,
        confidence=None,
        light_band=None,
        light_mean=None,
        light_std=None,
        light_dark_ratio=None,
        light_bright_ratio=None,
        light_asymmetry=None,
    ):
        self.prediction_checks += 1
        if usable:
            self.usable_checks += 1
        if accepted:
            self.accepted_checks += 1
            if confidence is not None:
                self.accepted_confidences.append(confidence)
        else:
            self.rejected_checks += 1
            if confidence is not None:
                self.rejected_confidences.append(confidence)
        
        band_key = light_band if light_band in self.light_band_counts else "unknown"
        self.light_band_counts[band_key] += 1
        if light_mean is not None:
            self.light_means.append(light_mean)
        if light_std is not None:
            self.light_stds.append(light_std)
        if light_dark_ratio is not None:
            self.light_dark_ratios.append(light_dark_ratio)
        if light_bright_ratio is not None:
            self.light_bright_ratios.append(light_bright_ratio)
        if light_asymmetry is not None:
            self.light_asymmetries.append(light_asymmetry)
        self.last_final_name = final_name

    def register_overlay_transition(self, state):
        if state == "SHOW_KNOWN":
            self.overlay_known_transitions += 1
        elif state == "SHOW_UNKNOWN":
            self.overlay_unknown_transitions += 1

    def maybe_log_summary(self, now, logger):
        if (now - self.last_summary_time) < RECOGNITION_SUMMARY_INTERVAL_SECONDS:
            return

        elapsed = now - self.start_time
        logger(
            "Recognition summary: "
            f"elapsed={elapsed:.1f}s | "
            f"prediction_checks={self.prediction_checks} | "
            f"usable_checks={self.usable_checks} | "
            f"accepted={self.accepted_checks} | "
            f"rejected={self.rejected_checks} | "
            f"light_bands={self._format_light_band_counts()} | "
            f"known_transitions={self.overlay_known_transitions} | "
            f"unknown_transitions={self.overlay_unknown_transitions} | "
            f"last_final={self.last_final_name}"
        )
        self.last_summary_time = now

    def _format_confidence_stats(self, confidences):
        if not confidences:
            return "n/a"

        min_conf = min(confidences)
        max_conf = max(confidences)
        mean_conf = sum(confidences) / len(confidences)
        return f"mean={mean_conf:.2f}, min={min_conf:.2f}, max={max_conf:.2f}"

    def _format_series_mean(self, values):
        if not values:
            return "n/a"

        return f"{(sum(values) / len(values)):.2f}"

    def _format_light_band_counts(self):
        return (
            f"good={self.light_band_counts['good_light']}, "
            f"mid={self.light_band_counts['mid_light']}, "
            f"bad={self.light_band_counts['bad_light']}, "
            f"unknown={self.light_band_counts['unknown']}"
        )

    def format_final_summary(self):
        elapsed = time.monotonic() - self.start_time
        usable_ratio = (
            (self.usable_checks / self.prediction_checks) * 100.0
            if self.prediction_checks > 0
            else 0.0
        )
        accepted_ratio = (
            (self.accepted_checks / self.prediction_checks) * 100.0
            if self.prediction_checks > 0
            else 0.0
        )
        report_lines = [
            "Recognition final report:",
            f"  elapsed={elapsed:.1f}s",
            f"  prediction_checks={self.prediction_checks}",
            f"  usable_checks={self.usable_checks} ({usable_ratio:.1f}%)",
            f"  accepted={self.accepted_checks} ({accepted_ratio:.1f}%)",
            f"  rejected={self.rejected_checks}",
            f"  accepted_confidence={self._format_confidence_stats(self.accepted_confidences)}",
            f"  rejected_confidence={self._format_confidence_stats(self.rejected_confidences)}",
            f"  light_bands={self._format_light_band_counts()}",
            f"  light_mean={self._format_series_mean(self.light_means)}",
            f"  light_std={self._format_series_mean(self.light_stds)}",
            f"  light_dark_ratio={self._format_series_mean(self.light_dark_ratios)}",
            f"  light_bright_ratio={self._format_series_mean(self.light_bright_ratios)}",
            f"  light_asymmetry={self._format_series_mean(self.light_asymmetries)}",
            f"  show_known_transitions={self.overlay_known_transitions}",
            f"  show_unknown_transitions={self.overlay_unknown_transitions}",
            f"  last_final={self.last_final_name}",
        ]
        return "\n".join(report_lines)


def build_recognition_log_signature(recognition_result):
    """
    Build the compact signature used to suppress repeated debug lines.
    """

    return (
        recognition_result["current_name"],
        recognition_result["candidate_name"],
        recognition_result["candidate_votes"],
        recognition_result["unknown_votes"],
        recognition_result["final_name"],
        recognition_result["stable_count"],
        recognition_result["history_size"],
        recognition_result.get("light_band"),
    )


def create_recognition_runtime_state():
    """
    Build the temporal recognition state used by the recognition-only
    runtime.
    """

    return RecognitionRuntimeState(
        window_size=RECOGNITION_WINDOW_SIZE,
        min_confirmations=RECOGNITION_MIN_CONFIRMATIONS,
        reset_after_misses=RECOGNITION_RESET_AFTER_MISSES,
    )


def draw_recognition_overlay(
    frame,
    face_rect,
    recognition_text,
    identity_text,
    identity_color,
):
    """
    Draw a compact overlay for the recognition-only runtime.
    """

    x, y, w, h = face_rect
    top_line_y = max(25, y - 35)
    bottom_line_1_y = y + h + 25
    bottom_line_2_y = y + h + 55

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        frame,
        "State: RECOGNITION",
        (x, top_line_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        recognition_text,
        (x, bottom_line_1_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        identity_text,
        (x, bottom_line_2_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        identity_color,
        2,
        cv2.LINE_AA,
    )


def run_pipeline_bs():
    """
    Execute the recognition-only runtime pipeline.

    This entry point is meant for biometric-system experiments where
    the recognition module must be observed without interference from
    PAD logic.
    """

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    detector = load_mediapipe_detector()
    recognizer, label_map = load_recognizer()
    recognition_state = create_recognition_runtime_state()
    display_state = RecognitionDisplayState()
    runtime_log_stats = RecognitionRuntimeLogStats()
    last_recognition_result = {
        "final_name": "Unknown",
        "stable_confidence": None,
        "stable_count": 0,
        "history_size": 0,
        "candidate_name": None,
        "candidate_votes": 0,
        "unknown_votes": 0,
        "current_name": None,
        "current_confidence": None,
        "missing_face_counter": 0,
    }
    last_logged_signature = None
    previous_face_rect = None

    frame_counter = 0
    log_bs_event(
        f"Recognition runtime threshold: {RUNTIME_RECOGNITION_THRESHOLD:.2f}"
    )

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        frame_counter += 1
        face_rect, gray_frame, message = get_valid_face(
            frame,
            detector,
            THRESHOLD_AREA,
        )

        if message != "Face detected":
            previous_face_rect = None
            recognition_result = recognition_state.update(face_missing=True)
            last_recognition_result = recognition_result
            display_state.update(
                face_present=False,
                recognition_result=recognition_result,
                now=time.monotonic(),
                logger=log_bs_event,
            )
            runtime_log_stats.register_overlay_transition(display_state.state)
            runtime_log_stats.maybe_log_summary(time.monotonic(), log_bs_event)
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

            if recognition_result["final_name"] != "Unknown":
                cv2.putText(
                    frame,
                    f"Last identity: {recognition_result['final_name']}",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Recognition Runtime", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            continue

        x, y, w, h = face_rect
        current_face_rect = (x, y, w, h)
        geometry_result = compute_bbox_geometry(
            current_face_rect,
            previous_face_rect,
        )
        stability_result = evaluate_bbox_stability(
            current_face_rect,
            previous_face_rect,
        )
        bbox_stable = stability_result["is_stable"]
        previous_face_rect = current_face_rect
        recognition_face_crop = extract_pad_face_crop(gray_frame, x, y, w, h)
        quality_result = evaluate_pad_crop_quality(
            recognition_face_crop,
            PAD_MIN_VARIANCE_LAPLACIAN,
        )
        should_run_recognition = (frame_counter % RECOGNITION_RUN_INTERVAL == 0)

        if should_run_recognition:
            recognition_frame_usable = (
                bbox_stable
                and quality_result["is_valid_crop"]
                and quality_result["is_sharp_enough"]
            )
            gate_reasons = []
            if not bbox_stable:
                gate_reasons.append("bbox_unstable")
            if not quality_result["is_valid_crop"]:
                gate_reasons.append("crop_invalid")
            if not quality_result["is_sharp_enough"]:
                gate_reasons.append("crop_blurry")
            gate_reason_text = "ok" if len(gate_reasons) == 0 else ",".join(gate_reasons)
            face = None
            light_diagnostics = None
            if recognition_frame_usable:
                light_diagnostics = compute_light_diagnostics(recognition_face_crop)
                face = preprocess_face_for_recognition(gray_frame, x, y, w, h)

            if face is None:
                recognition_result = recognition_state.update(
                    frame_usable=False,
                    recognition_allowed=True,
                    face_missing=False,
                )
            else:
                predicted_name, confidence = predict_identity(
                    face,
                    recognizer,
                    label_map,
                    threshold=RUNTIME_RECOGNITION_THRESHOLD,
                )
                recognition_result = recognition_state.update(
                    predicted_name=predicted_name,
                    confidence=confidence,
                    frame_usable=True,
                    recognition_allowed=True,
                    face_missing=False,
                )
                if light_diagnostics is not None:
                    recognition_result["light_band"] = light_diagnostics["band"]
                    recognition_result["light_mean"] = light_diagnostics["mean"]
                    recognition_result["light_std"] = light_diagnostics["std"]
                    recognition_result["light_dark_ratio"] = light_diagnostics["dark_ratio"]
                    recognition_result["light_bright_ratio"] = light_diagnostics["bright_ratio"]
                    recognition_result["light_asymmetry"] = light_diagnostics["asymmetry"]
                runtime_log_stats.register_prediction_check(
                    usable=recognition_frame_usable,
                    accepted=(confidence < RUNTIME_RECOGNITION_THRESHOLD),
                    final_name=recognition_result["final_name"],
                    confidence=confidence,
                    light_band=recognition_result.get("light_band"),
                    light_mean=recognition_result.get("light_mean"),
                    light_std=recognition_result.get("light_std"),
                    light_dark_ratio=recognition_result.get("light_dark_ratio"),
                    light_bright_ratio=recognition_result.get("light_bright_ratio"),
                    light_asymmetry=recognition_result.get("light_asymmetry"),
                )
                confidence_text = (
                    f"{recognition_result['current_confidence']:.2f}"
                    if recognition_result["current_confidence"] is not None
                    else "None"
                )
                threshold_text = f"{RUNTIME_RECOGNITION_THRESHOLD:.2f}"
                accepted_text = (
                    "accepted"
                    if confidence < RUNTIME_RECOGNITION_THRESHOLD
                    else "rejected"
                )
                current_signature = build_recognition_log_signature(recognition_result)
                if current_signature != last_logged_signature:
                    light_text = "n/a"
                    if light_diagnostics is not None:
                        light_text = (
                            f"{light_diagnostics['band']} | "
                            f"mean={light_diagnostics['mean']:.2f} | "
                            f"std={light_diagnostics['std']:.2f} | "
                            f"dark={light_diagnostics['dark_ratio']:.2f} | "
                            f"bright={light_diagnostics['bright_ratio']:.2f} | "
                            f"asym={light_diagnostics['asymmetry']:.2f}"
                        )
                    geometry_text = (
                        f"bbox=({geometry_result['x']},{geometry_result['y']},"
                        f"{geometry_result['w']},{geometry_result['h']}) | "
                        f"delta=({geometry_result['delta_x']},"
                        f"{geometry_result['delta_y']},"
                        f"{geometry_result['delta_w']},"
                        f"{geometry_result['delta_h']}) | "
                        f"area={geometry_result['area']} | "
                        f"aspect={geometry_result['aspect_ratio']:.2f} | "
                        f"gate={gate_reason_text}"
                    )
                    log_bs_event(
                        "Recognition check: "
                        f"usable={recognition_frame_usable} | "
                        f"threshold={threshold_text} | "
                        f"decision={accepted_text} | "
                        f"{geometry_text} | "
                        f"current={recognition_result['current_name']} | "
                        f"confidence={confidence_text} | "
                        f"light={light_text} | "
                        f"candidate={recognition_result['candidate_name']} | "
                        f"candidate_votes={recognition_result['candidate_votes']} | "
                        f"unknown_votes={recognition_result['unknown_votes']} | "
                        f"final={recognition_result['final_name']} | "
                        f"votes={recognition_result['stable_count']}/"
                        f"{recognition_result['history_size']}"
                    )
                    last_logged_signature = current_signature
            last_recognition_result = recognition_result
        else:
            recognition_result = last_recognition_result

        display_state.update(
            face_present=True,
            recognition_result=recognition_result,
            now=time.monotonic(),
            logger=log_bs_event,
        )
        runtime_log_stats.register_overlay_transition(display_state.state)
        runtime_log_stats.maybe_log_summary(time.monotonic(), log_bs_event)

        recognition_text = "Recognition active"
        identity_text = display_state.get_identity_text()
        identity_color = display_state.get_identity_color()

        draw_recognition_overlay(
            frame,
            face_rect,
            recognition_text,
            identity_text,
            identity_color,
        )

        cv2.imshow("Recognition Runtime", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    log_bs_event(runtime_log_stats.format_final_summary())
