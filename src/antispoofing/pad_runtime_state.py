"""
Module: pad_runtime_state

Description:
This module manages the temporal PAD state used during runtime face
anti-spoofing inference.

The state manager receives the PAD result of the current frame together
with frame usability and face-missing information. It updates the
temporal history, applies hysteresis on the averaged live probability,
and returns the current and final PAD outputs used by the pipeline.
"""

from collections import deque


class PadRuntimeState:
    """
    Runtime PAD state manager with temporal smoothing and hysteresis.
    """

    def __init__(
        self,
        window_size,
        enter_live_threshold,
        exit_live_threshold,
        enter_spoof_threshold,
        exit_spoof_threshold,
        reset_after_misses,
    ):
        """
        Initialize the runtime PAD state and its temporal thresholds.
        """

        self.pad_history = deque(maxlen=window_size)
        self.pad_state = "unknown"
        self.missing_face_counter = 0

        self.enter_live_threshold = enter_live_threshold
        self.exit_live_threshold = exit_live_threshold
        self.enter_spoof_threshold = enter_spoof_threshold
        self.exit_spoof_threshold = exit_spoof_threshold
        self.reset_after_misses = reset_after_misses

    def update(self, pad_result=None, frame_usable=False, face_missing=False):
        """
        Update the runtime PAD state using the current frame result.
        """

        if face_missing:
            self.missing_face_counter += 1
            if self.missing_face_counter >= self.reset_after_misses:
                self.pad_history.clear()
                self.pad_state = "unknown"

            return {
                "final_label": self.pad_state,
                "mean_live_probability": None,
                "current_status": None,
                "current_label": None,
                "current_score": None,
                "missing_face_counter": self.missing_face_counter,
            }

        self.missing_face_counter = 0

        if pad_result is None:
            pad_result = {"status": "invalid_crop", "label": None, "score": None}

        current_status = pad_result["status"]
        current_label = pad_result["label"]
        current_score = pad_result["score"]

        if current_status == "ok":
            if current_label == "live":
                live_probability = current_score
            elif current_label == "spoof":
                live_probability = 1.0 - current_score
            else:
                live_probability = None
        else:
            live_probability = None

        if live_probability is not None and frame_usable:
            self.pad_history.append(live_probability)

        if len(self.pad_history) > 0:
            mean_live_probability = sum(self.pad_history) / len(self.pad_history)
        else:
            mean_live_probability = None

        if mean_live_probability is None:
            self.pad_state = "unknown"

        elif self.pad_state == "unknown":
            if mean_live_probability >= self.enter_live_threshold:
                self.pad_state = "live"
            elif mean_live_probability <= self.enter_spoof_threshold:
                self.pad_state = "spoof"
            else:
                self.pad_state = "unknown"

        elif self.pad_state == "live":
            if mean_live_probability >= self.exit_live_threshold:
                self.pad_state = "live"
            elif mean_live_probability <= self.enter_spoof_threshold:
                self.pad_state = "spoof"
            else:
                self.pad_state = "unknown"

        elif self.pad_state == "spoof":
            if mean_live_probability <= self.exit_spoof_threshold:
                self.pad_state = "spoof"
            elif mean_live_probability >= self.enter_live_threshold:
                self.pad_state = "live"
            else:
                self.pad_state = "unknown"

        return {
            "final_label": self.pad_state,
            "mean_live_probability": mean_live_probability,
            "current_status": current_status,
            "current_label": current_label,
            "current_score": current_score,
            "missing_face_counter": self.missing_face_counter,
        }
