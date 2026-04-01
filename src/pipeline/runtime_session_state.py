"""
Module: runtime_session_state

Description:
This module stores the lightweight session state used by the runtime
pipeline.

The runtime now behaves as a small state machine with four phases:
- DETECT
- PAD_CHECK
- RECOGNITION_LOCKED
- RESET

This helper keeps the persistent variables needed to move between those
phases without coupling the session bookkeeping too tightly to the main
frame-processing loop.
"""


class RuntimeSessionState:
    """
    Lightweight runtime session state for the webcam pipeline.
    """

    DETECT = "DETECT"
    PAD_CHECK = "PAD_CHECK"
    RECOGNITION_LOCKED = "RECOGNITION_LOCKED"
    RESET = "RESET"

    def __init__(self):
        """
        Initialize the runtime session in the DETECT phase.
        """

        self.current_phase = self.DETECT
        self.last_valid_face_rect = None
        self.previous_face_rect = None
        self.pad_live_confirmation_count = 0
        self.live_lock_active = False
        self.missing_face_counter = 0
        self.locked_unstable_counter = 0
        self.locked_invalid_crop_counter = 0
        self.last_stable_identity = "Unknown"
        self.last_recognition_result = {
            "final_name": "Unknown",
            "stable_confidence": None,
            "stable_count": 0,
            "history_size": 0,
            "current_name": None,
            "current_confidence": None,
            "missing_face_counter": 0,
        }
        self.last_pad_result = {
            "final_label": "unknown",
            "mean_live_probability": None,
            "current_status": None,
            "current_label": None,
            "current_score": None,
            "missing_face_counter": 0,
        }

    def transition_to(self, new_phase):
        """
        Move the session to a new phase.

        The method returns ``True`` only when the phase actually changes.
        """

        if self.current_phase == new_phase:
            return False

        self.current_phase = new_phase
        return True

    def register_missing_face(self):
        """
        Record one frame without a usable face.
        """

        self.missing_face_counter += 1
        self.last_valid_face_rect = None

    def clear_missing_face_counter(self):
        """
        Clear the missing-face counter after a valid face is found again.
        """

        self.missing_face_counter = 0

    def store_face_rect(self, face_rect):
        """
        Persist the most recent valid face rectangle.
        """

        self.last_valid_face_rect = face_rect
        self.previous_face_rect = face_rect

    def register_pad_live_confirmation(self):
        """
        Count one more PAD update that supports live locking.
        """

        self.pad_live_confirmation_count += 1
        return self.pad_live_confirmation_count

    def reset_pad_live_confirmation(self):
        """
        Clear the live-lock confirmation counter.
        """

        self.pad_live_confirmation_count = 0

    def register_locked_bbox_stability(self, is_stable):
        """
        Update the counter used to detect instability during lock.
        """

        if is_stable:
            self.locked_unstable_counter = 0
        else:
            self.locked_unstable_counter += 1

        return self.locked_unstable_counter

    def register_locked_crop_validity(self, is_valid_crop):
        """
        Update the counter used to detect persistent invalid crops
        during the recognition lock phase.
        """

        if is_valid_crop:
            self.locked_invalid_crop_counter = 0
        else:
            self.locked_invalid_crop_counter += 1

        return self.locked_invalid_crop_counter

    def store_pad_result(self, pad_result):
        """
        Save the latest PAD state result.
        """

        self.last_pad_result = pad_result

    def store_recognition_result(self, recognition_result):
        """
        Save the latest recognition state result.
        """

        self.last_recognition_result = recognition_result

        final_name = recognition_result.get("final_name")
        if final_name is not None and final_name != "Unknown":
            self.last_stable_identity = final_name

    def reset_session(self):
        """
        Reset the session to its initial DETECT state.
        """

        self.current_phase = self.DETECT
        self.last_valid_face_rect = None
        self.previous_face_rect = None
        self.pad_live_confirmation_count = 0
        self.live_lock_active = False
        self.missing_face_counter = 0
        self.locked_unstable_counter = 0
        self.locked_invalid_crop_counter = 0
        self.last_stable_identity = "Unknown"
        self.last_recognition_result = {
            "final_name": "Unknown",
            "stable_confidence": None,
            "stable_count": 0,
            "history_size": 0,
            "current_name": None,
            "current_confidence": None,
            "missing_face_counter": 0,
        }
        self.last_pad_result = {
            "final_label": "unknown",
            "mean_live_probability": None,
            "current_status": None,
            "current_label": None,
            "current_score": None,
            "missing_face_counter": 0,
        }
