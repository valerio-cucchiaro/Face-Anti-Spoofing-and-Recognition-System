"""
Module: recognition_runtime_state

Description:
This module manages the temporal recognition state used during runtime
face recognition.

The state manager receives the recognition result of the current frame
together with frame usability and face-missing information. It updates
the short-term recognition history and returns both the current
frame-level prediction and the temporally confirmed identity used by
the runtime pipeline.
"""

from collections import Counter, deque


class RecognitionRuntimeState:
    """
    Runtime recognition state manager with short temporal smoothing.
    """

    def __init__(self, window_size, min_confirmations, reset_after_misses):
        """
        Initialize the runtime recognition state and its temporal rules.
        """

        self.recognition_history = deque(maxlen=window_size)
        self.final_name = "Unknown"
        self.missing_face_counter = 0

        self.min_confirmations = min_confirmations
        self.reset_after_misses = reset_after_misses

    def update(
        self,
        predicted_name=None,
        confidence=None,
        frame_usable=False,
        recognition_allowed=True,
        face_missing=False,
    ):
        """
        Update the runtime recognition state using the current frame result.

        When the face is missing or recognition is blocked by a previous
        stage such as PAD, the method keeps the current state for a short
        grace period and then resets it. When the current frame is
        simply not usable for recognition, the state is held without
        adding a new vote to the short-term history.

        The runtime may still store `"Unknown"` inside the short-term
        history. This is intentional: the temporal logic tracks both
        stable known identities and persistent rejection of weak matches.
        """

        if face_missing or not recognition_allowed:
            self.missing_face_counter += 1
            if self.missing_face_counter >= self.reset_after_misses:
                self.recognition_history.clear()
                self.final_name = "Unknown"

            return {
                "final_name": self.final_name,
                "stable_confidence": None,
                "stable_count": 0,
                "history_size": len(self.recognition_history),
                "candidate_name": None,
                "candidate_votes": 0,
                "unknown_votes": 0,
                "current_name": None,
                "current_confidence": None,
                "missing_face_counter": self.missing_face_counter,
            }

        self.missing_face_counter = 0

        if frame_usable and predicted_name is not None:
            self.recognition_history.append(
                {
                    "name": predicted_name,
                    "confidence": confidence,
                }
            )

        if len(self.recognition_history) == 0:
            self.final_name = "Unknown"
            stable_confidence = None
            stable_count = 0
            candidate_name = None
            candidate_votes = 0
            unknown_votes = 0
        else:
            # The most frequent recent identity becomes the candidate
            # stable output, but it must appear often enough and it must
            # not simply be the "Unknown" fallback.
            name_counts = Counter(
                entry["name"] for entry in self.recognition_history
            )
            stable_name, stable_count = name_counts.most_common(1)[0]
            candidate_name = stable_name
            candidate_votes = stable_count
            unknown_votes = name_counts.get("Unknown", 0)

            if stable_name != "Unknown" and stable_count >= self.min_confirmations:
                self.final_name = stable_name
                stable_confidences = [
                    entry["confidence"]
                    for entry in self.recognition_history
                    if entry["name"] == stable_name
                    and entry["confidence"] is not None
                ]
                if len(stable_confidences) > 0:
                    stable_confidence = sum(stable_confidences) / len(stable_confidences)
                else:
                    stable_confidence = None
            else:
                self.final_name = "Unknown"
                stable_confidence = None

        return {
            "final_name": self.final_name,
            "stable_confidence": stable_confidence,
            "stable_count": stable_count,
            "history_size": len(self.recognition_history),
            "candidate_name": candidate_name,
            "candidate_votes": candidate_votes,
            "unknown_votes": unknown_votes,
            "current_name": predicted_name,
            "current_confidence": confidence,
            "missing_face_counter": self.missing_face_counter,
        }
