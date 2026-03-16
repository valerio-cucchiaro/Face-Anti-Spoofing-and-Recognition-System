"""
Module: spoof_detector

Description:
This module implements the main anti-spoofing controller used in the
face recognition pipeline.

It combines texture analysis and robust blink information to determine
whether a detected face is real or spoofed.

The detector does not validate blinks internally. Blink detection is
handled by the dedicated blink detector module. This class only keeps
track of liveness confirmation over time and aggregates the final
anti-spoofing decision.
"""

from src.antispoofing.texture_analyzer import analyze_texture


class AntiSpoofingDetector:
    """
    Anti-spoofing controller that combines texture analysis and robust
    blink detection to determine whether a face is real or spoofed.
    """

    def __init__(self):
        """
        Initialize the anti-spoofing detector state.
        """

        # Flag indicating whether liveness has been confirmed
        self.liveness_confirmed = False

    def reset_state(self):
        """
        Reset the internal anti-spoofing state.

        This should be called when the face disappears or tracking is lost,
        so that liveness information is not reused across different subjects
        or disconnected face tracks.
        """

        self.liveness_confirmed = False

    def check_spoof(self, face, blink_detected, blink_count, eye_closed, ear):
        """
        Perform the final anti-spoofing evaluation.

        The function combines:
        - texture analysis on the current face crop
        - robust blink information received from the blink detector

        Liveness is confirmed only when a valid blink is detected on a face
        whose texture is considered acceptable.
        """

        # Analyze the texture of the face region
        texture_ok, sharpness, edge_density = analyze_texture(face)

        # Confirm liveness only when both cues agree
        if texture_ok and blink_detected:
            self.liveness_confirmed = True

        # Final anti-spoofing decision
        is_real = texture_ok and self.liveness_confirmed

        # High-level anti-spoofing status for UI/debug purposes
        if is_real:
            status = "real"
        elif not texture_ok:
            status = "spoof"
        else:
            status = "checking"

        return {
            "status": status,
            "is_real": is_real,
            "texture_ok": texture_ok,
            "sharpness": sharpness,
            "edge_density": edge_density,
            "blink_detected": blink_detected,
            "eye_closed": eye_closed,
            "ear": ear,
            "blink_count": blink_count,
            "liveness_confirmed": self.liveness_confirmed
        }