"""
Module: face_bbox_visualization

Description:
This module provides a single visual face-box contract for webcam
overlays across runtime pipelines and dataset collectors.

The visual box is intentionally separate from the PAD crop so the
display geometry can evolve independently from the model input
geometry.
"""

import cv2


VISUAL_FACE_BOX_SCALE = 1.28
VISUAL_FACE_BOX_TOP_BIAS_RATIO = 0.14


def build_visual_face_bbox(
    frame,
    x,
    y,
    w,
    h,
    box_scale=VISUAL_FACE_BOX_SCALE,
    top_bias_ratio=VISUAL_FACE_BOX_TOP_BIAS_RATIO,
):
    """
    Build the visual bbox used only for on-screen overlays.
    """

    if frame is None or frame.size == 0:
        return None

    box_side = int(max(w, h) * box_scale)
    box_side = max(box_side, 1)

    center_x = x + (w // 2)
    center_y = y + int(h * (0.5 - top_bias_ratio))

    x1 = center_x - (box_side // 2)
    y1 = center_y - (box_side // 2)
    x2 = x1 + box_side
    y2 = y1 + box_side

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > frame.shape[1]:
        shift = x2 - frame.shape[1]
        x1 = max(0, x1 - shift)
        x2 = frame.shape[1]
    if y2 > frame.shape[0]:
        shift = y2 - frame.shape[0]
        y1 = max(0, y1 - shift)
        y2 = frame.shape[0]

    if x1 >= x2 or y1 >= y2:
        return None

    return x1, y1, x2, y2


def draw_visual_face_bbox(frame, face_rect, color, thickness=2):
    """
    Draw the shared visual bbox on the provided frame.
    """

    if frame is None or face_rect is None:
        return None

    x, y, w, h = face_rect
    bounds = build_visual_face_bbox(frame, x, y, w, h)
    if bounds is None:
        return None

    x1, y1, x2, y2 = bounds
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return bounds
