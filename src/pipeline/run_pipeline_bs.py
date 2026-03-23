"""
Module: run_pipeline_bs

Description:
This legacy pipeline entry point is intentionally disabled.

The active project pipeline is ``run_pipeline_full.py``, which already
integrates MediaPipe detection, PAD, and recognition. This placeholder
remains only to avoid broken imports while the older experiment is
retired.
"""


def run_pipeline_bs():
    """Raise an explicit error because this legacy pipeline is disabled."""

    raise NotImplementedError(
        "run_pipeline_bs is disabled. Use run_pipeline_full instead."
    )
