"""
Module: run_pipeline_cv

Description:
This legacy pipeline entry point is intentionally disabled.

The active project pipeline is ``run_pipeline_full.py``, which already
integrates MediaPipe detection, PAD, and recognition. This placeholder
stays in the repository only as a retired compatibility stub.
"""


def run_pipeline_cv():
    """Raise an explicit error because this legacy pipeline is disabled."""

    raise NotImplementedError(
        "run_pipeline_cv is disabled. Use run_pipeline_full instead."
    )
