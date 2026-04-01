"""
Entry point for the Computer Vision PAD-only runtime.

This script launches the webcam pipeline dedicated to face
anti-spoofing. It is useful when PAD and liveness behavior must be
tested without the recognition stage.
"""

from src.pipeline.run_pipeline_cv import run_pipeline_cv
    

if __name__ == "__main__":
    run_pipeline_cv()
