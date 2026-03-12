"""
Entry point for the Computer Vision pipeline.

This script launches the pipeline used for the Computer Vision
project, focusing on face detection, preprocessing, and visual
analysis tasks such as anti-spoofing or liveness detection.

Recognition may be excluded depending on the experimental setup.
"""

from src.pipeline.run_pipeline_cv import run_pipeline_cv
    

if __name__ == "__main__":
    run_pipeline_cv()