"""
Entry point for the full biometric system pipeline.

This script starts the complete runtime pipeline of the project,
including face detection, preprocessing, recognition, and any
additional modules integrated into the full system.

It is intended to run the complete demonstration version of the
project.
"""

from src.pipeline.run_pipeline_full import run_pipeline_full
    

if __name__ == "__main__":
    run_pipeline_full()