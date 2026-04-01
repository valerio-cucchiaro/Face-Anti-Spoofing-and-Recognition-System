"""
Entry point for the full biometric runtime.

This script launches the complete webcam pipeline that combines
face detection, PAD, and recognition inside the runtime state
machine used for the integrated demo.
"""

from src.pipeline.run_pipeline_full import run_pipeline_full
    

if __name__ == "__main__":
    run_pipeline_full()
