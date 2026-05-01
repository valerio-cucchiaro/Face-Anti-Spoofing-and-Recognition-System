"""
Entry point for the Biometric Systems recognition-only runtime.

This script launches the webcam pipeline dedicated to face
recognition. It is useful when recognition must be tested in
isolation without the PAD gate.
"""

from src.pipeline.run_pipeline_bs import run_pipeline_bs
    

if __name__ == "__main__":
    run_pipeline_bs()
