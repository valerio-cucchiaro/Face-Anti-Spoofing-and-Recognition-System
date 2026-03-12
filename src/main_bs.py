"""
Entry point for the Biometric Systems pipeline.

This script runs the biometric pipeline of the project, including
face detection, preprocessing, recognition, and anti-spoofing
modules used for biometric identity verification.
"""


from src.pipeline.run_pipeline_bs import run_pipeline_bs
    

if __name__ == "__main__":
    run_pipeline_bs()