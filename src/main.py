"""
Entry point for the full biometric runtime.

This script launches the complete webcam pipeline that combines
face detection, PAD, and recognition inside the runtime state
machine used for the integrated demo.
"""

import argparse

from src.pipeline.run_pipeline_full import run_pipeline_full


def build_arg_parser():
    """
    Build the CLI for the full webcam runtime.
    """

    parser = argparse.ArgumentParser(
        description="Run the full webcam runtime with detection, PAD, and recognition."
    )
    parser.add_argument(
        "--log-level",
        choices=["normal", "verbose"],
        default="normal",
        help="Terminal logging level for the full runtime. Defaults to normal.",
    )
    return parser


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    run_pipeline_full(log_level=cli_args.log_level)
