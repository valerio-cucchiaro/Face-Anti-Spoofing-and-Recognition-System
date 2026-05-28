"""
Entry point for the recognition-only runtime.

This script launches the webcam pipeline dedicated to face
recognition. It is useful when recognition must be tested in
isolation without the PAD gate.
"""

import argparse

from src.pipeline.run_pipeline_recognition import run_pipeline_recognition


def build_arg_parser():
    """
    Build the CLI for the recognition-only webcam runtime.
    """

    parser = argparse.ArgumentParser(
        description="Run the recognition-only webcam runtime."
    )
    parser.add_argument(
        "--log-level",
        choices=["normal", "verbose"],
        default="normal",
        help="Terminal logging level for the recognition runtime. Defaults to normal.",
    )
    return parser


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    run_pipeline_recognition(log_level=cli_args.log_level)
