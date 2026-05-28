"""
Entry point for the PAD-only runtime.

This script launches the webcam pipeline dedicated to face
anti-spoofing. It is useful when PAD and liveness behavior must be
tested without the recognition stage.
"""

import argparse

from src.pipeline.run_pipeline_pad import run_pipeline_pad


def build_arg_parser():
    """
    Build the CLI for the PAD-only webcam runtime.
    """

    parser = argparse.ArgumentParser(
        description="Run the PAD-only webcam runtime with optional artifact overrides."
    )
    parser.add_argument("--pad-artifact-profile", choices=["runtime", "full", "fast"])
    parser.add_argument(
        "--pad-artifact-tag",
        help="Optional PAD artifact tag used to load a non-canonical PAD bundle.",
    )
    parser.add_argument(
        "--log-level",
        choices=["normal", "verbose"],
        default="normal",
        help="Terminal logging level for the PAD runtime. Defaults to normal.",
    )
    return parser


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    run_pipeline_pad(
        pad_artifact_profile=cli_args.pad_artifact_profile or "runtime",
        pad_artifact_tag=cli_args.pad_artifact_tag,
        log_level=cli_args.log_level,
    )
