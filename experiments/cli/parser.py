"""argparse definition for ``pinn-starlight``.

Importing this module must stay cheap (no numpy / torch): ``version``, ``doctor``
and ``samples`` should answer in milliseconds on a fresh machine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.cli.commands.doctor import cmd_doctor
from experiments.cli.commands.estimate import cmd_estimate
from experiments.cli.commands.samples import cmd_samples
from experiments.cli.commands.version import cmd_version
from experiments.cli.context import DATA_ROOT_HELP, EXIT_OK, TOOL_NAME, tool_version
from experiments.configs.formal_experiment import FORMAL_SEEDS

EPILOG = """\
modes:
  star       SGC-PINN pipeline (best background / point-source accuracy)
  nebula     joint data + physics PINN (best diffuse-structure retention)
  balanced   physics-first init + anchored finetune (experimental, demo only)

examples:
  pinn-starlight doctor
  pinn-starlight samples --stage a
  pinn-starlight estimate --image photo.jpg --mode star
  pinn-starlight estimate --image photo.jpg --mode star,nebula
  pinn-starlight estimate --sample clean_single_002_multi_source_high --methods all

All estimate runs use CPU/demo budgets; they are not the paper numbers.
"""


def _cmd_help(args) -> int:
    args.top_level_parser.print_help()
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=TOOL_NAME,
        description="PINN-Starlight demo CLI (not the formal experiment pipeline).",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"{TOOL_NAME} {tool_version()}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")

    version = subparsers.add_parser("version", help="print the tool and core versions")
    version.add_argument(
        "--json", action="store_true", help="machine-readable output (default: off)"
    )
    version.set_defaults(func=cmd_version)

    doctor = subparsers.add_parser(
        "doctor",
        help="report environment and data readiness",
        description=(
            "Check Python, the scientific stack, torch/CUDA and the data layout. "
            "Exit code 3 means something needed by estimate is missing."
        ),
    )
    doctor.add_argument(
        "--json", action="store_true", help="machine-readable output (default: off)"
    )
    doctor.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=DATA_ROOT_HELP,
    )
    doctor.set_defaults(func=cmd_doctor)

    samples = subparsers.add_parser(
        "samples",
        help="list validation samples on disk",
        description="List the validation split samples of the extracted synthetic datasets.",
    )
    samples.add_argument(
        "--stage",
        choices=("a", "b", "all"),
        default="all",
        help="which dataset to list (default: all)",
    )
    samples.add_argument(
        "--json", action="store_true", help="machine-readable output (default: off)"
    )
    samples.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=DATA_ROOT_HELP,
    )
    samples.set_defaults(func=cmd_samples)

    estimate = subparsers.add_parser(
        "estimate",
        help="estimate the background of one image",
        description=(
            "Estimate the light-pollution background of one image and write the "
            "background, residual, preview and summary.json to the output directory."
        ),
    )
    source = estimate.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--image", type=Path, help="any supported image file; needs no data root (no ground truth)"
    )
    source.add_argument(
        "--sample",
        help="validation sample_id from the extracted synthetic dataset (see `samples`)",
    )
    estimate.add_argument(
        "--dataset",
        choices=("stage_a", "stage_b"),
        default="stage_a",
        help="dataset that --sample belongs to (default: stage_a)",
    )
    estimate.add_argument(
        "--methods",
        default=None,
        help=(
            "all or a comma separated subset of "
            "fft_gaussian,fourier_ls,pinn,final_pinn,balanced; "
            "mutually exclusive with --mode (default: all four paper methods)"
        ),
    )
    estimate.add_argument(
        "--mode",
        default=None,
        metavar="MODE[,MODE...]",
        help=(
            "operating point(s), comma separated: star=final_pinn, nebula=pinn, "
            "balanced=anchored; each mode writes its own <out>/<name>/<method>/ "
            "subdirectory; mutually exclusive with --methods (default: not set)"
        ),
    )
    estimate.add_argument(
        "--max-side",
        type=int,
        default=384,
        help="downsample so the longest side is at most this many pixels (default: 384)",
    )
    estimate.add_argument(
        "--steps",
        type=int,
        default=500,
        help="training steps per stage; demo budget, not the paper budget (default: 500)",
    )
    estimate.add_argument(
        "--seed",
        type=int,
        default=FORMAL_SEEDS[0],
        help=f"random seed (default: {FORMAL_SEEDS[0]})",
    )
    estimate.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, or cuda:N; auto picks CUDA when available (default: auto)",
    )
    estimate.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "output root; one subdirectory per input is created. Default "
            "./pinn_starlight_output (current directory, so results stay visible and are "
            "not lost with a temp/cache cleanup; rerunning the same input overwrites it)"
        ),
    )
    estimate.add_argument(
        "--json", action="store_true", help="print summary.json to stdout (default: off)"
    )
    estimate.add_argument(
        "--no-figure",
        action="store_true",
        help="skip preview.png, useful on headless machines (default: off)",
    )
    estimate.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the progress bar and the per-method report (default: off)",
    )
    estimate.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="only used with --sample; " + DATA_ROOT_HELP,
    )
    estimate.set_defaults(func=cmd_estimate)

    help_parser = subparsers.add_parser(
        "help",
        help="show this message and exit",
        description="Same output as `pinn-starlight --help`.",
    )
    help_parser.set_defaults(func=_cmd_help, top_level_parser=parser)
    return parser
