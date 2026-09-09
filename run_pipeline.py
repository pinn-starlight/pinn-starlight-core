"""Stage10-A formal pipeline runner.

The runner is deliberately conservative:
- existing synthetic data are NOT rebuilt unless execution starts at ``synthetic``;
- E1 explicitly uses the validated nested-autograd PINN Laplacian;
- Fourier-LS selection explicitly uses the full-resolution CUDA backend;
- Global Lock is created before E2/E3/E4;
- formal benchmark stages use the central single-GPU execution policy.

Recommended: run this file with the project's Pixi Python interpreter.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


STAGES = (
    ("synthetic", [sys.executable, "-m", "experiments.common.data.generate_synthetic"]),
    ("preflight", [sys.executable, "-m", "experiments.scripts.preflight.run"]),
    ("e0", [sys.executable, "-m", "experiments.scripts.e0.run"]),
    (
        "e1",
        [
            sys.executable,
            "-m",
            "experiments.scripts.e1.run",
            "--device",
            "cuda:0",
            "--laplacian-backend",
            "autograd",
        ],
    ),
    ("gaussian", [sys.executable, "-m", "experiments.scripts.selection.gaussian.run"]),
    (
        "fourier",
        [
            sys.executable,
            "-m",
            "experiments.scripts.selection.fourier_ls.run",
            "--backend",
            "cuda",
        ],
    ),
    ("unet", [sys.executable, "-m", "experiments.scripts.selection.unet.run"]),
    ("lock", [sys.executable, "-m", "experiments.scripts.config_lock.run"]),
    ("e2", [sys.executable, "-m", "experiments.scripts.e2.run"]),
    ("e3", [sys.executable, "-m", "experiments.scripts.e3.run"]),
    ("e4", [sys.executable, "-m", "experiments.scripts.e4.run"]),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-at",
        choices=[name for name, _ in STAGES],
        default="preflight",
        help="first stage to execute; default preserves existing synthetic data",
    )
    parser.add_argument(
        "--stop-after",
        choices=[name for name, _ in STAGES],
        default="e4",
        help="last stage to execute",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="pass --force to stages that support destructive output replacement",
    )
    args = parser.parse_args()

    names = [name for name, _ in STAGES]
    start = names.index(args.start_at)
    stop = names.index(args.stop_after)
    if start > stop:
        raise SystemExit("--start-at must not come after --stop-after")

    force_capable = {
        "synthetic", "e0", "e1", "gaussian", "fourier", "unet",
        "lock", "e2", "e3", "e4",
    }
    selected = STAGES[start : stop + 1]

    for idx, (name, base_cmd) in enumerate(selected, start=1):
        cmd = list(base_cmd)
        if args.force and name in force_capable:
            cmd.append("--force")
        print(
            f"\n===== Stage10-A {idx}/{len(selected)} [{name}]: "
            f"{' '.join(cmd)} =====",
            flush=True,
        )
        proc = subprocess.run(cmd, env=os.environ.copy())
        if proc.returncode != 0:
            raise SystemExit(
                f"Stage10-A stage {name!r} failed with exit code "
                f"{proc.returncode}; pipeline stopped."
            )

    print("\nStage10-A requested pipeline segment finished successfully.")


if __name__ == "__main__":
    main()
