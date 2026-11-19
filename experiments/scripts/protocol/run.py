"""Run the ordinary or grouped-five-fold Stage10 protocol using one code path.

This wrapper does not implement any scientific method.  It only selects a split
context and invokes the existing Stage10-A / Stage10-B selectors, freeze gates,
and formal evaluators in separate Python processes so every module sees the
same fold-specific manifest and output namespace.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from experiments.configs.split_protocols import validate_fold
from experiments.scripts.protocol.prepare_cv5 import prepare_stage

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _env(mode: str, fold: int | None) -> dict[str, str]:
    env = os.environ.copy()
    env["PINN_STARLIGHT_SPLIT_MODE"] = mode
    if mode == "cv5":
        env["PINN_STARLIGHT_CV_FOLD"] = str(validate_fold(fold))
    else:
        env.pop("PINN_STARLIGHT_CV_FOLD", None)
    return env


def _run_module(module: str, args: list[str], *, env: dict[str, str]) -> None:
    command = [sys.executable, "-m", module, *args]
    print("\n>", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def _phase_modules(stage: str, phase: str, *, force: bool, device: str, fourier_backend: str):
    force_arg = ["--force"] if force else []
    if stage == "a":
        if phase == "preflight":
            return [("experiments.scripts.preflight.run", [])]
        if phase == "select":
            return [
                ("experiments.scripts.selection.gaussian", force_arg),
                (
                    "experiments.scripts.selection.fourier_ls",
                    [*force_arg, "--backend", fourier_backend],
                ),
                (
                    "experiments.scripts.selection.pinn",
                    [*force_arg, "--device", device, "--laplacian-backend", "autograd"],
                ),
            ]
        if phase == "lock":
            return [("experiments.scripts.config_lock.run", force_arg)]
        if phase == "benchmark":
            return [("experiments.scripts.benchmark", force_arg)]
        if phase == "ablation":
            return [("experiments.scripts.ablation", force_arg)]
    else:
        if phase == "preflight":
            return [("experiments.scripts.stage_b.preflight", [])]
        if phase == "select":
            return [
                ("experiments.scripts.stage_b.select_gaussian", force_arg),
                (
                    "experiments.scripts.stage_b.select_fourier_ls",
                    [*force_arg, "--backend", fourier_backend],
                ),
                (
                    "experiments.scripts.stage_b.select_pinn",
                    [*force_arg, "--device", device, "--laplacian-backend", "autograd"],
                ),
            ]
        if phase == "lock":
            return [("experiments.scripts.stage_b.lock", force_arg)]
        if phase == "benchmark":
            return [("experiments.scripts.stage_b.formal_run", force_arg)]
    raise ValueError(f"unsupported Stage10-{stage.upper()} phase: {phase}")


def run_one(
    *,
    stage: str,
    mode: str,
    fold: int | None,
    phase: str,
    force: bool,
    device: str,
    fourier_backend: str,
    include_ablation: bool,
) -> None:
    if mode == "cv5":
        try:
            prepare_stage(stage, force=False)
        except FileExistsError:
            pass
    env = _env(mode, fold)
    phases = [phase]
    if phase == "all":
        phases = ["preflight", "select", "lock", "benchmark"]
        if stage == "a" and include_ablation:
            phases.append("ablation")
    for current in phases:
        for module, args in _phase_modules(
            stage,
            current,
            force=force,
            device=device,
            fourier_backend=fourier_backend,
        ):
            _run_module(module, args, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("a", "b"), required=True)
    parser.add_argument("--mode", choices=("standard", "cv5"), default="standard")
    parser.add_argument(
        "--fold",
        default=None,
        help="0..4 or 'all'; required for cv5, ignored for standard",
    )
    parser.add_argument(
        "--phase",
        choices=("preflight", "select", "lock", "benchmark", "ablation", "all"),
        default="preflight",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fourier-backend", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument(
        "--include-ablation",
        action="store_true",
        help="with Stage-A --phase all, also run E3 for each fold",
    )
    args = parser.parse_args()

    if args.stage == "b" and args.phase == "ablation":
        parser.error("Stage10-B has no formal ablation phase")

    if args.mode == "standard":
        run_one(
            stage=args.stage,
            mode="standard",
            fold=None,
            phase=args.phase,
            force=args.force,
            device=args.device,
            fourier_backend=args.fourier_backend,
            include_ablation=args.include_ablation,
        )
        return

    if args.fold is None:
        parser.error("--mode cv5 requires --fold 0..4 or --fold all")
    folds = range(5) if str(args.fold).lower() == "all" else (validate_fold(args.fold),)
    # Prepare all views once before invoking fold-specific subprocesses.
    try:
        prepare_stage(args.stage, force=False)
    except FileExistsError:
        pass
    for fold in folds:
        print(f"\n===== Stage10-{args.stage.upper()} CV5 fold {fold} =====", flush=True)
        run_one(
            stage=args.stage,
            mode="cv5",
            fold=fold,
            phase=args.phase,
            force=args.force,
            device=args.device,
            fourier_backend=args.fourier_backend,
            include_ablation=args.include_ablation,
        )


if __name__ == "__main__":
    main()
