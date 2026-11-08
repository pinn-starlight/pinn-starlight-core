'''Stage11 clean experiment entry point with adaptive GPU task scheduling.

Scientific invariant
--------------------
Every formal scientific subprocess sees exactly one CUDA device through
CUDA_VISIBLE_DEVICES and therefore still executes on ``cuda:0`` internally.
No model is converted to DataParallel/DDP and no batch/optimizer/scientific
configuration is changed.

Scheduling policy
-----------------
- Stage-A standard: one GPU, sequential E1/lock/E2/E3 followed by Final-PINN.
- Stage-B stress test: one GPU, conservative sequential execution followed by Final-PINN.
- Grouped CV5 evaluation:
    * one GPU  -> five folds sequentially;
    * N GPUs   -> up to N independent fold workers in parallel.
      A given GPU never runs two folds at once.
- E4 real-image experiment: one GPU, five frozen methods, no cross-validation.
- ``all`` runs A -> B -> CV5 -> E4.

This keeps the formal single-visible-GPU execution policy intact while using
multiple physical GPUs for independent experimental tasks.
''' 
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import platform
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import threading
from typing import Iterable


ROOT = Path(__file__).resolve().parent
LOG_ROOT = ROOT / "logs"
_PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class GPUInfo:
    index: int
    uuid: str
    name: str
    memory_total_mib: int | None
    memory_used_mib: int | None
    utilization_percent: int | None
    driver_version: str

    @property
    def cuda_visible_token(self) -> str:
        return self.uuid if self.uuid.startswith("GPU-") else str(self.index)

    @property
    def label(self) -> str:
        mem = (
            f"{self.memory_used_mib}/{self.memory_total_mib} MiB"
            if self.memory_used_mib is not None and self.memory_total_mib is not None
            else "memory n/a"
        )
        util = f"{self.utilization_percent}%" if self.utilization_percent is not None else "n/a"
        return f"GPU {self.index}: {self.name} ({mem}, util={util}, driver={self.driver_version})"


def _int_or_none(value: str) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _query_nvidia_smi() -> list[GPUInfo]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "nvidia-smi was not found. Stage11 formal GPU runs require an NVIDIA GPU/driver."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "nvidia-smi failed before Stage11 started:\n"
            + (exc.stderr or exc.stdout or "<no output>")
        ) from exc

    rows: list[GPUInfo] = []
    for raw in proc.stdout.splitlines():
        if not raw.strip():
            continue
        fields = next(csv.reader([raw], skipinitialspace=True))
        if len(fields) != 7:
            raise RuntimeError(f"Unexpected nvidia-smi row: {raw!r}")
        index, uuid, name, total, used, util, driver = [x.strip() for x in fields]
        rows.append(
            GPUInfo(
                index=int(index),
                uuid=uuid,
                name=name,
                memory_total_mib=_int_or_none(total),
                memory_used_mib=_int_or_none(used),
                utilization_percent=_int_or_none(util),
                driver_version=driver,
            )
        )
    if not rows:
        raise RuntimeError("nvidia-smi reported zero GPUs.")
    return rows


def _filter_parent_visible(all_gpus: list[GPUInfo]) -> list[GPUInfo]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        return list(all_gpus)

    tokens = [x.strip() for x in raw.split(",") if x.strip()]
    if not tokens or tokens == ["-1"]:
        return []

    by_index = {str(g.index): g for g in all_gpus}
    selected: list[GPUInfo] = []
    for token in tokens:
        gpu = by_index.get(token)
        if gpu is None:
            matches = [
                g
                for g in all_gpus
                if g.uuid == token or g.uuid.startswith(token) or token.startswith(g.uuid)
            ]
            if len(matches) == 1:
                gpu = matches[0]
        if gpu is None:
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES contains {token!r}, but it could not be matched "
                "to nvidia-smi physical indices/UUIDs."
            )
        if gpu not in selected:
            selected.append(gpu)
    return selected


def _select_gpus(spec: str, all_gpus: list[GPUInfo]) -> list[GPUInfo]:
    visible = _filter_parent_visible(all_gpus)
    if not visible:
        raise RuntimeError("No NVIDIA GPUs remain visible to Stage11.")

    spec = str(spec).strip()
    if not spec or spec.lower() in {"auto", "all"}:
        return visible

    requested = [x.strip() for x in spec.split(",") if x.strip()]
    selected: list[GPUInfo] = []
    for token in requested:
        matches = [
            g
            for g in visible
            if str(g.index) == token or g.uuid == token or g.uuid.startswith(token)
        ]
        if len(matches) != 1:
            available = ", ".join(f"{g.index}:{g.name}" for g in visible)
            raise ValueError(
                f"--gpus token {token!r} did not uniquely match a visible GPU. "
                f"Visible: {available}"
            )
        if matches[0] not in selected:
            selected.append(matches[0])
    if not selected:
        raise ValueError("--gpus selected no devices.")
    return selected


def _is_blackwell_name(name: str) -> bool:
    import re

    patterns = (
        r"\bRTX\s*(?:PRO\s*)?50\d{2}\b",
        r"\bGeForce\s+RTX\s+50\d{2}\b",
        r"\bBlackwell\b",
        r"\bB100\b",
        r"\bB200\b",
        r"\bGB200\b",
        r"\bGB300\b",
    )
    return any(re.search(pattern, str(name), flags=re.IGNORECASE) for pattern in patterns)


def _driver_major(version: str) -> int | None:
    try:
        return int(str(version).split(".", 1)[0])
    except (TypeError, ValueError):
        return None


def _validate_runtime_profile(gpus: list[GPUInfo]) -> dict:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch could not be imported in the selected Pixi environment.") from exc

    runtime = str(torch.version.cuda or "")
    torch_version = str(torch.__version__)
    profile = os.environ.get("STAGE11_CUDA_PROFILE", "unknown")
    environment = os.environ.get("STAGE11_PIXI_ENVIRONMENT", "unknown")

    blackwell = any(_is_blackwell_name(g.name) for g in gpus)
    majors = [m for m in (_driver_major(g.driver_version) for g in gpus) if m is not None]
    minimum_driver_major = min(majors) if majors else None

    if blackwell:
        try:
            runtime_parts = tuple(int(x) for x in runtime.split(".")[:2])
        except ValueError:
            runtime_parts = ()
        if runtime_parts and runtime_parts < (12, 8):
            raise RuntimeError(
                "A Blackwell/RTX-50 GPU was detected, but the active PyTorch CUDA "
                f"runtime is {runtime!r}. Start Stage11 with the launcher so it selects "
                "the remote-blackwell profile."
            )

    if runtime.startswith("13.") and minimum_driver_major is not None and minimum_driver_major < 580:
        raise RuntimeError(
            f"PyTorch reports CUDA runtime {runtime}, but the oldest visible NVIDIA "
            f"driver is {minimum_driver_major}xx. CUDA 13.x needs an R580+ driver. "
            "Use the remote/cu126 profile on non-Blackwell GPUs, or upgrade the "
            "driver for Blackwell."
        )

    return {
        "torch_version": torch_version,
        "torch_cuda_runtime": runtime,
        "stage11_cuda_profile": profile,
        "stage11_pixi_environment": environment,
        "blackwell_detected": blackwell,
        "minimum_driver_major": minimum_driver_major,
    }


def _single_gpu_env(
    gpu: GPUInfo,
    *,
    fold: int | None = None,
    scheduler_mode: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env["CUDA_VISIBLE_DEVICES"] = gpu.cuda_visible_token
    env["STAGE11_ASSIGNED_GPU_PHYSICAL_INDEX"] = str(gpu.index)
    env["STAGE11_ASSIGNED_GPU_UUID"] = gpu.uuid
    env["STAGE11_ASSIGNED_GPU_NAME"] = gpu.name
    env["STAGE11_GPU_SCHEDULER_MODE"] = scheduler_mode
    env["PYTHONUTF8"] = "1"
    if fold is not None:
        env["PINN_STARLIGHT_SPLIT_MODE"] = "cv5"
        env["PINN_STARLIGHT_CV_FOLD"] = str(int(fold))
    return env


def _gpu_health_probe(gpu: GPUInfo) -> dict:
    code = r'''
import json
import torch

if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() is False")
if torch.cuda.device_count() != 1:
    raise RuntimeError(f"expected exactly one visible GPU, got {torch.cuda.device_count()}")

x = torch.ones((256, 256), dtype=torch.float32, device="cuda:0")
y = x @ x
torch.cuda.synchronize()

print(json.dumps({
    "torch": str(torch.__version__),
    "cuda_runtime": str(torch.version.cuda),
    "gpu": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)),
    "visible_device_count": torch.cuda.device_count(),
    "probe_value": float(y[0, 0].item()),
}))
'''
    env = _single_gpu_env(gpu, scheduler_mode="health_probe")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"CUDA health probe failed on physical GPU {gpu.index} ({gpu.name}).\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    try:
        result = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not parse CUDA health probe output for GPU {gpu.index}: {proc.stdout!r}"
        ) from exc
    result["physical_index"] = gpu.index
    result["physical_uuid"] = gpu.uuid
    result["driver_version"] = gpu.driver_version
    return result


def _run(
    command: Iterable[object],
    *,
    env: dict[str, str] | None = None,
    prefix: str | None = None,
    log_path: Path | None = None,
) -> None:
    command = [str(x) for x in command]
    printable = " ".join(command)

    if prefix is None and log_path is None:
        print("\n>", printable, flush=True)
        subprocess.run(
            command,
            cwd=ROOT,
            env=(os.environ.copy() if env is None else env),
            check=True,
        )
        return

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    with _PRINT_LOCK:
        print(f"\n{prefix or ''}> {printable}", flush=True)
        if log_path is not None:
            print(f"{prefix or ''}log: {log_path}", flush=True)

    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        env=(os.environ.copy() if env is None else env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    log_file = (
        log_path.open("a", encoding="utf-8", buffering=1)
        if log_path is not None
        else None
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if log_file is not None:
                log_file.write(line)
            with _PRINT_LOCK:
                print(f"{prefix or ''}{line}", end="", flush=True)
    finally:
        if log_file is not None:
            log_file.close()

    return_code = proc.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _module(
    module: str,
    *args: str,
    env: dict[str, str] | None = None,
    prefix: str | None = None,
    log_path: Path | None = None,
) -> None:
    _run(
        [sys.executable, "-u", "-m", module, *args],
        env=env,
        prefix=prefix,
        log_path=log_path,
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _scheduler_record_path(command: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_ROOT / f"stage11_scheduler_{command}_{stamp}.json"


def _assignment_plan(command: str, gpus: list[GPUInfo], max_parallel: int) -> dict:
    primary = gpus[0]
    worker_count = min(len(gpus), 5)
    if max_parallel > 0:
        worker_count = min(worker_count, max_parallel)
    worker_count = max(1, worker_count)
    cv_gpus = gpus[:worker_count]

    fold_groups = {
        str(gpu.index): [fold for fold in range(5) if fold % worker_count == worker_idx]
        for worker_idx, gpu in enumerate(cv_gpus)
    }

    return {
        "command": command,
        "standard_stage_gpu": {
            "physical_index": primary.index,
            "name": primary.name,
            "uuid": primary.uuid,
        },
        "cv_worker_count": worker_count,
        "cv_fold_groups_by_physical_gpu": fold_groups,
        "policy": {
            "standard_a_b_e4": "single_gpu_sequential",
            "cv5": "independent_fold_task_parallelism",
            "model_parallelism": False,
            "ddp": False,
            "data_parallel": False,
            "scientific_subprocess_visible_gpu_count": 1,
            "scientific_internal_device": "cuda:0",
        },
    }


def _print_plan(gpus: list[GPUInfo], plan: dict, runtime: dict) -> None:
    print("\n=== Stage11 adaptive GPU scheduler ===")
    print(f"Pixi env      : {runtime.get('stage11_pixi_environment')}")
    print(f"CUDA profile  : {runtime.get('stage11_cuda_profile')}")
    print(f"PyTorch       : {runtime.get('torch_version')}")
    print(f"Torch CUDA    : {runtime.get('torch_cuda_runtime')}")
    print(f"Selected GPUs : {len(gpus)}")
    for gpu in gpus:
        print("  ", gpu.label)
    print(
        "Standard A/B/E4: physical GPU "
        f"{plan['standard_stage_gpu']['physical_index']} "
        "(child process sees it as cuda:0)"
    )
    print(f"CV5 workers   : {plan['cv_worker_count']}")
    for physical, folds in plan["cv_fold_groups_by_physical_gpu"].items():
        print(f"  physical GPU {physical} <- folds {folds}")
    print("Model parallel: OFF (intentional)")
    print("DDP/DataParallel: OFF (intentional)")
    print()



STAGE_A_DATA_DIR = ROOT / "collections" / "synthetic_formal"
STAGE_A_MANIFEST = STAGE_A_DATA_DIR / "manifest.csv"
STAGE_B_DATA_DIR = ROOT / "collections" / "synthetic_stage_b"
STAGE_B_MANIFEST = STAGE_B_DATA_DIR / "manifest.csv"


def _validate_manifest(label: str, manifest: Path) -> None:
    if not manifest.is_file():
        raise RuntimeError(f"{label} manifest was not produced: {manifest}")
    if manifest.stat().st_size <= 0:
        raise RuntimeError(f"{label} manifest is empty: {manifest}")

    with manifest.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        raise RuntimeError(
            f"{label} manifest has no data rows (expected header + data): {manifest}"
        )


def _require_module(module: str, *, label: str) -> str:
    module = str(module).strip()
    if not module:
        raise RuntimeError(f"{label} data-generator module is empty.")
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, AttributeError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            f"Could not resolve {label} data-generator module {module!r}."
        ) from exc
    if spec is None:
        raise RuntimeError(
            f"{label} data-generator module {module!r} was not found. "
            f"Set --{label.lower().replace('-', '-')}-data-module to the actual module."
        )
    return module


def _archive_dataset_dir(dataset_dir: Path, *, label: str) -> Path | None:
    if not dataset_dir.exists():
        return None

    archive_root = ROOT / "collections" / "_stage11_data_archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = archive_root / f"{dataset_dir.name}_{stamp}"

    suffix = 1
    while target.exists():
        target = archive_root / f"{dataset_dir.name}_{stamp}_{suffix}"
        suffix += 1

    print(
        f"{label}: archiving existing dataset directory before regeneration:\n"
        f"  {dataset_dir.relative_to(ROOT)} -> {target.relative_to(ROOT)}",
        flush=True,
    )
    shutil.move(str(dataset_dir), str(target))
    return target


def ensure_dataset(
    *,
    label: str,
    dataset_dir: Path,
    manifest: Path,
    generator_module: str,
    force_data: bool,
) -> None:
    """Ensure a formal dataset exists before any dependent experiment starts.

    Normal behavior:
      - valid existing manifest -> reuse the frozen dataset;
      - missing manifest -> archive any partial directory, then generate it.

    With --force-data:
      - archive the existing dataset directory;
      - regenerate from the declared generator module.

    Training --force and data --force-data are deliberately separate scientific
    controls. A training rerun must not silently change the formal dataset.
    """
    if manifest.is_file() and not force_data:
        _validate_manifest(label, manifest)
        print(
            f"{label}: using existing frozen dataset: {manifest.relative_to(ROOT)}",
            flush=True,
        )
        return

    generator_module = _require_module(generator_module, label=label)

    # A directory without a valid manifest is treated as partial/stale and is
    # never mixed with a newly generated formal dataset.
    if dataset_dir.exists():
        _archive_dataset_dir(dataset_dir, label=label)

    print(f"{label}: generating formal dataset via {generator_module}", flush=True)
    _module(generator_module)
    _validate_manifest(label, manifest)
    print(f"{label}: dataset ready: {manifest.relative_to(ROOT)}", flush=True)

def run_a(*, force: bool, gpu: GPUInfo) -> None:
    env = _single_gpu_env(gpu, scheduler_mode="stage_a_standard_single_gpu")
    args = [
        "--stage", "a",
        "--mode", "standard",
        "--phase", "all",
        "--device", "cuda:0",
        "--fourier-backend", "cuda",
    ]
    if force:
        args.append("--force")
    # E3 is part of the frozen Stage-A standard experiment, not an optional add-on.
    args.append("--include-ablation")
    _module("experiments.scripts.protocol.run", *args, env=env)

    follow = ["--force"] if force else []
    _module("experiments.scripts.final_pinn.stage_a", *follow, env=env)


def run_e4(*, force: bool, gpu: GPUInfo) -> None:
    """Run the independent five-method real-image E4 experiment."""
    env = _single_gpu_env(gpu, scheduler_mode="e4_real_single_gpu")
    args = ["--force"] if force else []
    _module("experiments.scripts.real_images.run", *args, env=env)


def run_b(*, force: bool, gpu: GPUInfo) -> None:
    env = _single_gpu_env(gpu, scheduler_mode="stage_b_standard_single_gpu")
    args = [
        "--stage", "b",
        "--mode", "standard",
        "--phase", "all",
        "--device", "cuda:0",
        "--fourier-backend", "cuda",
    ]
    if force:
        args.append("--force")
    _module("experiments.scripts.protocol.run", *args, env=env)

    follow = ["--force"] if force else []
    _module("experiments.scripts.final_pinn.stage_b", *follow, env=env)


def _run_one_cv_fold(*, fold: int, gpu: GPUInfo, force: bool) -> dict:
    env = _single_gpu_env(
        gpu,
        fold=fold,
        scheduler_mode="cv5_fold_task_parallelism",
    )
    prefix = f"[fold {fold} | physical GPU {gpu.index}] "
    log_path = LOG_ROOT / f"cv5_fold_{fold}_gpu_{gpu.index}.log"

    status_path = LOG_ROOT / f"cv5_fold_{fold}_status.json"
    status = {
        "fold": fold,
        "physical_gpu_index": gpu.index,
        "physical_gpu_uuid": gpu.uuid,
        "gpu_name": gpu.name,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "log": str(log_path.relative_to(ROOT)),
    }
    _write_json(status_path, status)

    try:
        args = [
            "--stage", "a",
            "--mode", "cv5",
            "--fold", str(fold),
            "--phase", "all",
            "--device", "cuda:0",
            "--fourier-backend", "cuda",
        ]
        if force:
            args.append("--force")
        _module(
            "experiments.scripts.protocol.run",
            *args,
            env=env,
            prefix=prefix,
            log_path=log_path,
        )

        follow = ["--force"] if force else []
        _module(
            "experiments.scripts.final_pinn.stage_a",
            *follow,
            env=env,
            prefix=prefix,
            log_path=log_path,
        )

        status["status"] = "complete"
        status["finished_at"] = datetime.now().astimezone().isoformat()
        _write_json(status_path, status)
        with _PRINT_LOCK:
            print(f"\n{prefix}COMPLETE", flush=True)
        return status
    except Exception as exc:
        status["status"] = "failed"
        status["finished_at"] = datetime.now().astimezone().isoformat()
        status["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(status_path, status)
        with _PRINT_LOCK:
            print(f"\n{prefix}FAILED: {exc}", flush=True)
        raise


def _cv_worker(*, gpu: GPUInfo, folds: list[int], force: bool) -> list[dict]:
    results = []
    for fold in folds:
        results.append(_run_one_cv_fold(fold=fold, gpu=gpu, force=force))
    return results


def run_cv5(*, force: bool, gpus: list[GPUInfo], max_parallel: int) -> None:
    prep = ["--stage", "a"]
    if force:
        prep.append("--force")
    _module("experiments.scripts.protocol.prepare_cv5", *prep)

    worker_count = min(len(gpus), 5)
    if max_parallel > 0:
        worker_count = min(worker_count, max_parallel)
    worker_count = max(1, worker_count)
    worker_gpus = gpus[:worker_count]

    assignments = [
        (gpu, [fold for fold in range(5) if fold % worker_count == worker_idx])
        for worker_idx, gpu in enumerate(worker_gpus)
    ]

    if worker_count == 1:
        gpu, folds = assignments[0]
        _cv_worker(gpu=gpu, folds=folds, force=force)
    else:
        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="stage11-gpu") as pool:
            futures = {
                pool.submit(_cv_worker, gpu=gpu, folds=folds, force=force): (gpu, folds)
                for gpu, folds in assignments
            }
            for future in as_completed(futures):
                gpu, folds = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failures.append(
                        f"physical GPU {gpu.index}, folds {folds}: {type(exc).__name__}: {exc}"
                    )
        if failures:
            raise RuntimeError(
                "One or more CV5 GPU workers failed. Completed folds were preserved.\n - "
                + "\n - ".join(failures)
            )

    _module("experiments.scripts.protocol.aggregate_cv5", "--stage", "a")
    _module("experiments.scripts.final_pinn.aggregate_cv5")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("a", "b", "cv5", "e4", "all"),
        help=(
            "a=Stage-A standard experiment (including E3 and Final-PINN); "
            "b=Stage-B stress test; cv5=five-fold grouped evaluation; "
            "e4=independent five-method real-image experiment; "
            "all=complete paper workflow (A -> B -> CV5 -> E4)."
        ),
    )
    parser.add_argument(
        "--gpus",
        default="auto",
        help=(
            "GPU pool: 'auto'/'all' or physical nvidia-smi indices/UUIDs, "
            "e.g. --gpus 0,1. Each scientific child still sees one GPU as cuda:0."
        ),
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=0,
        help="Cap CV5 concurrent GPU workers; 0 means auto (= min(selected GPUs, 5)).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run training/evaluation stages. Does NOT regenerate formal datasets.",
    )
    parser.add_argument(
        "--force-data",
        action="store_true",
        help=(
            "Regenerate required formal datasets after archiving the existing dataset "
            "directories. Requires --force because model outputs must then be recomputed."
        ),
    )
    parser.add_argument(
        "--stage-a-data-module",
        default=os.environ.get(
            "STAGE11_STAGE_A_DATA_MODULE",
            "experiments.common.data.generate_synthetic",
        ),
        help=(
            "Python module that generates collections/synthetic_formal. "
            "Default: experiments.common.data.generate_synthetic"
        ),
    )
    parser.add_argument(
        "--stage-b-data-module",
        default=os.environ.get(
            "STAGE11_STAGE_B_DATA_MODULE",
            "experiments.common.data.generate_stage_b",
        ),
        help=(
            "Python module that generates collections/synthetic_stage_b. "
            "Default: experiments.common.data.generate_stage_b"
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Probe GPUs and print the schedule without opening datasets or running experiments.",
    )
    parser.add_argument(
        "--skip-gpu-probe",
        action="store_true",
        help="Skip the tiny per-GPU PyTorch matmul health probe (not recommended).",
    )
    parser.add_argument("--device", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.max_parallel < 0:
        parser.error("--max-parallel must be >= 0")
    if args.force_data and not args.force:
        parser.error(
            "--force-data changes the formal dataset and therefore requires --force "
            "so all dependent training/evaluation outputs are recomputed."
        )

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    all_gpus = _query_nvidia_smi()
    gpu_spec = args.gpus
    if args.device and str(args.gpus).lower() == "auto":
        device = str(args.device)
        if device.startswith("cuda:"):
            gpu_spec = device.split(":", 1)[1]
            print(
                f"NOTE: legacy --device {device} interpreted as --gpus {gpu_spec}; "
                "scientific subprocesses still use cuda:0 after isolation.",
                flush=True,
            )
        else:
            parser.error("legacy --device only supports cuda:N")

    gpus = _select_gpus(gpu_spec, all_gpus)
    runtime = _validate_runtime_profile(gpus)

    probes = []
    if not args.skip_gpu_probe:
        print("Running isolated CUDA health probes...", flush=True)
        for gpu in gpus:
            probe = _gpu_health_probe(gpu)
            probes.append(probe)
            print(
                f"  PASS physical GPU {gpu.index}: {probe['gpu']} "
                f"(torch={probe['torch']}, runtime={probe['cuda_runtime']}, "
                f"capability={tuple(probe['capability'])})",
                flush=True,
            )

    plan = _assignment_plan(args.command, gpus, args.max_parallel)
    _print_plan(gpus, plan, runtime)

    record = {
        "created_at": datetime.now().astimezone().isoformat(),
        "host_platform": platform.platform(),
        "python": sys.version,
        "command": args.command,
        "gpu_spec": gpu_spec,
        "max_parallel": args.max_parallel,
        "force_training": bool(args.force),
        "force_data": bool(args.force_data),
        "stage_a_data_module": args.stage_a_data_module,
        "stage_b_data_module": args.stage_b_data_module,
        "parent_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selected_gpus": [asdict(g) for g in gpus],
        "runtime": runtime,
        "health_probes": probes,
        "plan": plan,
    }
    record_path = _scheduler_record_path(args.command)
    _write_json(record_path, record)
    print("Scheduler record:", record_path.relative_to(ROOT), flush=True)

    if args.plan_only:
        print("Plan-only mode: no experiment data were opened and no experiment was run.")
        return

    primary = gpus[0]

    if args.command in ("a", "all"):
        ensure_dataset(
            label="Stage-A",
            dataset_dir=STAGE_A_DATA_DIR,
            manifest=STAGE_A_MANIFEST,
            generator_module=args.stage_a_data_module,
            force_data=args.force_data,
        )
        run_a(force=args.force, gpu=primary)

    if args.command in ("b", "all"):
        ensure_dataset(
            label="Stage-B",
            dataset_dir=STAGE_B_DATA_DIR,
            manifest=STAGE_B_MANIFEST,
            generator_module=args.stage_b_data_module,
            force_data=args.force_data,
        )
        run_b(force=args.force, gpu=primary)

    if args.command in ("cv5", "all"):
        # CV5 is an independent evaluation protocol that reuses the frozen
        # Stage-A semi-synthetic dataset construction, not a Stage-A sub-experiment.
        ensure_dataset(
            label="Stage-A",
            dataset_dir=STAGE_A_DATA_DIR,
            manifest=STAGE_A_MANIFEST,
            generator_module=args.stage_a_data_module,
            force_data=False,
        )
        run_cv5(force=args.force, gpus=gpus, max_parallel=args.max_parallel)

    if args.command in ("e4", "all"):
        run_e4(force=args.force, gpu=primary)

    record["completed_at"] = datetime.now().astimezone().isoformat()
    record["status"] = "complete"
    _write_json(record_path, record)


if __name__ == "__main__":
    main()
