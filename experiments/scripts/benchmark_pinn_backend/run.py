"""Validation-only PINN execution-backend benchmark; never touches test outputs."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from experiments.common.baselines import coordinate_pinn as pinn
from experiments.common.utils import experiment_utils as utils
from experiments.configs.formal_experiment import selection_dependency_config


def main(*, device_name="auto", steps=300, sample_index=0):
    rows = utils.load_manifest(utils.SYNTHETIC_MANIFEST, split="validation")
    if not rows:
        raise ValueError("validation split is empty; generate formal synthetic data first")
    sample_index = int(sample_index)
    if sample_index < 0 or sample_index >= len(rows):
        raise ValueError(f"sample_index must be in [0, {len(rows) - 1}]")
    sample = utils.load_synthetic_sample(rows[sample_index])
    formal = selection_dependency_config("pinn")
    config = dict(formal["training"])
    config["steps"] = int(steps)
    seed = int(formal["selection"]["seed"])
    device = _resolve_device(device_name)
    centers = pinn.initial_centers_from_observed(sample["observed"], config)

    print("PINN backend benchmark (validation only)")
    print(f"sample={sample['sample_id']} device={device} steps={steps} seed={seed}")
    if device.type == "cuda":
        print(
            f"gpu={torch.cuda.get_device_name(device)} "
            f"torch={torch.__version__} torch_cuda={torch.version.cuda}"
        )

    results = {}
    for backend in ("autograd", "analytic"):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = pinn.train_background(
            sample["observed"],
            config=config,
            device=device,
            seed=seed,
            show_progress=True,
            laplacian_backend=backend,
            initial_centers=centers,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall = time.perf_counter() - started
        results[backend] = result
        print(
            f"{backend}: wall={wall:.3f}s train_runtime={result['runtime_s']:.3f}s "
            f"total={result['total_loss']:.9g} data={result['data_loss']:.9g} "
            f"physics={result['physics_loss']:.9g} peak_vram_mb={result['peak_vram_mb']}"
        )

    ref = results["autograd"]
    fast = results["analytic"]
    diff = np.abs(
        np.asarray(ref["background_pred"], dtype=np.float64)
        - np.asarray(fast["background_pred"], dtype=np.float64)
    )
    speedup = float(ref["runtime_s"]) / max(float(fast["runtime_s"]), 1e-12)
    print(f"speedup_train_runtime={speedup:.3f}x")
    print(f"prediction_max_abs_diff={float(np.max(diff)):.9g}")
    print(f"prediction_mean_abs_diff={float(np.mean(diff)):.9g}")
    for key in ("total_loss", "data_loss", "physics_loss", "sparse_loss"):
        print(f"{key}_abs_diff={abs(float(ref[key]) - float(fast[key])):.9g}")


def _resolve_device(device_name):
    if device_name is None or str(device_name).strip().lower() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(str(device_name))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        index = 0 if device.index is None else int(device.index)
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(
                f"CUDA index {index} is out of range for {torch.cuda.device_count()} visible GPUs"
            )
        return torch.device(f"cuda:{index}")
    if device.type != "cpu":
        raise ValueError("device must be cpu or cuda:N")
    return device


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--sample-index", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(device_name=args.device, steps=args.steps, sample_index=args.sample_index)
