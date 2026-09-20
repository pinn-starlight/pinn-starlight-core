"""Compatibility checks between the completed Stage10-A and Stage10-B.

Important:
- Do NOT call Stage10-A's full Global-Lock verifier here.
- Stage10-B only needs to verify that the scientific method implementations and
  method-specific selection/training dependency configs still match Stage10-A.
- Unrelated later changes to the overall formal config must not invalidate B.
"""
from __future__ import annotations

from experiments.common.utils import experiment_utils as utils
from experiments.common.utils.stage_b_provenance import stable_json_sha256
from experiments.configs.formal_experiment import FORMAL_METHODS, selection_dependency_config

CORE_METHOD_SOURCE_FILES = (
    "experiments/common/baselines/fft_gaussian.py",
    "experiments/common/baselines/fft_screened_poisson.py",
    "experiments/common/baselines/unet_small.py",
    "experiments/common/baselines/coordinate_pinn.py",
    "experiments/common/baselines/decoupled_gradient_pinn.py",
    "src/pinn_starlight_core/physics/forcing.py",
    "src/pinn_starlight_core/nn/physics_model.py",
    "src/pinn_starlight_core/nn/pinn_layers.py",
    "src/pinn_starlight_core/nn/pinn_loss.py",
)


def load_stage_a_anchor():
    """Read A's frozen record without enforcing an irrelevant whole-config hash."""
    lock_path = utils.GLOBAL_CONFIGURATION_LOCK
    if not lock_path.is_file():
        raise FileNotFoundError(f"Stage10-A Global Lock not found: {lock_path}")

    lock = utils.read_json(lock_path)
    if lock.get("lock_state") != "FROZEN" or not lock.get("formal_test_ready"):
        raise RuntimeError("Stage10-A Global Lock is not in a completed FROZEN state")

    # 1) Verify method-specific selection/training dependency configs.
    artifacts = lock.get("selection_artifacts", {})
    problems = []
    for method in FORMAL_METHODS:
        expected = artifacts.get(method, {}).get("formal_config_sha256")
        if not expected:
            problems.append(f"{method}: A lock has no formal_config_sha256")
            continue
        current = stable_json_sha256(selection_dependency_config(method))
        if current != expected:
            problems.append(
                f"{method}: method-specific selection/training config differs from Stage10-A "
                f"(A={expected[:12]}..., current={current[:12]}...)"
            )

    # 2) Verify only scientific method implementation source files.
    frozen_records = {
        record["path"]: record["sha256"]
        for record in lock.get("source_provenance", {}).get("files", [])
    }
    for relative in CORE_METHOD_SOURCE_FILES:
        expected = frozen_records.get(relative)
        path = utils.PROJECT_ROOT / relative
        if expected is None:
            problems.append(f"{relative}: not recorded in A lock")
            continue
        if not path.is_file():
            problems.append(f"{relative}: current source file is missing")
            continue
        current = utils.file_sha256(path)
        if current != expected:
            problems.append(
                f"{relative}: core method source differs from Stage10-A "
                f"(A={expected[:12]}..., current={current[:12]}...)"
            )

    if problems:
        formatted = "\n - " + "\n - ".join(problems)
        raise ValueError(
            "Stage10-B compatibility check found a scientifically relevant A/B mismatch."
            + formatted
            + "\nDo not start Stage-B selection until this is reconciled."
        )

    return lock
