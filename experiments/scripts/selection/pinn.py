"""E1: select the formal PINN configuration on validation only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.common.baselines import coordinate_pinn as pinn
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import metrics
from experiments.common.utils import selection_artifacts as artifacts
from experiments.configs.formal_experiment import selection_dependency_config

MANIFEST = utils.SYNTHETIC_MANIFEST
OUTPUT_ROOT = utils.PINN_SELECTION_ROOT
ARTIFACT_PATH = utils.PINN_SELECTION_ARTIFACT


def run_candidate(
    config: dict,
    validation_manifest: list[dict],
    output_dir: Path,
    seed: int,
    device,
    resume_states=None,
    return_states=False,
    laplacian_backend=None,
    initial_centers_by_sample=None,
    evaluator=None,
):
    """Run one PINN candidate on the fixed validation split.

    ``evaluator`` defaults to ``metrics.evaluate_synthetic``.
    """
    if evaluator is None:
        evaluator = metrics.evaluate_synthetic
    output_dir.mkdir(parents=True, exist_ok=True)
    utils.save_run_metadata(
        output_dir,
        {
            "experiment": "E1 PINN selection",
            "seed": seed,
            "pinn_config": config,
            "evaluation_schema_version": metrics.EVALUATION_SCHEMA_VERSION,
            "star_detection_rule": metrics.DETECTION_IMAGE_RULE,
            "flux_metric": metrics.MATCHED_STAR_FLUX_METRIC,
            "compute_device": str(device),
            "pinn_laplacian_backend": pinn.resolve_laplacian_backend(laplacian_backend),
        },
    )
    rows = []
    states = {}
    for manifest_row in validation_manifest:
        sample = utils.load_synthetic_sample(manifest_row)
        sample_id = str(sample["sample_id"])
        result = pinn.train_background(
            sample["observed"],
            config=config,
            device=device,
            seed=seed,
            resume_state=(resume_states or {}).get(sample_id),
            return_state=return_states,
            show_progress=True,
            laplacian_backend=laplacian_backend,
            initial_centers=(initial_centers_by_sample or {}).get(sample_id),
        )
        prediction = {
            "background_pred": result["background_pred"],
            "residual_pred": utils.reconstruct_residual(
                sample["observed"], result["background_pred"]
            ),
        }
        sample_output = output_dir / sample_id
        utils.save_prediction_arrays(
            sample_output,
            sample["observed"],
            prediction["background_pred"],
            prediction["residual_pred"],
        )
        utils.save_rows_csv(sample_output / "history.csv", result["history"])
        utils.write_json(sample_output / "final_parameters.json", _final_parameters(result))

        scores = evaluator(sample, prediction)
        rows.append(
            {
                "sample_id": sample_id,
                "background_type": sample["background_type"],
                "intensity_level": sample["intensity_level"],
                "seed": seed,
                "step": result["step"],
                "runtime_s": result["runtime_s"],
                "peak_vram_mb": result["peak_vram_mb"],
                "parameter_count": result["parameter_count"],
                **scores,
            }
        )
        if return_states:
            states[sample_id] = result["state"]

    utils.save_rows_csv(output_dir / "metrics.csv", rows)
    aggregate = _aggregate(rows)
    utils.write_json(output_dir / "summary.json", aggregate)
    return {"rows": rows, "aggregate": aggregate, "states": states}


def main(*, force: bool = False, device_name: str | None = None, laplacian_backend=None):
    print("E1 PINN validation selection")
    output_root = utils.prepare_output_root(OUTPUT_ROOT, force=force)
    validation_rows = utils.load_manifest(MANIFEST, split="validation")
    if not validation_rows:
        raise ValueError("validation split is empty; generate formal synthetic data first")

    formal_config = selection_dependency_config("pinn")
    selection = formal_config["selection"]
    base_config = dict(formal_config["training"])
    seed = int(selection["seed"])
    device = _resolve_device(device_name)
    laplacian_backend = pinn.resolve_laplacian_backend(laplacian_backend)
    utils.set_seed(seed)
    utils.save_run_metadata(
        output_root,
        {
            "experiment": "E1 PINN selection",
            "manifest": utils.project_relative(MANIFEST),
            "formal_config": formal_config,
            "force_overwrite_output": force,
            "validation_samples": [row["sample_id"] for row in validation_rows],
            "compute_device": str(device),
            "pinn_laplacian_backend": laplacian_backend,
        },
    )

    initial_centers_by_sample = _precompute_initial_centers(
        validation_rows,
        base_config,
    )

    candidate_table = []
    selected_config, _ = _select_stage(
        "T1_network",
        base_config,
        [{"hidden_dims": list(candidate)} for candidate in selection["network_candidates"]],
        validation_rows,
        output_root,
        seed,
        device,
        candidate_table,
        keep_states=False,
        laplacian_backend=laplacian_backend,
        initial_centers_by_sample=initial_centers_by_sample,
    )
    selected_config, _ = _select_stage(
        "T2_physics_weight",
        selected_config,
        [{"physics_weight": value} for value in selection["physics_weight_candidates"]],
        validation_rows,
        output_root,
        seed,
        device,
        candidate_table,
        keep_states=False,
        laplacian_backend=laplacian_backend,
        initial_centers_by_sample=initial_centers_by_sample,
    )
    selected_config, selected_result = _select_stage(
        "T3_sparse_weight",
        selected_config,
        [{"sparse_weight": value} for value in selection["sparse_weight_candidates"]],
        validation_rows,
        output_root,
        seed,
        device,
        candidate_table,
        keep_states=True,
        laplacian_backend=laplacian_backend,
        initial_centers_by_sample=initial_centers_by_sample,
    )
    selected_config, selected_result = _continue_until_stable(
        selected_config,
        selected_result,
        validation_rows,
        output_root,
        device,
        candidate_table,
        selection,
        seed,
        laplacian_backend,
        initial_centers_by_sample,
    )

    selected_config = dict(selected_config)
    selected_config["steps"] = int(max(row["step"] for row in selected_result["rows"]))
    selected_config = dict(pinn.normalized_config(selected_config))
    selected = {
        "config": selected_config,
        "selection_seed": seed,
        "selection_metric": selection["selection_metric"],
        "selection_score": selected_result["aggregate"],
    }
    artifact = artifacts.build_selection_artifact(
        method="pinn",
        formal_config=formal_config,
        selected=selected,
        manifest=MANIFEST,
        rows_by_split={"validation": validation_rows},
    )
    utils.save_rows_csv(output_root / "candidate_results.csv", candidate_table)
    utils.write_json(ARTIFACT_PATH, artifact)
    print(f"E1 PINN selection complete: {ARTIFACT_PATH}")
    print(
        "Selected:",
        f"hidden={selected_config['hidden_dims']}",
        f"physics_weight={selected_config['physics_weight']}",
        f"sparse_weight={selected_config['sparse_weight']}",
        f"steps={selected_config['steps']}",
    )


def _select_stage(
    stage,
    base_config,
    updates,
    validation_rows,
    output_root,
    seed,
    device,
    candidate_table,
    keep_states,
    laplacian_backend,
    initial_centers_by_sample,
    evaluator=None,
):
    best = None
    for index, update in enumerate(updates, start=1):
        config = dict(base_config)
        config.update(update)
        name = _candidate_name(index, update)
        result = run_candidate(
            config,
            validation_rows,
            output_root / stage / name,
            seed,
            device,
            return_states=keep_states,
            laplacian_backend=laplacian_backend,
            initial_centers_by_sample=initial_centers_by_sample,
            evaluator=evaluator,
        )
        row = {
            "stage": stage,
            "candidate": name,
            "config": config,
            "compute_device": str(device),
            "pinn_laplacian_backend": laplacian_backend,
            **result["aggregate"],
        }
        candidate_table.append(row)
        candidate = (config, result)
        if best is None or _selection_key(result["aggregate"]) < _selection_key(best[1]["aggregate"]):
            best = candidate
    if best is None:
        raise ValueError(f"{stage} has no candidate")
    return best


def _continue_until_stable(
    config,
    current_result,
    validation_rows,
    output_root,
    device,
    candidate_table,
    selection,
    seed,
    laplacian_backend,
    initial_centers_by_sample,
    evaluator=None,
):
    total_steps = int(config["steps"])
    previous_mae = current_result["aggregate"]["bg_mae"]
    states = current_result["states"]
    best_result = current_result
    best_steps = total_steps
    best_key = _selection_key(current_result["aggregate"])
    stable_rounds = 0
    max_steps = int(selection["max_steps"])
    continue_steps = int(selection["continue_steps"])
    tolerance = float(selection["stability_tolerance"])
    patience = int(selection["stability_patience"])

    while total_steps < max_steps:
        additional_steps = min(continue_steps, max_steps - total_steps)
        continuation_config = dict(config)
        continuation_config["steps"] = additional_steps
        next_total = total_steps + additional_steps
        continued = run_candidate(
            continuation_config,
            validation_rows,
            output_root / "T4_steps" / str(next_total),
            seed,
            device,
            resume_states=states,
            return_states=True,
            laplacian_backend=laplacian_backend,
            initial_centers_by_sample=initial_centers_by_sample,
            evaluator=evaluator,
        )
        current_mae = continued["aggregate"]["bg_mae"]
        relative_improvement = (previous_mae - current_mae) / max(previous_mae, 1e-12)
        candidate_table.append(
            {
                "stage": "T4_steps",
                "candidate": str(next_total),
                "config": {**config, "steps": next_total},
                "relative_bg_mae_improvement": relative_improvement,
                "compute_device": str(device),
                "pinn_laplacian_backend": laplacian_backend,
                **continued["aggregate"],
            }
        )
        total_steps = next_total
        current_result = continued
        states = continued["states"]
        previous_mae = current_mae
        current_key = _selection_key(current_result["aggregate"])
        if current_key < best_key:
            best_result = current_result
            best_steps = total_steps
            best_key = current_key
        if relative_improvement < tolerance:
            stable_rounds += 1
        else:
            stable_rounds = 0
        if stable_rounds >= patience:
            break

    selected_config = dict(config)
    selected_config["steps"] = best_steps
    return selected_config, best_result



def _precompute_initial_centers(validation_rows, config):
    """Cache observed-only initialization shared by all E1 candidates.

    E1 repeatedly refits the same validation images while changing only network
    width, loss weights, and continuation steps.  The formal center initializer
    is therefore invariant across candidates and can be computed once per image.
    """
    cache = {}
    for manifest_row in validation_rows:
        sample = utils.load_synthetic_sample(manifest_row)
        sample_id = str(sample["sample_id"])
        cache[sample_id] = pinn.initial_centers_from_observed(
            sample["observed"],
            config,
        )
    return cache

def _aggregate(rows):
    flux_values = _finite_values(rows, "matched_star_mean_relative_flux_error")
    return {
        "bg_mae": float(np.mean([row["bg_mae"] for row in rows])),
        "residual_ssim": float(np.mean([row["residual_ssim"] for row in rows])),
        "star_f1": float(np.mean([row["star_f1"] for row in rows])),
        "matched_star_mean_relative_flux_error": (
            float(np.mean(flux_values)) if flux_values else float("nan")
        ),
        "matched_star_flux_valid_count": len(flux_values),
        "runtime_s": float(np.sum([row["runtime_s"] for row in rows])),
        "parameter_count": float(np.mean([row["parameter_count"] for row in rows])),
    }


def _finite_values(rows, key):
    return [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float, np.number)) and np.isfinite(row[key])
    ]


def _selection_key(aggregate):
    return (
        aggregate["bg_mae"],
        -aggregate["residual_ssim"],
        _finite_or_inf(aggregate["matched_star_mean_relative_flux_error"]),
        aggregate["parameter_count"],
    )


def _finite_or_inf(value):
    value = float(value)
    return value if np.isfinite(value) else float("inf")


def _candidate_name(index, update):
    value = next(iter(update.values()))
    if isinstance(value, list):
        value = "x".join(str(item) for item in value)
    return f"{index:02d}_{next(iter(update))}_{value}".replace(".", "p")


def _final_parameters(result):
    values = {
        key: result[key]
        for key in (
            "step",
            "total_loss",
            "data_loss",
            "physics_loss",
            "sparse_loss",
            "effective_component_count",
            "components",
            "alpha",
            "center_x",
            "center_y",
            "sigma_x",
            "sigma_y",
            "theta",
            "runtime_s",
            "peak_vram_mb",
        )
    }
    values["laplacian_backend"] = result.get("laplacian_backend", "unknown")
    return values



def _resolve_device(device_name):
    if device_name is None or str(device_name).strip().lower() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(str(device_name))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
        index = 0 if device.index is None else int(device.index)
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {index} is out of range; "
                f"visible device count={torch.cuda.device_count()}"
            )
        return torch.device(f"cuda:{index}")
    if device.type != "cpu":
        raise ValueError("E1 device must be cpu or cuda:N")
    return device


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace the PINN selection output")
    parser.add_argument(
        "--device",
        default="auto",
        help="single execution device for E1, e.g. cuda:0, cuda:1, cpu, or auto",
    )
    parser.add_argument(
        "--laplacian-backend",
        choices=("analytic", "autograd"),
        default="autograd",
        help="analytic chain-rule Laplacian (fast candidate) or nested-autograd reference",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        force=args.force,
        device_name=args.device,
        laplacian_backend=args.laplacian_backend,
    )
