"""E1：只使用合成验证集锁定 PINN 配置。"""

from pathlib import Path

import numpy as np
import torch

from experiments.common.baselines import coordinate_pinn as pinn
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import metrics

FORCE_OVERWRITE_OUTPUT = True

# E1可选配置
NETWORK_CANDIDATES = ([256, 128], [256, 64], [128, 128], [128, 64])
PHYSICS_WEIGHT_CANDIDATES = (0.4, 0.5, 0.6, 0.7)
KERNEL_SIZE_CANDIDATES = (21, 31)

# E1 实验配置；需要调整实验时直接修改这里即可。
MANIFEST = utils.SYNTHETIC_MANIFEST
OUTPUT_ROOT = utils.OUTPUT_ROOT / "e1_tuning"
#E1使用单seed
SEED = utils.SEEDS[0]
INITIAL_STEPS = 10000
CONTINUE_STEPS = 10000
MAX_STEPS = 50000
BATCH_SIZE = 8192
ALPHA = 0.5
MODEL_LR = 1e-3
ICITY_LR = 1e-3
DEFAULT_PHYSICS_WEIGHT = 0.5
DEFAULT_KERNEL_SIZE = 31
STABILITY_TOLERANCE = 0.005
STABILITY_PATIENCE = 3
CENTER_MODE = "bright_init_learnable"


def run_candidate(
    config: dict,
    validation_manifest: list[dict],
    output_dir: Path,
    seed: int,
    device,
    resume_states=None,
    return_states=False,
):
    """在固定验证集上运行一个候选配置，并保存逐图结果。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    utils.save_run_metadata(
        output_dir,
        {"experiment": "E1", "seed": seed, "pinn_config": config},
    )
    rows = []
    states = {}
    for manifest_row in validation_manifest:
        sample = utils.load_synthetic_sample(manifest_row)
        reference_stars = metrics.extract_stars(sample["clean_true"])
        sample = {
            **sample,
            "metadata": {
                "star_reference": {
                    "threshold": 0.3,
                    "matching_radius": 3,
                    "stars": reference_stars,
            },
        },
        }
        sample_id = str(sample["sample_id"])
        result = pinn.train_background(
            sample["observed"],
            config=config,
            device=device,
            seed=seed,
            resume_state=(resume_states or {}).get(sample_id),
            return_state=return_states,
            show_progress=False
        )
        sample_output = output_dir / sample_id
        utils.save_prediction_arrays(
            sample_output,
            result["observed"],
            result["background_pred"],
            result["residual_pred"],
        )
        utils.save_rows_csv(sample_output / "history.csv", result["history"])
        utils.write_json(
            sample_output / "final_parameters.json",
            _final_parameters(result),
        )

        scores = metrics.evaluate_synthetic(sample, result)
        rows.append(
            {
                "sample_id": sample_id,
                "background_type": sample["background_type"],
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


def main():
    print("E1")
    output_root = utils.prepare_output_root(OUTPUT_ROOT, force=FORCE_OVERWRITE_OUTPUT)
    validation_rows = utils.load_manifest(MANIFEST, split="validation")

    if not validation_rows:
        raise ValueError("验证集为空，请先运行正式合成数据生成脚本")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_config = dict(pinn.DEFAULT_CONFIG)
    base_config.update(
        {
            "alpha": ALPHA,
            "steps": INITIAL_STEPS,
            "batch_size": BATCH_SIZE,
            "model_lr": MODEL_LR,
            "icity_lr": ICITY_LR,
            "physics_weight": DEFAULT_PHYSICS_WEIGHT,
            "kernel_size": DEFAULT_KERNEL_SIZE,
            "center_mode": CENTER_MODE,
        }
    )
    utils.set_seed(SEED)
    utils.save_run_metadata(
        output_root,
        {
            "experiment": "E1",
            "manifest": utils.project_relative(MANIFEST),
            "seed": SEED,
            "selection_order": ["network", "physics_weight", "kernel_size", "steps"],
            "center_mode": CENTER_MODE,
            "initial_steps": INITIAL_STEPS,
            "continue_steps": CONTINUE_STEPS,
            "max_steps": MAX_STEPS,
            "batch_size": BATCH_SIZE,
            "alpha": ALPHA,
            "model_lr": MODEL_LR,
            "icity_lr": ICITY_LR,
            "default_physics_weight": DEFAULT_PHYSICS_WEIGHT,
            "default_kernel_size": DEFAULT_KERNEL_SIZE,
            "force_overwrite_output": FORCE_OVERWRITE_OUTPUT,
            "stability_tolerance": STABILITY_TOLERANCE,
            "stability_patience": STABILITY_PATIENCE,
            "validation_samples": [row["sample_id"] for row in validation_rows],
        },
    )

    candidate_table = []
    selected_config, _ = _select_stage(
        "T1_network",
        base_config,
        [{"hidden_dims": list(candidate)} for candidate in NETWORK_CANDIDATES],
        validation_rows,
        output_root,
        SEED,
        device,
        candidate_table,
        keep_states=False,
    )
    selected_config, _ = _select_stage(
        "T2_physics_weight",
        selected_config,
        [{"physics_weight": candidate} for candidate in PHYSICS_WEIGHT_CANDIDATES],
        validation_rows,
        output_root,
        SEED,
        device,
        candidate_table,
        keep_states=False,
    )
    selected_config, selected_result = _select_stage(
        "T3_kernel_size",
        selected_config,
        [{"kernel_size": candidate} for candidate in KERNEL_SIZE_CANDIDATES],
        validation_rows,
        output_root,
        SEED,
        device,
        candidate_table,
        keep_states=True,
    )

    selected_config, selected_result = _continue_until_stable(
        selected_config,
        selected_result,
        validation_rows,
        output_root,
        device,
        candidate_table,
    )
    locked_config = dict(selected_config)
    locked_config["steps"] = int(
        max(row["step"] for row in selected_result["rows"])
    )
    locked_config["selection_seed"] = SEED
    locked_config["validation_manifest"] = utils.project_relative(MANIFEST)
    locked_config["selection_score"] = selected_result["aggregate"]

    utils.save_rows_csv(output_root / "candidate_results.csv", candidate_table)
    utils.write_json(output_root / "locked_pinn_config.json", locked_config)
    print(f"E1 完成：{output_root / 'locked_pinn_config.json'}")
    print(
        "锁定配置：",
        f"hidden={locked_config['hidden_dims']}",
        f"physics_weight={locked_config['physics_weight']}",
        f"kernel={locked_config['kernel_size']}",
        f"steps={locked_config['steps']}",
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
        )
        row = {
            "stage": stage,
            "candidate": name,
            "config": config,
            **result["aggregate"],
        }
        candidate_table.append(row)
        candidate = (config, result)
        if best is None or _selection_key(result["aggregate"]) < _selection_key(
            best[1]["aggregate"]
        ):
            best = candidate
    if best is None:
        raise ValueError(f"{stage} 没有候选配置")
    return best


def _continue_until_stable(
    config,
    current_result,
    validation_rows,
    output_root,
    device,
    candidate_table,
):
    total_steps = int(config["steps"])
    previous_mae = current_result["aggregate"]["bg_mae"]
    states = current_result["states"]
    best_result = current_result
    best_steps = total_steps
    best_key = _selection_key(current_result["aggregate"])
    stable_rounds = 0
    while total_steps < MAX_STEPS:
        additional_steps = min(CONTINUE_STEPS, MAX_STEPS - total_steps)
        continuation_config = dict(config)
        continuation_config["steps"] = additional_steps
        next_total = total_steps + additional_steps
        continued = run_candidate(
            continuation_config,
            validation_rows,
            output_root / "T4_steps" / str(next_total),
            SEED,
            device,
            resume_states=states,
            return_states=True,
        )
        current_mae = continued["aggregate"]["bg_mae"]
        relative_improvement = (previous_mae - current_mae) / max(previous_mae, 1e-12)
        candidate_table.append(
            {
                "stage": "T4_steps",
                "candidate": str(next_total),
                "config": {**config, "steps": next_total},
                "relative_bg_mae_improvement": relative_improvement,
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
        if relative_improvement < STABILITY_TOLERANCE:
            stable_rounds += 1
        else:
            stable_rounds = 0
        if stable_rounds >= STABILITY_PATIENCE:
            break
    config = dict(config)
    config["steps"] = best_steps
    return config, best_result


def _aggregate(rows):
    return {
        "bg_mae": float(np.mean([row["bg_mae"] for row in rows])),
        "residual_ssim": float(np.mean([row["residual_ssim"] for row in rows])),
        "star_f1": float(np.mean([row["star_f1"] for row in rows])),
        "flux_error": float(np.mean([row["flux_error"] for row in rows])),
        "runtime_s": float(np.sum([row["runtime_s"] for row in rows])),
        "parameter_count": float(np.mean([row["parameter_count"] for row in rows])),
    }


def _selection_key(aggregate):
    return (
        aggregate["bg_mae"],
        -aggregate["residual_ssim"],
        aggregate["flux_error"],
        aggregate["parameter_count"],
    )


def _candidate_name(index, update):
    value = next(iter(update.values()))
    if isinstance(value, list):
        value = "x".join(str(item) for item in value)
    return f"{index:02d}_{next(iter(update))}_{value}".replace(".", "p")


def _final_parameters(result):
    return {
        "step": result["step"],
        "total_loss": result["total_loss"],
        "data_loss": result["data_loss"],
        "physics_loss": result["physics_loss"],
        "alpha": result["alpha"],
        "center_x": result["center_x"],
        "center_y": result["center_y"],
        "sigma_x": result["sigma_x"],
        "sigma_y": result["sigma_y"],
        "theta": result["theta"],
        "runtime_s": result["runtime_s"],
        "peak_vram_mb": result["peak_vram_mb"],
    }


if __name__ == "__main__":
    main()
