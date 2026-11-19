"""Stage10-B cosine-domain LS validation selector using the Stage10-A search protocol."""

from __future__ import annotations

import argparse

from experiments.common.baselines import cosine_ls as fourier_ls
from experiments.common.data.generate_stage_b import STAGE_B_MANIFEST
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import stage_b_provenance as provenance
from experiments.configs.stage_b_experiment import stage_b_selection_config
from experiments.scripts.selection.fourier_ls import select_hyperparameters

OUTPUT_ROOT = utils.OUTPUT_ROOT / "stage_b/selection/fourier_ls"
ARTIFACT = OUTPUT_ROOT / "fourier_ls_selected.json"
SOURCE_FILES = (
    "experiments/scripts/stage_b/select_fourier_ls.py",
    "experiments/scripts/selection/fourier_ls.py",
    "experiments/configs/stage_b_experiment.py",
    "experiments/common/data/generate_stage_b.py",
    "experiments/common/data/stage_b_backgrounds.py",
    "experiments/common/baselines/cosine_ls.py",
    "src/pinn_starlight_core/physics/forcing.py",
    "experiments/common/utils/experiment_utils.py",
)


def main(*, force=False, backend="auto", fft_workers=None):
    output_root = utils.prepare_output_root(OUTPUT_ROOT, force=force)
    rows = utils.load_manifest(STAGE_B_MANIFEST, split="validation")
    samples = [utils.load_synthetic_sample(row) for row in rows]
    config = stage_b_selection_config("fourier_ls")

    resolved_backend = fourier_ls._resolve_backend(backend)
    resolved_workers = (
        fourier_ls._resolve_fft_workers(fft_workers) if resolved_backend == "cpu" else 0
    )
    details = select_hyperparameters(
        samples,
        return_details=True,
        show_progress=True,
        backend=resolved_backend,
        fft_workers=(resolved_workers if resolved_backend == "cpu" else None),
    )
    selected = {
        "lambda_ls": float(details["selected_lambda_ls"]),
        "sparse_weight": float(details["selected_sparse_weight"]),
        "selection_metric": details["selection_metric"],
        "lambda_stage_sparse_weight": details["lambda_stage_sparse_weight"],
        "initial_lambda_ls_candidates": details["initial_lambda_ls_candidates"],
        "lambda_boundary_extension": details["lambda_boundary_extension"],
        "sparse_weight_candidates": details["sparse_weight_candidates"],
        "lambda_candidates": details["lambda_candidates"],
        "sparse_candidates": details["sparse_candidates"],
        **config["method"],
    }
    fourier_ls.validate_frozen_config(selected)
    artifact = provenance.build_selection_artifact(
        method="fourier_ls",
        selection_config=config,
        selected=selected,
        manifest=STAGE_B_MANIFEST,
        rows_by_split={"validation": rows},
        source_files=SOURCE_FILES,
        metadata={
            "execution_backend": resolved_backend,
            "numeric_dtype": "float64",
            "fft_workers": resolved_workers,
        },
    )
    utils.save_rows_csv(output_root / "lambda_candidates.csv", details["lambda_candidates"])
    utils.save_rows_csv(output_root / "sparse_candidates.csv", details["sparse_candidates"])
    utils.write_json(ARTIFACT, artifact)
    print(
        "Stage-B Cosine-LS selected:",
        f"lambda={selected['lambda_ls']}",
        f"sparse={selected['sparse_weight']}",
        ARTIFACT,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fft-workers", type=int, default=None)
    args = parser.parse_args()
    main(force=args.force, backend=args.backend, fft_workers=args.fft_workers)
