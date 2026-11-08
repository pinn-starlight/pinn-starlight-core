"""Select Stage-10 cosine-domain LS lambda_LS and sparse weight on validation only."""

from __future__ import annotations

import argparse

import numpy as np
from tqdm import tqdm

from experiments.common.baselines import cosine_ls as fourier_ls
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import selection_artifacts as artifacts
from experiments.configs.formal_experiment import selection_dependency_config

OUTPUT_ROOT = utils.FOURIER_LS_SELECTION_ROOT
ARTIFACT_PATH = utils.FOURIER_LS_SELECTION_ARTIFACT
MANIFEST = utils.SYNTHETIC_MANIFEST


def select_hyperparameters(
    validation_samples,
    *,
    return_details=False,
    show_progress=False,
    backend=None,
    fft_workers=None,
):
    """Two-stage validation selection: lambda first, then L1 sparse coefficient."""
    formal = selection_dependency_config("fourier_ls")
    selection = formal["selection"]
    samples = list(validation_samples)
    if not samples:
        raise ValueError("validation_samples must not be empty")

    lambdas = tuple(float(v) for v in selection["lambda_ls_candidates"])
    sparse_for_lambda = float(selection["lambda_stage_sparse_weight"])
    sparse_candidates = tuple(float(v) for v in selection["sparse_weight_candidates"])
    if any(v <= 0 or not np.isfinite(v) for v in lambdas):
        raise ValueError("lambda candidates must be finite and positive")
    if any(v < 0 or not np.isfinite(v) for v in sparse_candidates):
        raise ValueError("sparse-weight candidates must be finite and non-negative")

    total = (len(lambdas) + 1 + len(sparse_candidates)) * len(samples)
    progress = (
        tqdm(total=total, desc="Cosine-LS Stage-10 selection", unit="fit", leave=True)
        if show_progress
        else None
    )
    try:
        lambda_rows = [
            _evaluate_candidate(
                samples,
                lambda_ls=value,
                sparse_weight=sparse_for_lambda,
                phase="lambda_initial",
                progress_bar=progress,
                show_inner_progress=show_progress,
                backend=backend,
                fft_workers=fft_workers,
            )
            for value in lambdas
        ]
        initial_best = min(lambda_rows, key=_selection_key)
        extension = None
        if np.isclose(initial_best["lambda_ls"], min(lambdas)):
            extension = float(selection["lower_boundary_extension"])
        elif np.isclose(initial_best["lambda_ls"], max(lambdas)):
            extension = float(selection["upper_boundary_extension"])
        if extension is not None and not any(np.isclose(extension, v) for v in lambdas):
            lambda_rows.append(
                _evaluate_candidate(
                    samples,
                    lambda_ls=extension,
                    sparse_weight=sparse_for_lambda,
                    phase="lambda_boundary_extension",
                    progress_bar=progress,
                    show_inner_progress=show_progress,
                    backend=backend,
                    fft_workers=fft_workers,
                )
            )
        selected_lambda = float(min(lambda_rows, key=_selection_key)["lambda_ls"])

        sparse_rows = [
            _evaluate_candidate(
                samples,
                lambda_ls=selected_lambda,
                sparse_weight=value,
                phase="sparse_weight",
                progress_bar=progress,
                show_inner_progress=show_progress,
                backend=backend,
                fft_workers=fft_workers,
            )
            for value in sparse_candidates
        ]
        best = min(sparse_rows, key=_selection_key)
    finally:
        if progress is not None:
            progress.close()

    details = {
        "selected_lambda_ls": selected_lambda,
        "selected_sparse_weight": float(best["sparse_weight"]),
        "selection_metric": selection["selection_metric"],
        "lambda_stage_sparse_weight": sparse_for_lambda,
        "initial_lambda_ls_candidates": list(lambdas),
        "lambda_boundary_extension": extension,
        "sparse_weight_candidates": list(sparse_candidates),
        "lambda_candidates": lambda_rows,
        "sparse_candidates": sparse_rows,
    }
    if return_details:
        return details
    return selected_lambda, float(best["sparse_weight"])


def _evaluate_candidate(
    samples,
    *,
    lambda_ls,
    sparse_weight,
    phase,
    progress_bar=None,
    show_inner_progress=False,
    backend=None,
    fft_workers=None,
):
    sample_mae, sample_obj, sample_runtime, sample_nfev, active_counts = [], [], [], [], []
    for sample_index, sample in enumerate(samples, start=1):
        observed = np.asarray(sample["observed"], dtype=np.float32)
        background_true = np.asarray(sample["background_true"], dtype=np.float64)
        prediction, info = fourier_ls.estimate_background(
            observed,
            float(lambda_ls),
            float(sparse_weight),
            return_info=True,
            show_progress=show_inner_progress,
            progress_desc=(
                f"Cosine-LS λ={float(lambda_ls):g} μ={float(sparse_weight):g} "
                f"[{sample_index}/{len(samples)}]"
            ),
            backend=backend,
            fft_workers=fft_workers,
        )
        sample_mae.append(float(np.mean(np.abs(prediction - background_true))))
        sample_obj.append(float(info["objective"]))
        sample_runtime.append(float(info["runtime_s"]))
        sample_nfev.append(float(info["profile_nfev"]))
        active_counts.append(float(info["effective_component_count"]))
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix(
                phase=phase,
                lambda_ls=f"{float(lambda_ls):g}",
                sparse=f"{float(sparse_weight):g}",
            )
    return {
        "phase": phase,
        "lambda_ls": float(lambda_ls),
        "sparse_weight": float(sparse_weight),
        "validation_background_mae": float(np.mean(sample_mae)),
        "validation_background_mae_std": float(np.std(sample_mae)),
        "mean_profile_objective": float(np.mean(sample_obj)),
        "mean_runtime_s": float(np.mean(sample_runtime)),
        "mean_profile_nfev": float(np.mean(sample_nfev)),
        "mean_effective_component_count": float(np.mean(active_counts)),
        "sample_count": len(sample_mae),
    }


def _selection_key(row):
    # Bg MAE is primary.  The active-count tie-break rewards a simpler forcing
    # only when validation errors are numerically tied.
    return (
        float(row["validation_background_mae"]),
        float(row["mean_effective_component_count"]),
        float(row["sparse_weight"]),
    )


def main(*, force: bool = False, backend=None, fft_workers=None):
    print("Cosine-LS Stage-10 validation selection")
    resolved_backend = fourier_ls._resolve_backend(backend)
    resolved_workers = (
        fourier_ls._resolve_fft_workers(fft_workers) if resolved_backend == "cpu" else 0
    )
    print(
        "Cosine-LS execution:",
        f"backend={resolved_backend}",
        "dtype=float64",
        f"fft_workers={resolved_workers}",
    )
    output_root = utils.prepare_output_root(OUTPUT_ROOT, force=force)
    validation_rows = utils.load_manifest(MANIFEST, split="validation")
    if not validation_rows:
        raise ValueError("validation split must be non-empty")
    validation_samples = [utils.load_synthetic_sample(row) for row in validation_rows]

    config = selection_dependency_config("fourier_ls")
    details = select_hyperparameters(
        validation_samples,
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
    artifact = artifacts.build_selection_artifact(
        method="fourier_ls",
        formal_config=config,
        selected=selected,
        manifest=MANIFEST,
        rows_by_split={"validation": validation_rows},
        metadata={
            "selection_strategy": (
                "lambda selected with frozen middle sparse candidate, then sparse coefficient "
                "selected at the chosen lambda"
            ),
            "execution_backend": resolved_backend,
            "numeric_dtype": "float64",
            "fft_workers": resolved_workers,
            "spectral_profile": "cached_dct_parseval_gram",
        },
    )
    utils.save_rows_csv(output_root / "lambda_candidates.csv", details["lambda_candidates"])
    utils.save_rows_csv(output_root / "sparse_candidates.csv", details["sparse_candidates"])
    utils.write_json(ARTIFACT_PATH, artifact)
    print(
        "Cosine-LS selected:",
        f"lambda_LS={selected['lambda_ls']}",
        f"sparse_weight={selected['sparse_weight']}",
        ARTIFACT_PATH,
    )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace this selector output")
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="execution backend; auto uses CUDA when available",
    )
    parser.add_argument(
        "--fft-workers",
        type=int,
        default=None,
        help="SciPy FFT workers for CPU backend (ignored on CUDA)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(force=args.force, backend=args.backend, fft_workers=args.fft_workers)
