"""Stage-10 cosine-domain least-squares screened-Poisson baseline.

For fixed geometry the screened-Poisson background is solved analytically on a
pixel-centred grid with a zero-normal-derivative (Neumann) boundary condition.
The corresponding discrete Laplacian is diagonalized by an orthonormal DCT-II.
The non-negative effective-forcing weights are profiled out exactly by
enumerating all 2^K active sets (K=3), while geometry is refined with a small
bounded block-coordinate Powell search.

The source term is a signed effective forcing, not a physical non-negative
city-radiance image.  The historical internal module name is retained for
compatibility, but the paper-facing method is cosine-domain least squares.
"""

from __future__ import annotations

import itertools
import math
import os
import time

import numpy as np
from scipy import fft
from scipy.optimize import minimize
from tqdm import tqdm

try:
    import torch
except ImportError:  # CPU-only installations remain supported.
    torch = None

from experiments.configs.formal_experiment import FOURIER_LS_CONFIG
from pinn_starlight_core.physics.forcing import (
    build_forcing_component_numpy,
    candidate_centers_from_observed,
    forcing_negative_energy_fraction,
)

_METHOD_CONFIG = FOURIER_LS_CONFIG["method"]
ALPHA = float(_METHOD_CONFIG["alpha"])
MAX_COMPONENTS = int(_METHOD_CONFIG["max_components"])
_CENTER = _METHOD_CONFIG["center_initialization"]
WINSOR_PERCENTILE = float(_CENTER["winsor_percentile"])
LOW_PASS_SIGMA_FRACTION = float(_CENTER["low_pass_sigma_fraction"])
MIN_CENTER_DISTANCE = float(_CENTER["min_center_distance"])
CENTER_REFINEMENT = float(_CENTER["center_refinement"])
_FORCING = _METHOD_CONFIG["forcing"]
RMS_GRID_SIZE = int(_FORCING["rms_grid_size"])
RMS_EPSILON = float(_FORCING["rms_epsilon"])
INACTIVE_RELATIVE_THRESHOLD = float(_FORCING["inactive_relative_threshold"])
_GEOMETRY = _METHOD_CONFIG["geometry"]
SIGMA_BOUNDS = tuple(float(value) for value in _GEOMETRY["sigma_bounds"])
THETA_BOUNDS = tuple(float(value) for value in _GEOMETRY["theta_bounds"])
INITIAL_SIGMA = float(_GEOMETRY["initial_sigma"])
INITIAL_THETA = float(_GEOMETRY["initial_theta"])
_OPTIMIZER = _METHOD_CONFIG["optimizer"]
MAX_SWEEPS = int(_OPTIMIZER["max_sweeps"])
RELATIVE_OBJECTIVE_EARLY_STOP = float(_OPTIMIZER["relative_objective_early_stop"])
COMPONENT_MAXITER = int(_OPTIMIZER["component_maxiter"])
POWELL_XTOL = float(_OPTIMIZER["xtol"])
POWELL_FTOL = float(_OPTIMIZER["ftol"])


def estimate_background(
    observed,
    lambda_ls: float,
    sparse_weight: float,
    *,
    return_info: bool = False,
    show_progress: bool = False,
    progress_desc: str = "Cosine-LS profile BCD",
    backend: str | None = None,
    fft_workers: int | None = None,
):
    """Estimate B_hat using only observed and validation-selected global scalars."""
    return _estimate_background_with_settings(
        observed,
        lambda_ls,
        sparse_weight,
        alpha=ALPHA,
        max_components=MAX_COMPONENTS,
        max_sweeps=MAX_SWEEPS,
        component_maxiter=COMPONENT_MAXITER,
        xtol=POWELL_XTOL,
        ftol=POWELL_FTOL,
        return_info=return_info,
        show_progress=show_progress,
        progress_desc=progress_desc,
        backend=backend,
        fft_workers=fft_workers,
    )


def _estimate_background_with_settings(
    observed,
    lambda_ls: float,
    sparse_weight: float,
    *,
    alpha: float = ALPHA,
    max_components: int = MAX_COMPONENTS,
    max_sweeps: int = MAX_SWEEPS,
    component_maxiter: int = COMPONENT_MAXITER,
    xtol: float = POWELL_XTOL,
    ftol: float = POWELL_FTOL,
    return_info: bool = False,
    show_progress: bool = False,
    progress_desc: str = "Cosine-LS profile BCD",
    backend: str | None = None,
    fft_workers: int | None = None,
):
    """Full-resolution cosine-domain LS with cached spectral profiling.

    The optimization problem and Powell budget are unchanged from Stage 10.
    The acceleration is implementation-only: invariant cosine-domain terms are cached,
    only the component currently varied by Powell is rebuilt, and the quadratic
    profile objective is evaluated from Parseval-domain Gram terms.
    """
    image = _validate_observed(observed)
    lambda_ls = _validate_positive(lambda_ls, "lambda_ls")
    sparse_weight = _validate_nonnegative(sparse_weight, "sparse_weight")
    alpha = _validate_positive(alpha, "alpha")
    if alpha != ALPHA:
        raise ValueError("formal Fourier-LS requires alpha fixed at 0.5")
    max_components = int(max_components)
    if max_components <= 0 or max_components > MAX_COMPONENTS:
        raise ValueError(f"max_components must be in [1,{MAX_COMPONENTS}]")
    max_sweeps = int(max_sweeps)
    component_maxiter = int(component_maxiter)
    if max_sweeps <= 0 or component_maxiter <= 0:
        raise ValueError("max_sweeps and component_maxiter must be positive")

    started = time.perf_counter()
    initial_centers = candidate_centers_from_observed(
        image,
        max_components=max_components,
        winsor_percentile=WINSOR_PERCENTILE,
        low_pass_sigma_fraction=LOW_PASS_SIGMA_FRACTION,
        min_center_distance=MIN_CENTER_DISTANCE,
        coordinate_mode="pixel_center",
    )
    geometry = np.column_stack(
        [
            initial_centers[:, 0],
            initial_centers[:, 1],
            np.full(max_components, INITIAL_SIGMA),
            np.full(max_components, INITIAL_SIGMA),
            np.full(max_components, INITIAL_THETA),
        ]
    ).astype(np.float64)
    component_bounds = [_component_bounds(center) for center in initial_centers]
    workspace = _prepare_workspace(
        image,
        lambda_ls,
        alpha,
        backend=backend,
        fft_workers=fft_workers,
    )

    states = [
        _component_state_from_geometry(workspace, row, alpha=alpha)
        for row in geometry
    ]
    g, q = _gram_from_states(workspace, states)
    current = _profile_from_gram(
        workspace["base_smooth_objective"],
        g,
        q,
        sparse_weight=sparse_weight,
    )

    # profile_nfev keeps the historical meaning: number of profiled objective
    # evaluations, including accepted/final states outside scipy's own nfev.
    nfev = 1
    nit = 0
    component_records = []
    outer_bar = (
        tqdm(
            total=max_sweeps * max_components,
            desc=str(progress_desc),
            unit="component",
            leave=False,
        )
        if show_progress
        else None
    )
    completed_sweeps = 0
    try:
        for sweep in range(max_sweeps):
            sweep_start = float(current["objective"])
            for component_index in range(max_components):
                inner_bar = (
                    tqdm(
                        total=component_maxiter,
                        desc=f"{progress_desc} s{sweep + 1} c{component_index + 1}",
                        unit="iter",
                        leave=False,
                    )
                    if show_progress
                    else None
                )
                callback_count = 0

                def objective_one(vector):
                    nonlocal nfev
                    nfev += 1
                    candidate_state = _component_state_from_geometry(
                        workspace,
                        np.asarray(vector, dtype=np.float64),
                        alpha=alpha,
                    )
                    candidate_g, candidate_q_row = _candidate_gram_row(
                        workspace,
                        candidate_state,
                        states,
                        replaced_index=component_index,
                    )
                    g_candidate = g.copy()
                    q_candidate = q.copy()
                    g_candidate[component_index] = candidate_g
                    q_candidate[component_index, :] = candidate_q_row
                    q_candidate[:, component_index] = candidate_q_row
                    profiled = _profile_from_gram(
                        workspace["base_smooth_objective"],
                        g_candidate,
                        q_candidate,
                        sparse_weight=sparse_weight,
                    )
                    return float(profiled["objective"])

                def callback(_xk):
                    nonlocal callback_count, nit
                    callback_count += 1
                    nit += 1
                    if inner_bar is not None and inner_bar.n < inner_bar.total:
                        inner_bar.update(1)

                try:
                    result = minimize(
                        objective_one,
                        geometry[component_index],
                        method="Powell",
                        bounds=component_bounds[component_index],
                        callback=callback,
                        options={
                            "maxiter": component_maxiter,
                            "xtol": float(xtol),
                            "ftol": float(ftol),
                            "disp": False,
                        },
                    )
                finally:
                    if inner_bar is not None:
                        inner_bar.close()

                geometry[component_index] = np.asarray(result.x, dtype=np.float64)
                states[component_index] = _component_state_from_geometry(
                    workspace,
                    geometry[component_index],
                    alpha=alpha,
                )
                nfev += 1
                g, q = _gram_from_states(workspace, states)
                current = _profile_from_gram(
                    workspace["base_smooth_objective"],
                    g,
                    q,
                    sparse_weight=sparse_weight,
                )
                component_records.append(
                    {
                        "sweep": sweep + 1,
                        "component": component_index + 1,
                        "success": bool(result.success),
                        "message": str(result.message),
                        "nit": int(getattr(result, "nit", callback_count)),
                        "nfev": int(getattr(result, "nfev", 0)),
                        "objective_after": float(current["objective"]),
                    }
                )
                if outer_bar is not None:
                    outer_bar.update(1)
                    outer_bar.set_postfix(objective=f"{current['objective']:.4g}")

            completed_sweeps = sweep + 1
            relative_improvement = (
                sweep_start - float(current["objective"])
            ) / max(abs(sweep_start), 1e-12)
            if relative_improvement < RELATIVE_OBJECTIVE_EARLY_STOP:
                break
    finally:
        if outer_bar is not None:
            outer_bar.close()

    # Final profile does not rebuild any unchanged full-resolution component.
    nfev += 1
    final = _profile_from_gram(
        workspace["base_smooth_objective"],
        g,
        q,
        sparse_weight=sparse_weight,
    )
    weights = np.asarray(final["weights"], dtype=np.float64)
    background = _reconstruct_background(workspace, states, weights)
    background = np.asarray(background, dtype=np.float32)
    relative = weights / max(float(np.max(weights)), 1e-12)

    # These diagnostics are evaluated only once at the final geometry.
    components = []
    forcing_images, scales = _components_from_geometry(image.shape, geometry, alpha=alpha)
    for index in range(max_components):
        components.append(
            {
                "index": index + 1,
                "initial_center_x": float(initial_centers[index, 0]),
                "initial_center_y": float(initial_centers[index, 1]),
                "center_x": float(geometry[index, 0]),
                "center_y": float(geometry[index, 1]),
                "sigma_x": float(geometry[index, 2]),
                "sigma_y": float(geometry[index, 3]),
                "theta": float(geometry[index, 4]),
                "weight": float(weights[index]),
                "relative_weight": float(relative[index]),
                "active": bool(relative[index] >= INACTIVE_RELATIVE_THRESHOLD),
                "canonical_rms_before_normalization": float(scales[index]),
                "negative_energy_fraction": forcing_negative_energy_fraction(
                    forcing_images[index]
                ),
            }
        )
    strongest = max(components, key=lambda item: item["weight"])
    _synchronize_workspace(workspace)
    runtime_s = time.perf_counter() - started
    info = {
        "alpha": float(alpha),
        "lambda_ls": float(lambda_ls),
        "sparse_weight": float(sparse_weight),
        "objective": float(final["objective"]),
        "smooth_objective": float(final["smooth_objective"]),
        "sparse_objective": float(final["sparse_objective"]),
        "weights": [float(value) for value in weights],
        "effective_component_count": int(sum(item["active"] for item in components)),
        "components": components,
        "center_x": float(strongest["center_x"]),
        "center_y": float(strongest["center_y"]),
        "sigma_x": float(strongest["sigma_x"]),
        "sigma_y": float(strongest["sigma_y"]),
        "theta": float(strongest["theta"]),
        "profile_nfev": int(nfev),
        "optimizer_nit": int(nit),
        "completed_sweeps": int(completed_sweeps),
        "component_optimizations": component_records,
        "optimizer": dict(_OPTIMIZER),
        "boundary_condition": "neumann_zero_normal_derivative",
        "spectral_transform": "dct_ii_ortho",
        "coordinate_system": "normalized_pixel_centers_minus1_to_1",
        "background_below_zero_fraction": float(np.mean(background < 0.0)),
        "background_above_one_fraction": float(np.mean(background > 1.0)),
        "compute_backend": str(workspace["backend"]),
        "numeric_dtype": "float64",
        "fft_workers": int(workspace.get("fft_workers", 0)),
        "spectral_profile": "cached_dct_parseval_gram",
        "runtime_s": float(runtime_s),
    }
    if return_info:
        return background, info
    return background


def _component_bounds(initial_center):
    x0, y0 = (float(initial_center[0]), float(initial_center[1]))
    return (
        (max(-1.0, x0 - CENTER_REFINEMENT), min(1.0, x0 + CENTER_REFINEMENT)),
        (max(-1.0, y0 - CENTER_REFINEMENT), min(1.0, y0 + CENTER_REFINEMENT)),
        SIGMA_BOUNDS,
        SIGMA_BOUNDS,
        THETA_BOUNDS,
    )


def _components_from_geometry(shape, geometry, *, alpha):
    components = []
    scales = []
    for row in np.asarray(geometry, dtype=np.float64):
        component, scale = build_forcing_component_numpy(
            shape,
            center=(row[0], row[1]),
            sigma_x=float(row[2]),
            sigma_y=float(row[3]),
            theta=float(row[4]),
            alpha=float(alpha),
            normalize=True,
            rms_grid_size=RMS_GRID_SIZE,
            rms_epsilon=RMS_EPSILON,
        )
        components.append(component)
        scales.append(scale)
    return np.asarray(components, dtype=np.float64), np.asarray(scales, dtype=np.float64)


def _profile_nonnegative_weights(
    observed,
    components,
    *,
    lambda_ls,
    sparse_weight,
    alpha,
    workspace=None,
):
    """Profile K non-negative weights exactly using cached spectral Gram terms.

    This compatibility helper accepts explicit component images.  The main
    geometry optimizer uses the even faster component-state cache directly.
    """
    image = np.asarray(observed, dtype=np.float64)
    components = np.asarray(components, dtype=np.float64)
    if components.ndim != 3 or components.shape[1:] != image.shape:
        raise ValueError("components must have shape (K,H,W)")
    if workspace is None:
        workspace = _prepare_workspace(image, lambda_ls, alpha, backend="cpu")
    states = [
        _component_state_from_array(workspace, component, scale=float("nan"))
        for component in components
    ]
    g, q = _gram_from_states(workspace, states)
    profiled = _profile_from_gram(
        workspace["base_smooth_objective"],
        g,
        q,
        sparse_weight=sparse_weight,
    )
    background = _reconstruct_background(workspace, states, profiled["weights"])
    return {
        **profiled,
        "background": np.asarray(background, dtype=np.float64),
    }


def _profile_quadratic_objective(
    image,
    b0,
    b_response,
    p0,
    p_response,
    weights,
    *,
    lambda_ls,
    sparse_weight,
):
    background = b0 + np.tensordot(weights, b_response, axes=(0, 0))
    physics_residual = p0 + np.tensordot(weights, p_response, axes=(0, 0))
    data_residual = background - image
    smooth = float(
        np.mean(data_residual * data_residual)
        + float(lambda_ls) * np.mean(physics_residual * physics_residual)
    )
    sparse = float(float(sparse_weight) * np.mean(weights))
    return {
        "objective": smooth + sparse,
        "smooth_objective": smooth,
        "sparse_objective": sparse,
    }


def validate_frozen_config(config: dict) -> tuple[float, float]:
    """Validate a Stage-10 frozen Fourier-LS config and return (lambda, sparse)."""
    if not isinstance(config, dict):
        raise TypeError("Fourier-LS frozen config must be a dict")
    if "lambda_ls" not in config or "sparse_weight" not in config:
        raise ValueError("Fourier-LS frozen config requires lambda_ls and sparse_weight")
    lambda_ls = _validate_positive(config["lambda_ls"], "lambda_ls")
    sparse_weight = _validate_nonnegative(config["sparse_weight"], "sparse_weight")
    if float(config.get("alpha", float("nan"))) != ALPHA:
        raise ValueError("Frozen Fourier-LS alpha does not match the Stage-10 method")
    if int(config.get("max_components", -1)) != MAX_COMPONENTS:
        raise ValueError("Frozen Fourier-LS max_components mismatch")

    center = config.get("center_initialization", {})
    expected_center = _METHOD_CONFIG["center_initialization"]
    for key in (
        "rule",
        "winsor_percentile",
        "low_pass_sigma_fraction",
        "min_center_distance",
        "center_refinement",
        "coordinate_rule",
    ):
        if not _same_value(center.get(key), expected_center.get(key)):
            raise ValueError(f"Frozen Fourier-LS center setting mismatch: {key}")
    forcing = config.get("forcing", {})
    for key, expected in _METHOD_CONFIG["forcing"].items():
        if not _same_value(forcing.get(key), expected):
            raise ValueError(f"Frozen Fourier-LS forcing setting mismatch: {key}")
    geometry = config.get("geometry", {})
    for key, expected in _METHOD_CONFIG["geometry"].items():
        if not _same_value(geometry.get(key), expected):
            raise ValueError(f"Frozen Fourier-LS geometry setting mismatch: {key}")
    optimizer = config.get("optimizer", {})
    for key, expected in _METHOD_CONFIG["optimizer"].items():
        if not _same_value(optimizer.get(key), expected):
            raise ValueError(f"Frozen Fourier-LS optimizer setting mismatch: {key}")
    for key in (
        "boundary_condition",
        "spectral_transform",
        "coordinate_system",
        "scientific_output_clipping",
        "deterministic",
    ):
        if not _same_value(config.get(key), _METHOD_CONFIG[key]):
            raise ValueError(f"Frozen Fourier-LS setting mismatch: {key}")
    return lambda_ls, sparse_weight


def _same_value(actual, expected):
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            return False
        return all(_same_value(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, float):
        try:
            return bool(np.isclose(float(actual), expected, rtol=0.0, atol=1e-12))
        except (TypeError, ValueError):
            return False
    return actual == expected


def _resolve_backend(backend=None):
    requested = str(
        backend
        if backend is not None
        else os.environ.get("PINN_STARLIGHT_FOURIER_BACKEND", "auto")
    ).strip().lower()
    aliases = {
        "gpu": "cuda",
        "cuda_float64": "cuda",
        "cpu_float64": "cpu",
    }
    requested = aliases.get(requested, requested)
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Cosine-LS backend must be one of: auto, cpu, cuda")
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "Cosine-LS CUDA backend requested but torch.cuda.is_available() is false"
        )
    return requested


def _resolve_fft_workers(fft_workers=None):
    if fft_workers is None:
        env = os.environ.get("PINN_STARLIGHT_FOURIER_FFT_WORKERS")
        if env is not None:
            fft_workers = int(env)
        else:
            fft_workers = min(8, max(1, os.cpu_count() or 1))
    workers = int(fft_workers)
    if workers <= 0:
        raise ValueError("fft_workers must be positive")
    return workers


def _dct_1d_torch(values, dim=-1):
    """Orthonormal DCT-II implemented with torch.fft for CUDA/CPU tensors."""
    if torch is None:
        raise RuntimeError("torch is unavailable")
    values = values.movedim(dim, -1)
    shape = values.shape
    n = int(shape[-1])
    flat = values.contiguous().reshape(-1, n)
    reordered = torch.cat((flat[:, ::2], flat[:, 1::2].flip((1,))), dim=1)
    spectrum = torch.fft.fft(reordered, dim=1)
    k = -torch.arange(n, dtype=values.dtype, device=values.device) * math.pi / (2.0 * n)
    transformed = spectrum.real * torch.cos(k)[None, :] - spectrum.imag * torch.sin(k)[None, :]
    transformed[:, 0] /= (math.sqrt(n) * 2.0)
    if n > 1:
        transformed[:, 1:] /= (math.sqrt(n / 2.0) * 2.0)
    transformed = 2.0 * transformed
    return transformed.reshape(shape).movedim(-1, dim)


def _idct_1d_torch(values, dim=-1):
    """Inverse of :func:`_dct_1d_torch` (orthonormal DCT-III)."""
    if torch is None:
        raise RuntimeError("torch is unavailable")
    values = values.movedim(dim, -1)
    shape = values.shape
    n = int(shape[-1])
    coeff = values.contiguous().reshape(-1, n) / 2.0
    coeff[:, 0] *= math.sqrt(n) * 2.0
    if n > 1:
        coeff[:, 1:] *= math.sqrt(n / 2.0) * 2.0
    k = torch.arange(n, dtype=values.dtype, device=values.device) * math.pi / (2.0 * n)
    cos_k = torch.cos(k)[None, :]
    sin_k = torch.sin(k)[None, :]
    imag_seed = torch.cat((coeff[:, :1] * 0.0, -coeff.flip((1,))[:, :-1]), dim=1)
    real_part = coeff * cos_k - imag_seed * sin_k
    imag_part = coeff * sin_k + imag_seed * cos_k
    spectrum = torch.complex(real_part, imag_part)
    reordered = torch.fft.ifft(spectrum, dim=1).real
    output = torch.zeros_like(reordered)
    output[:, ::2] = reordered[:, : n - n // 2]
    output[:, 1::2] = reordered.flip((1,))[:, : n // 2]
    return output.reshape(shape).movedim(-1, dim)


def _dct2_torch(values):
    return _dct_1d_torch(_dct_1d_torch(values, dim=-1), dim=-2)


def _idct2_torch(values):
    return _idct_1d_torch(_idct_1d_torch(values, dim=-1), dim=-2)


def _pixel_center_axis_numpy(size):
    size = int(size)
    return -1.0 + (np.arange(size, dtype=np.float64) + 0.5) * (2.0 / size)


def _pixel_center_axis_torch(size, device):
    size = int(size)
    return -1.0 + (torch.arange(size, dtype=torch.float64, device=device) + 0.5) * (2.0 / size)


def _prepare_workspace(observed, lambda_ls, alpha, *, backend=None, fft_workers=None):
    image = np.asarray(observed, dtype=np.float64)
    backend = _resolve_backend(backend)
    height, width = image.shape
    n_pixels = float(height * width)
    if backend == "cpu":
        workers = _resolve_fft_workers(fft_workers)
        image_spectrum = fft.dctn(image, type=2, norm="ortho", workers=workers)
        screened_operator = _screened_operator_symbol(image.shape, alpha)
        denominator = 1.0 + float(lambda_ls) * screened_operator**2
        b0_spectrum = image_spectrum / denominator
        d0_spectrum = b0_spectrum - image_spectrum
        p0_spectrum = screened_operator * b0_spectrum
        workspace = {
            "backend": "cpu",
            "fft_workers": workers,
            "shape": image.shape,
            "n_pixels": n_pixels,
            "lambda_ls": float(lambda_ls),
            "alpha": float(alpha),
            "image_spectrum": image_spectrum,
            "screened_operator": screened_operator,
            "denominator": denominator,
            "b0_spectrum": b0_spectrum,
            "d0_spectrum": d0_spectrum,
            "p0_spectrum": p0_spectrum,
            "transfer_b": -float(lambda_ls) * screened_operator / denominator,
            "transfer_p": 1.0 / denominator,
            "x_axis": _pixel_center_axis_numpy(width)[None, :],
            "y_axis": _pixel_center_axis_numpy(height)[:, None],
            "canonical_axis": _pixel_center_axis_numpy(RMS_GRID_SIZE),
        }
    else:
        device = torch.device("cuda")
        image_tensor = torch.as_tensor(image, dtype=torch.float64, device=device)
        image_spectrum = _dct2_torch(image_tensor)
        screened_operator = _screened_operator_symbol_torch(image.shape, alpha, device)
        denominator = 1.0 + float(lambda_ls) * screened_operator.square()
        b0_spectrum = image_spectrum / denominator
        d0_spectrum = b0_spectrum - image_spectrum
        p0_spectrum = screened_operator * b0_spectrum
        workspace = {
            "backend": "cuda",
            "fft_workers": 0,
            "shape": image.shape,
            "n_pixels": n_pixels,
            "lambda_ls": float(lambda_ls),
            "alpha": float(alpha),
            "device": device,
            "image_spectrum": image_spectrum,
            "screened_operator": screened_operator,
            "denominator": denominator,
            "b0_spectrum": b0_spectrum,
            "d0_spectrum": d0_spectrum,
            "p0_spectrum": p0_spectrum,
            "transfer_b": -float(lambda_ls) * screened_operator / denominator,
            "transfer_p": 1.0 / denominator,
            "x_axis": _pixel_center_axis_torch(width, device)[None, :],
            "y_axis": _pixel_center_axis_torch(height, device)[:, None],
            "canonical_axis": _pixel_center_axis_torch(RMS_GRID_SIZE, device),
        }
    workspace["q_kernel"] = (
        workspace["transfer_b"] * workspace["transfer_b"]
        + float(lambda_ls) * workspace["transfer_p"] * workspace["transfer_p"]
    )
    workspace["g_reference_spectrum"] = (
        workspace["transfer_b"] * workspace["d0_spectrum"]
        + float(lambda_ls) * workspace["transfer_p"] * workspace["p0_spectrum"]
    )
    workspace["base_smooth_objective"] = (
        _spectral_mean_inner(
            workspace,
            workspace["d0_spectrum"],
            workspace["d0_spectrum"],
        )
        + float(lambda_ls)
        * _spectral_mean_inner(
            workspace,
            workspace["p0_spectrum"],
            workspace["p0_spectrum"],
        )
    )
    return workspace


def _spectral_mean_inner(workspace, left, right, *, kernel=None):
    # Orthonormal DCT-II preserves the Euclidean inner product (Parseval).
    scale = workspace["n_pixels"]
    if workspace["backend"] == "cpu":
        product = np.asarray(left, dtype=np.float64) * np.asarray(right, dtype=np.float64)
        if kernel is not None:
            product = product * np.asarray(kernel, dtype=np.float64)
        return float(np.sum(product, dtype=np.float64) / scale)
    product = left * right
    if kernel is not None:
        product = product * kernel
    return float((torch.sum(product) / scale).item())


def _screened_operator_symbol_torch(shape, alpha, device):
    height, width = (int(shape[0]), int(shape[1]))
    if height < 2 or width < 2:
        raise ValueError("shape must be at least 2x2")
    hx = 2.0 / width
    hy = 2.0 / height
    q = torch.arange(height, dtype=torch.float64, device=device)[:, None]
    p = torch.arange(width, dtype=torch.float64, device=device)[None, :]
    laplacian_symbol = (
        -4.0 * torch.sin(math.pi * q / (2.0 * height)).square() / (hy * hy)
        -4.0 * torch.sin(math.pi * p / (2.0 * width)).square() / (hx * hx)
    )
    return laplacian_symbol - float(alpha)


def _component_state_from_geometry(workspace, geometry_row, *, alpha):
    row = np.asarray(geometry_row, dtype=np.float64)
    if row.shape != (5,):
        raise ValueError("geometry row must have five parameters")
    if workspace["backend"] == "cpu":
        component, scale = _build_forcing_component_cached_numpy(
            workspace,
            row,
            alpha=float(alpha),
        )
    else:
        component, scale = _build_forcing_component_torch(
            workspace,
            row,
            alpha=float(alpha),
        )
    return _component_state_from_array(workspace, component, scale=scale)


def _build_forcing_component_cached_numpy(workspace, row, *, alpha):
    center_x, center_y, sigma_x, sigma_y, theta = (float(v) for v in row)
    dx = workspace["x_axis"] - center_x
    dy = workspace["y_axis"] - center_y
    safe_r = np.sqrt(dx * dx + dy * dy + 2e-8)
    sqrt_alpha = math.sqrt(float(alpha))
    q = (
        2.0 * float(alpha) * np.cos(safe_r * sqrt_alpha)
        + (sqrt_alpha / safe_r) * np.sin(safe_r * sqrt_alpha)
        - 16.0 * safe_r**2 * float(alpha) ** 2
        + safe_r**4 * float(alpha) ** 3
    )
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    dx_rot = cos_theta * dx + sin_theta * dy
    dy_rot = -sin_theta * dx + cos_theta * dy
    envelope = np.exp(
        -0.5
        * (
            (dx_rot / max(sigma_x, 1e-12)) ** 2
            + (dy_rot / max(sigma_y, 1e-12)) ** 2
        )
    )
    component = q * envelope
    axis = workspace["canonical_axis"]
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    cdx = xx - center_x
    cdy = yy - center_y
    cr = np.sqrt(cdx * cdx + cdy * cdy + 2e-8)
    cq = (
        2.0 * float(alpha) * np.cos(cr * sqrt_alpha)
        + (sqrt_alpha / cr) * np.sin(cr * sqrt_alpha)
        - 16.0 * cr**2 * float(alpha) ** 2
        + cr**4 * float(alpha) ** 3
    )
    cdx_rot = cos_theta * cdx + sin_theta * cdy
    cdy_rot = -sin_theta * cdx + cos_theta * cdy
    cenvelope = np.exp(
        -0.5
        * (
            (cdx_rot / max(sigma_x, 1e-12)) ** 2
            + (cdy_rot / max(sigma_y, 1e-12)) ** 2
        )
    )
    canonical = cq * cenvelope
    scale = float(np.sqrt(np.mean(canonical * canonical) + RMS_EPSILON))
    return np.asarray(component / scale, dtype=np.float64), scale


def _build_forcing_component_torch(workspace, row, *, alpha):
    center_x, center_y, sigma_x, sigma_y, theta = (float(v) for v in row)
    with torch.no_grad():
        dx = workspace["x_axis"] - center_x
        dy = workspace["y_axis"] - center_y
        safe_r = torch.sqrt(dx.square() + dy.square() + 2e-8)
        sqrt_alpha = math.sqrt(float(alpha))
        q = (
            2.0 * float(alpha) * torch.cos(safe_r * sqrt_alpha)
            + (sqrt_alpha / safe_r) * torch.sin(safe_r * sqrt_alpha)
            - 16.0 * safe_r.square() * float(alpha) ** 2
            + safe_r.pow(4) * float(alpha) ** 3
        )
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        dx_rot = cos_theta * dx + sin_theta * dy
        dy_rot = -sin_theta * dx + cos_theta * dy
        envelope = torch.exp(
            -0.5
            * (
                (dx_rot / max(sigma_x, 1e-12)).square()
                + (dy_rot / max(sigma_y, 1e-12)).square()
            )
        )
        component = q * envelope

        axis = workspace["canonical_axis"]
        cdy = axis[:, None] - center_y
        cdx = axis[None, :] - center_x
        cr = torch.sqrt(cdx.square() + cdy.square() + 2e-8)
        cq = (
            2.0 * float(alpha) * torch.cos(cr * sqrt_alpha)
            + (sqrt_alpha / cr) * torch.sin(cr * sqrt_alpha)
            - 16.0 * cr.square() * float(alpha) ** 2
            + cr.pow(4) * float(alpha) ** 3
        )
        cdx_rot = cos_theta * cdx + sin_theta * cdy
        cdy_rot = -sin_theta * cdx + cos_theta * cdy
        cenvelope = torch.exp(
            -0.5
            * (
                (cdx_rot / max(sigma_x, 1e-12)).square()
                + (cdy_rot / max(sigma_y, 1e-12)).square()
            )
        )
        canonical = cq * cenvelope
        scale_tensor = torch.sqrt(torch.mean(canonical.square()) + RMS_EPSILON)
        scale = float(scale_tensor.item())
        return component / scale_tensor, scale


def _component_state_from_array(workspace, component, *, scale):
    if workspace["backend"] == "cpu":
        array = np.asarray(component, dtype=np.float64)
        component_spectrum = fft.dctn(
            array, type=2, norm="ortho", workers=workspace["fft_workers"]
        )
    else:
        with torch.no_grad():
            if isinstance(component, np.ndarray):
                component = torch.as_tensor(
                    component, dtype=torch.float64, device=workspace["device"]
                )
            component_spectrum = _dct2_torch(component)
    return {
        "component_spectrum": component_spectrum,
        "scale": float(scale),
    }


def _gram_from_states(workspace, states):
    k_count = len(states)
    g = np.empty(k_count, dtype=np.float64)
    q = np.empty((k_count, k_count), dtype=np.float64)
    for i, state_i in enumerate(states):
        spectrum_i = state_i["component_spectrum"]
        g[i] = _spectral_mean_inner(
            workspace, workspace["g_reference_spectrum"], spectrum_i
        )
        for j in range(i + 1):
            spectrum_j = states[j]["component_spectrum"]
            value = _spectral_mean_inner(
                workspace,
                spectrum_i,
                spectrum_j,
                kernel=workspace["q_kernel"],
            )
            q[i, j] = value
            q[j, i] = value
    return g, q


def _candidate_gram_row(workspace, candidate_state, states, *, replaced_index):
    candidate_spectrum = candidate_state["component_spectrum"]
    candidate_g = _spectral_mean_inner(
        workspace, workspace["g_reference_spectrum"], candidate_spectrum
    )
    row = np.empty(len(states), dtype=np.float64)
    for index, state in enumerate(states):
        comparison = candidate_state if index == int(replaced_index) else state
        row[index] = _spectral_mean_inner(
            workspace,
            candidate_spectrum,
            comparison["component_spectrum"],
            kernel=workspace["q_kernel"],
        )
    return float(candidate_g), row


def _profile_from_gram(base_smooth, g, q, *, sparse_weight):
    g = np.asarray(g, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    q = 0.5 * (q + q.T)
    k_count = int(g.size)
    sparse_gradient_half = float(sparse_weight) / (2.0 * k_count)
    candidates = []
    for mask in itertools.product((False, True), repeat=k_count):
        active = np.flatnonzero(mask)
        weights = np.zeros(k_count, dtype=np.float64)
        if active.size:
            system = q[np.ix_(active, active)]
            rhs = -(g[active] + sparse_gradient_half)
            try:
                solution = np.linalg.solve(system, rhs)
            except np.linalg.LinAlgError:
                solution = np.linalg.lstsq(system, rhs, rcond=None)[0]
            if np.any(solution < -1e-10) or not np.isfinite(solution).all():
                continue
            weights[active] = np.maximum(solution, 0.0)
        smooth = float(base_smooth + 2.0 * np.dot(g, weights) + weights @ q @ weights)
        sparse = float(float(sparse_weight) * np.mean(weights))
        candidates.append((smooth + sparse, weights, smooth, sparse))
    if not candidates:
        raise RuntimeError("non-negative active-set profiler found no feasible candidate")
    objective, weights, smooth, sparse = min(candidates, key=lambda item: item[0])
    return {
        "objective": float(objective),
        "smooth_objective": float(smooth),
        "sparse_objective": float(sparse),
        "weights": np.asarray(weights, dtype=np.float64),
    }


def _reconstruct_background(workspace, states, weights):
    weights = np.asarray(weights, dtype=np.float64)
    if workspace["backend"] == "cpu":
        forcing_spectrum = np.zeros_like(workspace["b0_spectrum"])
        for weight, state in zip(weights, states):
            forcing_spectrum += float(weight) * state["component_spectrum"]
        spectrum = (
            workspace["b0_spectrum"]
            + workspace["transfer_b"] * forcing_spectrum
        )
        return fft.idctn(
            spectrum,
            type=2,
            norm="ortho",
            workers=workspace["fft_workers"],
        )
    with torch.no_grad():
        forcing_spectrum = torch.zeros_like(workspace["b0_spectrum"])
        for weight, state in zip(weights, states):
            forcing_spectrum = (
                forcing_spectrum + float(weight) * state["component_spectrum"]
            )
        spectrum = (
            workspace["b0_spectrum"]
            + workspace["transfer_b"] * forcing_spectrum
        )
        result = _idct2_torch(spectrum)
        return result.detach().cpu().numpy().astype(np.float64, copy=False)


def _synchronize_workspace(workspace):
    if workspace.get("backend") == "cuda":
        torch.cuda.synchronize(workspace["device"])


def _screened_operator_symbol(shape, alpha):
    """DCT-II eigenvalues for a pixel-centred Neumann screened Laplacian."""
    height, width = (int(shape[0]), int(shape[1]))
    if height < 2 or width < 2:
        raise ValueError("shape must be at least 2x2")
    hx = 2.0 / width
    hy = 2.0 / height
    q = np.arange(height, dtype=np.float64)[:, None]
    p = np.arange(width, dtype=np.float64)[None, :]
    laplacian_symbol = (
        -4.0 * np.sin(np.pi * q / (2.0 * height)) ** 2 / (hy * hy)
        -4.0 * np.sin(np.pi * p / (2.0 * width)) ** 2 / (hx * hx)
    )
    return laplacian_symbol - float(alpha)


def _apply_screened_operator(image, alpha):
    """Apply the pixel-centred Neumann finite-difference operator in space."""
    array = np.asarray(image, dtype=np.float64)
    height, width = array.shape
    hx = 2.0 / width
    hy = 2.0 / height
    padded = np.pad(array, ((1, 1), (1, 1)), mode="edge")
    center = padded[1:-1, 1:-1]
    laplacian = (
        (padded[1:-1, 2:] - 2.0 * center + padded[1:-1, :-2]) / (hx * hx)
        + (padded[2:, 1:-1] - 2.0 * center + padded[:-2, 1:-1]) / (hy * hy)
    )
    return laplacian - float(alpha) * array


def _validate_observed(observed):
    if not isinstance(observed, np.ndarray):
        raise TypeError(
            "formal cosine-LS input must be an observed np.ndarray; "
            "load files in the experiment/data layer before calling the Method"
        )
    image = np.asarray(observed, dtype=np.float32)
    if image.ndim != 2 or min(image.shape) < 2:
        raise ValueError("observed must be a 2-D image of at least 2x2")
    if not np.isfinite(image).all():
        raise ValueError("observed must contain only finite values")
    return image


def _validate_positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _validate_nonnegative(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value
