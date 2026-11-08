"""Stage-10 shared effective-forcing geometry utilities.

The functions in this module are deliberately independent of synthetic truth.
Both PINN and cosine-domain LS use the same observed-only candidate-centre rule and
same signed forcing template family.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter

DEFAULT_MAX_COMPONENTS = 3
DEFAULT_WINSOR_PERCENTILE = 99.5
DEFAULT_LOW_PASS_SIGMA_FRACTION = 0.04
DEFAULT_MIN_CENTER_DISTANCE = 0.30
DEFAULT_CENTER_REFINEMENT = 0.08
DEFAULT_RMS_GRID_SIZE = 64
DEFAULT_RMS_EPSILON = 1e-8


def point_source_term_numpy(r, alpha: float):
    """Return q(r; alpha) = alpha b - Laplacian(b) for the Stage-10 ansatz."""
    radius = np.asarray(r, dtype=np.float64)
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    safe_r = np.sqrt(radius * radius + 1e-8)
    sqrt_alpha = math.sqrt(alpha)
    return (
        2.0 * alpha * np.cos(safe_r * sqrt_alpha)
        + (sqrt_alpha / safe_r) * np.sin(safe_r * sqrt_alpha)
        - 16.0 * (safe_r**2) * (alpha**2)
        + (safe_r**4) * (alpha**3)
    )


def candidate_centers_from_observed(
    observed,
    *,
    max_components: int = DEFAULT_MAX_COMPONENTS,
    winsor_percentile: float = DEFAULT_WINSOR_PERCENTILE,
    low_pass_sigma_fraction: float = DEFAULT_LOW_PASS_SIGMA_FRACTION,
    min_center_distance: float = DEFAULT_MIN_CENTER_DISTANCE,
    coordinate_mode: str = "endpoint",
):
    """Find up to K separated low-frequency peaks using observed only.

    The image used *only for initialization* is winsorized at P99.5 and then
    strongly Gaussian-smoothed. Greedy non-maximum suppression is performed in
    normalized [-1, 1]^2 coordinates, so the rule is resolution-independent.
    ``coordinate_mode='endpoint'`` preserves the PINN endpoint grid;
    ``'pixel_center'`` is used by the DCT-II cosine-domain LS discretization.
    """
    image = np.asarray(observed, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 2 or not np.isfinite(image).all():
        raise ValueError("observed must be a finite 2-D array")
    max_components = int(max_components)
    if max_components <= 0:
        raise ValueError("max_components must be positive")
    winsor_percentile = float(winsor_percentile)
    if not 0.0 < winsor_percentile <= 100.0:
        raise ValueError("winsor_percentile must lie in (0, 100]")
    low_pass_sigma_fraction = float(low_pass_sigma_fraction)
    if low_pass_sigma_fraction <= 0.0:
        raise ValueError("low_pass_sigma_fraction must be positive")
    min_center_distance = float(min_center_distance)
    if min_center_distance <= 0.0:
        raise ValueError("min_center_distance must be positive")

    cap = float(np.percentile(image, winsor_percentile))
    initialization_image = np.minimum(image, cap)
    sigma = low_pass_sigma_fraction * min(image.shape)
    low_frequency = gaussian_filter(
        initialization_image,
        sigma=(sigma, sigma),
        mode="mirror",
    )

    height, width = image.shape
    flat_order = np.argsort(low_frequency.reshape(-1))[::-1]
    selected = []
    for flat_index in flat_order:
        y_pixel, x_pixel = np.unravel_index(int(flat_index), low_frequency.shape)
        if coordinate_mode == "endpoint":
            x = -1.0 + 2.0 * float(x_pixel) / float(width - 1)
            y = -1.0 + 2.0 * float(y_pixel) / float(height - 1)
        elif coordinate_mode == "pixel_center":
            x = -1.0 + (float(x_pixel) + 0.5) * 2.0 / float(width)
            y = -1.0 + (float(y_pixel) + 0.5) * 2.0 / float(height)
        else:
            raise ValueError("coordinate_mode must be 'endpoint' or 'pixel_center'")
        if all(math.hypot(x - cx, y - cy) >= min_center_distance for cx, cy in selected):
            selected.append((x, y))
            if len(selected) >= max_components:
                break
    if len(selected) != max_components:
        raise RuntimeError(f"could not find {max_components} separated initialization centres")
    return np.asarray(selected, dtype=np.float64)


def _component_on_grid(
    x_axis,
    y_axis,
    *,
    center,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    alpha: float,
):
    yy, xx = np.meshgrid(y_axis, x_axis, indexing="ij")
    center_x, center_y = (float(center[0]), float(center[1]))
    dx = xx - center_x
    dy = yy - center_y
    r = np.sqrt(dx * dx + dy * dy + 1e-8)
    q = point_source_term_numpy(r, alpha)
    cos_theta = math.cos(float(theta))
    sin_theta = math.sin(float(theta))
    dx_rot = cos_theta * dx + sin_theta * dy
    dy_rot = -sin_theta * dx + cos_theta * dy
    envelope = np.exp(
        -0.5
        * ((dx_rot / max(float(sigma_x), 1e-12)) ** 2 + (dy_rot / max(float(sigma_y), 1e-12)) ** 2)
    )
    return q * envelope


def unit_rms_scale_numpy(
    *,
    center,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    alpha: float,
    grid_size: int = DEFAULT_RMS_GRID_SIZE,
    epsilon: float = DEFAULT_RMS_EPSILON,
):
    """Return the canonical-grid RMS used to normalize one forcing component."""
    grid_size = int(grid_size)
    if grid_size < 8:
        raise ValueError("grid_size must be at least 8")
    axis = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    component = _component_on_grid(
        axis,
        axis,
        center=center,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        theta=theta,
        alpha=alpha,
    )
    return float(np.sqrt(np.mean(component * component) + float(epsilon)))


def build_forcing_component_numpy(
    shape,
    *,
    center,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    alpha: float,
    normalize: bool = True,
    rms_grid_size: int = DEFAULT_RMS_GRID_SIZE,
    rms_epsilon: float = DEFAULT_RMS_EPSILON,
):
    """Construct one signed effective forcing component on an image grid."""
    height, width = (int(shape[0]), int(shape[1]))
    if height < 2 or width < 2:
        raise ValueError("shape must be at least 2x2")
    x_axis = np.linspace(-1.0, 1.0, width, dtype=np.float64)
    y_axis = np.linspace(-1.0, 1.0, height, dtype=np.float64)
    component = _component_on_grid(
        x_axis,
        y_axis,
        center=center,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        theta=theta,
        alpha=alpha,
    )
    scale = 1.0
    if normalize:
        scale = unit_rms_scale_numpy(
            center=center,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            theta=theta,
            alpha=alpha,
            grid_size=rms_grid_size,
            epsilon=rms_epsilon,
        )
        component = component / scale
    if not np.isfinite(component).all():
        raise ValueError("forcing component contains NaN or Inf")
    return component.astype(np.float64, copy=False), float(scale)


def forcing_negative_energy_fraction(component) -> float:
    """Fraction of squared forcing energy contributed by the negative lobe."""
    array = np.asarray(component, dtype=np.float64)
    denominator = float(np.sum(array * array))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return 0.0
    negative = np.minimum(array, 0.0)
    return float(np.sum(negative * negative) / denominator)
