"""Synthetic light-pollution backgrounds used by the formal experiments.

The generators intentionally do *not* reuse the Stage-10 PINN effective-forcing formula.  They
provide smooth, interpretable 2-D backgrounds together with the exact
parameters used to create them, which keeps the synthetic truth useful for
E3 without creating an inverse-crime baseline.

All spatial parameters live in the same normalized coordinate system used by
``coordinate_grid``: x,y in [-1, 1].  ``scale`` is a characteristic spatial
scale in that coordinate system.  ``intensity`` is the peak background level.
"""

from __future__ import annotations

import math

import numpy as np


BACKGROUND_TYPES = (
    "single_radial",
    "single_eccentric",
    "multi_source",
)


def generate_background(
    height: int,
    width: int,
    background_type: str,
    intensity: float,
    scale: float,
    seed: int,
):
    """Return ``(background_true, generation_truth)`` for one synthetic case."""
    height = int(height)
    width = int(width)
    intensity = float(intensity)
    scale = float(scale)
    if height <= 1 or width <= 1:
        raise ValueError("height and width must both be greater than one")
    if background_type not in BACKGROUND_TYPES:
        raise ValueError(f"unknown background_type: {background_type}")
    if not 0.0 < intensity < 1.0:
        raise ValueError("intensity must lie in (0, 1)")
    if scale <= 0.0:
        raise ValueError("scale must be positive")

    rng = np.random.default_rng(int(seed))
    x_axis = np.linspace(-1.0, 1.0, width, dtype=np.float64)
    y_axis = np.linspace(-1.0, 1.0, height, dtype=np.float64)
    yy, xx = np.meshgrid(y_axis, x_axis, indexing="ij")

    if background_type == "single_radial":
        center = _sample_center(rng, width, height)
        raw = _radial_profile(xx, yy, center["x"], center["y"], scale)
        sources = [
            {
                "profile": "radial_exponential",
                "amplitude_weight": 1.0,
                "scale": scale,
                "scale_x": scale,
                "scale_y": scale,
                "axis_ratio": 1.0,
                "theta": 0.0,
                "center": center,
            }
        ]
    elif background_type == "single_eccentric":
        center = _sample_center(rng, width, height)
        source = _sample_eccentric_source(rng, center, scale)
        raw = _eccentric_profile(xx, yy, source)
        sources = [source]
    else:
        centers = _sample_two_centers(rng, width, height)
        weights = rng.dirichlet(np.array([2.0, 2.0], dtype=np.float64))
        sources = []
        raw = np.zeros((height, width), dtype=np.float64)
        for index, center in enumerate(centers):
            local_scale = scale * float(rng.uniform(0.78, 1.08))
            source = _sample_eccentric_source(rng, center, local_scale)
            source["amplitude_weight"] = float(weights[index])
            raw += float(weights[index]) * _eccentric_profile(xx, yy, source)
            sources.append(source)

    peak = float(np.max(raw))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError("generated background has an invalid peak")
    background = (intensity * raw / peak).astype(np.float32)

    source_centers = []
    for source in sources:
        center = dict(source["center"])
        source_centers.append(center)
        source["amplitude_coefficient_after_normalization"] = (
            intensity * float(source.get("amplitude_weight", 1.0)) / peak
        )

    truth = {
        "background_type": background_type,
        "coordinate_system": "normalized_xy_minus1_to_1",
        "scale_definition": "characteristic spatial scale in normalized [-1,1]^2 coordinates",
        "intensity_definition": "peak value of the combined background_true",
        "intensity": intensity,
        "scale": scale,
        "source_centers": source_centers,
        "sources": sources,
        "raw_combined_peak_before_normalization": peak,
        "background_min": float(background.min()),
        "background_max": float(background.max()),
    }
    return background, truth


def _sample_center(rng, width, height):
    # Treat the centre as the centre of the dominant skyglow lobe rather than
    # the physical lamp position.  Keeping it inside the field makes the E3
    # centre-recovery quantity meaningful.
    x = float(rng.uniform(-0.55, 0.55))
    y = float(rng.uniform(0.30, 0.82))
    return _center_record(x, y, width, height)


def _sample_two_centers(rng, width, height):
    left = _center_record(
        float(rng.uniform(-0.72, -0.08)),
        float(rng.uniform(0.28, 0.82)),
        width,
        height,
    )
    right = _center_record(
        float(rng.uniform(0.08, 0.72)),
        float(rng.uniform(0.28, 0.82)),
        width,
        height,
    )
    return [left, right]


def _center_record(x, y, width, height):
    return {
        "x": float(x),
        "y": float(y),
        "x_pixel": float((x + 1.0) * 0.5 * (width - 1)),
        "y_pixel": float((y + 1.0) * 0.5 * (height - 1)),
    }


def _radial_profile(xx, yy, center_x, center_y, scale):
    radius = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    # Exponential falloff gives a smooth extended lobe while remaining
    # deliberately different from the evaluated q(r; alpha) * Gaussian forcing family.
    return np.exp(-radius / max(float(scale), 1e-6))


def _sample_eccentric_source(rng, center, scale):
    axis_ratio = float(rng.uniform(0.45, 0.72))
    theta = float(rng.uniform(-0.5 * math.pi, 0.5 * math.pi))
    return {
        "profile": "elliptical_exponential",
        "amplitude_weight": 1.0,
        "scale": float(scale),
        "scale_x": float(scale),
        "scale_y": float(scale * axis_ratio),
        "axis_ratio": axis_ratio,
        "theta": theta,
        "center": dict(center),
    }


def _eccentric_profile(xx, yy, source):
    center = source["center"]
    dx = xx - float(center["x"])
    dy = yy - float(center["y"])
    theta = float(source["theta"])
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    x_rot = cos_theta * dx + sin_theta * dy
    y_rot = -sin_theta * dx + cos_theta * dy
    scale_x = max(float(source["scale_x"]), 1e-6)
    scale_y = max(float(source["scale_y"]), 1e-6)
    elliptical_radius = np.sqrt((x_rot / scale_x) ** 2 + (y_rot / scale_y) ** 2)
    return np.exp(-elliptical_radius)
