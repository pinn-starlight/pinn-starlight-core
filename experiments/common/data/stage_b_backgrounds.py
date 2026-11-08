"""Method-independent Stage10-B stress-test scene generator.

The generator deliberately does not reuse the evaluated Gaussian filter,
Screened-Poisson forcing family, cosine-domain LS solver, or PINN.  It creates
smooth pollution fields whose spatial frequencies overlap a separate smooth
"protected" structure that belongs to the clean target and must not be removed.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

SCENE_TYPES = (
    "edge_asymmetric",
    "multiscale_sources",
    "irregular_smooth",
    "protected_overlap",
)

INTENSITY_LEVELS = {"low": 0.10, "high": 0.20}
PROTECTED_PEAK = {
    "edge_asymmetric": 0.025,
    "multiscale_sources": 0.030,
    "irregular_smooth": 0.030,
    "protected_overlap": 0.055,
}


def generate_stage_b_scene(
    height: int,
    width: int,
    scene_type: str,
    intensity_level: str,
    seed: int,
):
    height, width = int(height), int(width)
    if height <= 1 or width <= 1:
        raise ValueError("height and width must exceed one")
    if scene_type not in SCENE_TYPES:
        raise ValueError(f"unknown Stage-B scene_type: {scene_type}")
    if intensity_level not in INTENSITY_LEVELS:
        raise ValueError(f"unknown Stage-B intensity level: {intensity_level}")
    rng = np.random.default_rng(int(seed))
    y = np.linspace(-1.0, 1.0, height, dtype=np.float64)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float64)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    if scene_type == "edge_asymmetric":
        pollution_raw, pollution_meta = _edge_asymmetric(xx, yy, rng)
    elif scene_type == "multiscale_sources":
        pollution_raw, pollution_meta = _multiscale_sources(xx, yy, rng)
    elif scene_type == "irregular_smooth":
        pollution_raw, pollution_meta = _irregular_smooth(height, width, xx, yy, rng)
    else:
        pollution_raw, pollution_meta = _protected_overlap_pollution(height, width, xx, yy, rng)

    protected_raw, protected_meta = _protected_structure(xx, yy, rng, scene_type)
    pollution = _normalize_nonnegative(pollution_raw) * float(INTENSITY_LEVELS[intensity_level])
    protected = _normalize_nonnegative(protected_raw) * float(PROTECTED_PEAK[scene_type])

    overlap = spectral_overlap_coefficient(pollution, protected)
    truth = {
        "scene_type": scene_type,
        "intensity_level": intensity_level,
        "pollution_peak": float(INTENSITY_LEVELS[intensity_level]),
        "protected_peak": float(PROTECTED_PEAK[scene_type]),
        "coordinate_system": "normalized_xy_minus1_to_1",
        "pollution_generator": pollution_meta,
        "protected_generator": protected_meta,
        "spectral_overlap_coefficient": float(overlap),
        "generator_independence": (
            "does_not_reuse_gaussian_filter_screened_poisson_forcing_or_pinn"
        ),
    }
    return pollution.astype(np.float32), protected.astype(np.float32), truth


def _edge_asymmetric(xx, yy, rng):
    # Dominant source centre lies outside the image.  A soft slanted horizon term
    # breaks radial symmetry without introducing high-frequency noise.
    cx = float(rng.choice([-1.0, 1.0]) * rng.uniform(1.15, 1.55))
    cy = float(rng.uniform(0.25, 1.25))
    scale = float(rng.uniform(0.50, 0.82))
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    lobe = np.exp(-rr / scale)
    angle = float(rng.uniform(-0.8, 0.8))
    nx, ny = math.cos(angle), math.sin(angle)
    offset = float(rng.uniform(-0.35, 0.20))
    softness = float(rng.uniform(0.18, 0.30))
    signed_distance = nx * xx + ny * yy - offset
    horizon = 1.0 / (1.0 + np.exp(-signed_distance / softness))
    raw = 0.75 * lobe + 0.25 * horizon
    meta = {
        "family": "external_exponential_plus_soft_slanted_horizon",
        "external_center": [cx, cy],
        "scale": scale,
        "angle": angle,
        "softness": softness,
    }
    return raw, meta


def _multiscale_sources(xx, yy, rng):
    cx1 = float(rng.uniform(-0.55, 0.55))
    cy1 = float(rng.uniform(0.10, 0.85))
    cx2 = float(rng.uniform(-0.85, 0.85))
    cy2 = float(rng.uniform(-0.15, 0.85))
    broad_scale = float(rng.uniform(0.70, 1.05))
    medium_scale = float(rng.uniform(0.20, 0.36))
    r1 = np.sqrt((xx - cx1) ** 2 + (yy - cy1) ** 2)
    # Generalized Cauchy/Moffat-like medium component, not Gaussian.
    theta = float(rng.uniform(-math.pi / 2, math.pi / 2))
    c, s = math.cos(theta), math.sin(theta)
    dx, dy = xx - cx2, yy - cy2
    xr, yr = c * dx + s * dy, -s * dx + c * dy
    ratio = float(rng.uniform(0.45, 0.75))
    r2sq = (xr / medium_scale) ** 2 + (yr / (medium_scale * ratio)) ** 2
    broad = np.exp(-r1 / broad_scale)
    medium = 1.0 / (1.0 + r2sq) ** 1.4
    weight = float(rng.uniform(0.30, 0.48))
    raw = (1.0 - weight) * broad + weight * medium
    meta = {
        "family": "broad_exponential_plus_anisotropic_generalized_cauchy",
        "broad_center": [cx1, cy1],
        "medium_center": [cx2, cy2],
        "broad_scale": broad_scale,
        "medium_scale": medium_scale,
        "medium_axis_ratio": ratio,
        "theta": theta,
        "medium_weight": weight,
    }
    return raw, meta


def _irregular_smooth(height, width, xx, yy, rng):
    gh, gw = 5, 7
    controls = rng.uniform(0.0, 1.0, size=(gh, gw))
    # Deliberately low-resolution cubic control field, then smooth again.
    field = zoom(controls, (height / gh, width / gw), order=3, mode="reflect", prefilter=True)
    field = field[:height, :width]
    field = gaussian_filter(field, sigma=max(1.0, min(height, width) * 0.035), mode="reflect")
    angle = float(rng.uniform(-math.pi, math.pi))
    gradient = 0.5 + 0.5 * (math.cos(angle) * xx + math.sin(angle) * yy) / math.sqrt(2.0)
    raw = 0.72 * field + 0.28 * gradient
    meta = {
        "family": "cubic_coarse_control_field_plus_oblique_gradient",
        "control_grid": [gh, gw],
        "post_smooth_sigma_fraction": 0.035,
        "gradient_angle": angle,
        "control_values": controls.tolist(),
    }
    return raw, meta


def _protected_overlap_pollution(height, width, xx, yy, rng):
    irregular, meta = _irregular_smooth(height, width, xx, yy, rng)
    cx = float(rng.uniform(-1.35, 1.35))
    cy = float(rng.uniform(0.55, 1.35))
    scale = float(rng.uniform(0.38, 0.62))
    lobe = np.exp(-np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / scale)
    raw = 0.58 * irregular + 0.42 * lobe
    meta = {
        "family": "irregular_control_field_plus_external_lobe",
        "irregular": meta,
        "external_center": [cx, cy],
        "external_scale": scale,
    }
    return raw, meta


def _protected_structure(xx, yy, rng, scene_type):
    # Sinuous diffuse ridge + one broad diffuse knot.  These are synthetic
    # astrophysical-like structures belonging to the clean target, not pollution.
    phase = float(rng.uniform(-math.pi, math.pi))
    y0 = float(rng.uniform(-0.30, 0.25))
    slope = float(rng.uniform(-0.28, 0.28))
    curvature = float(rng.uniform(-0.20, 0.20))
    waviness = float(rng.uniform(0.08, 0.18))
    centerline = (
        y0
        + slope * xx
        + curvature * (xx**2 - 0.35)
        + waviness * np.sin(math.pi * (xx + 1.0) + phase)
    )
    width = float(
        rng.uniform(0.12, 0.22) if scene_type != "protected_overlap" else rng.uniform(0.16, 0.26)
    )
    ridge = np.exp(-0.5 * ((yy - centerline) / width) ** 2)
    taper_center = float(rng.uniform(-0.35, 0.35))
    taper_scale = float(rng.uniform(0.75, 1.05))
    ridge *= np.exp(-0.5 * ((xx - taper_center) / taper_scale) ** 4)
    kx = float(rng.uniform(-0.55, 0.55))
    ky = float(rng.uniform(-0.45, 0.45))
    kscale = float(rng.uniform(0.28, 0.48))
    knot = 1.0 / (1.0 + (((xx - kx) / kscale) ** 2 + ((yy - ky) / (kscale * 0.75)) ** 2)) ** 1.6
    raw = 0.75 * ridge + 0.25 * knot
    meta = {
        "family": "sinuous_diffuse_ridge_plus_broad_knot",
        "ridge_width": width,
        "ridge_y0": y0,
        "ridge_slope": slope,
        "ridge_curvature": curvature,
        "ridge_waviness": waviness,
        "ridge_phase": phase,
        "knot_center": [kx, ky],
        "knot_scale": kscale,
    }
    return raw, meta


def _normalize_nonnegative(field):
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 2 or not np.isfinite(field).all():
        raise ValueError("field must be finite 2-D")
    field = field - float(field.min())
    peak = float(field.max())
    if peak <= 1e-12:
        raise ValueError("generated field is constant")
    return field / peak


def radial_power_distribution(image, bins=64):
    arr = np.asarray(image, dtype=np.float64)
    arr = arr - float(np.mean(arr))
    spec = np.fft.fftshift(np.fft.fft2(arr))
    power = np.abs(spec) ** 2
    h, w = arr.shape
    yy, xx = np.mgrid[:h, :w]
    rr = np.sqrt(((yy - (h - 1) / 2) / max(h, 1)) ** 2 + ((xx - (w - 1) / 2) / max(w, 1)) ** 2)
    edges = np.linspace(0.0, float(rr.max()) + 1e-12, int(bins) + 1)
    idx = np.clip(np.digitize(rr.ravel(), edges) - 1, 0, int(bins) - 1)
    radial = np.bincount(idx, weights=power.ravel(), minlength=int(bins)).astype(np.float64)
    radial += 1e-30
    radial /= radial.sum()
    return radial


def spectral_overlap_coefficient(first, second, bins=64):
    p = radial_power_distribution(first, bins=bins)
    q = radial_power_distribution(second, bins=bins)
    return float(np.sum(np.sqrt(p * q)))
