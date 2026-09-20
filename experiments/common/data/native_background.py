"""Deterministic Stage-10 native-background flattening for dataset construction.

This preprocessing is intentionally independent of every evaluated method. It
removes only the low-frequency sky level already present in the camera-acquired
reference before synthetic skyglow is injected. The estimated native background
is never passed to PINN, cosine-domain LS, Gaussian, or U-Net.
"""

from __future__ import annotations

import math

import numpy as np
from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils.background import Background2D, MedianBackground

from experiments.common.utils import metrics


def mesh_size_for_shape(shape, *, divisor=16, minimum=64, maximum=256) -> int:
    height, width = (int(shape[0]), int(shape[1]))
    if height < 2 or width < 2:
        raise ValueError("image must be at least 2x2")
    value = int(round(min(height, width) / float(divisor)))
    value = int(np.clip(value, int(minimum), int(maximum)))
    # Background2D requires the box to fit the image. Formal images are larger
    # than 64 px; this fallback keeps small unit tests well defined.
    return max(2, min(value, height, width))


def estimate_native_background(image, config: dict):
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("reference image must be a finite 2-D array")
    mesh = mesh_size_for_shape(
        array.shape,
        divisor=int(config["mesh_divisor"]),
        minimum=int(config["mesh_min"]),
        maximum=int(config["mesh_max"]),
    )
    sigma_clip = SigmaClip(
        sigma=float(config["sigma_clip"]),
        maxiters=int(config["sigma_clip_maxiters"]),
    )
    estimator = Background2D(
        array.astype(np.float64, copy=False),
        box_size=(mesh, mesh),
        filter_size=(int(config["filter_size"]), int(config["filter_size"])),
        sigma_clip=sigma_clip,
        bkg_estimator=MedianBackground(),
    )
    background = np.asarray(estimator.background, dtype=np.float32)
    if background.shape != array.shape or not np.isfinite(background).all():
        raise ValueError("native-background estimator returned an invalid array")
    return background, {
        "method": "photutils.Background2D_sigma_clipped_median",
        "mesh_size": mesh,
        "sigma_clip": float(config["sigma_clip"]),
        "sigma_clip_maxiters": int(config["sigma_clip_maxiters"]),
        "filter_size": int(config["filter_size"]),
        "multiplicative_rescaling": False,
        "scientific_clipping": False,
    }


def flatten_reference(image, config: dict):
    reference = np.asarray(image, dtype=np.float32)
    background, estimator_info = estimate_native_background(reference, config)
    clean = (reference - background).astype(np.float32)
    if not np.isfinite(clean).all():
        raise ValueError("native-background flattening produced NaN or Inf")
    return clean, background, estimator_info


def audit_flattening(reference, clean, background, config: dict) -> dict:
    """Audit sky-zeroing and local-background-subtracted star preservation."""
    reference = np.asarray(reference, dtype=np.float32)
    clean = np.asarray(clean, dtype=np.float32)
    background = np.asarray(background, dtype=np.float32)
    if reference.shape != clean.shape or reference.shape != background.shape:
        raise ValueError("preprocessing audit arrays must share a shape")

    _, reference_sky_median, _ = sigma_clipped_stats(reference, sigma=3.0)
    _, clean_sky_median, _ = sigma_clipped_stats(clean, sigma=3.0)

    photometry = metrics.default_photometry_config()
    before_stars, _ = metrics.build_star_catalog(
        reference,
        threshold=float(config["star_threshold"]),
        fwhm=float(config["star_fwhm"]),
        max_stars=int(config["audit_max_stars"]),
        photometry=photometry,
        detection_mode=metrics.REFERENCE_DETECTION_MODE,
    )
    after_stars, _ = metrics.build_star_catalog(
        clean,
        threshold=float(config["star_threshold"]),
        fwhm=float(config["star_fwhm"]),
        max_stars=int(config["audit_max_stars"]),
        photometry=photometry,
        detection_mode=metrics.REFERENCE_DETECTION_MODE,
    )
    matches = _nearest_matches(before_stars, after_stars, max_distance=2.0)
    centroid_shifts = []
    flux_errors = []
    for before_index, after_index in matches:
        before = before_stars[before_index]
        after = after_stars[after_index]
        centroid_shifts.append(
            math.hypot(float(before["x"]) - float(after["x"]), float(before["y"]) - float(after["y"]))
        )
        before_flux = metrics.measure_raw_aperture_flux(reference, before["x"], before["y"])
        after_flux = metrics.measure_raw_aperture_flux(clean, before["x"], before["y"])
        if np.isfinite(before_flux) and np.isfinite(after_flux) and abs(before_flux) > 1e-8:
            flux_errors.append(abs(after_flux - before_flux) / abs(before_flux))

    median_shift = float(np.median(centroid_shifts)) if centroid_shifts else float("nan")
    median_flux_error = float(np.median(flux_errors)) if flux_errors else float("nan")
    p95_flux_error = float(np.percentile(flux_errors, 95.0)) if flux_errors else float("nan")
    sky_ok = abs(float(clean_sky_median)) < float(config["audit_sky_median_abs_max"])
    centroid_ok = bool(centroid_shifts) and median_shift < float(config["audit_centroid_shift_median_max"])
    median_flux_ok = (
        bool(flux_errors)
        and median_flux_error < float(config["audit_flux_error_median_max"])
    )
    p95_flux_ok = (
        bool(flux_errors)
        and p95_flux_error < float(config["audit_flux_error_p95_max"])
    )
    flux_ok = bool(median_flux_ok and p95_flux_ok)
    identity_error = float(np.max(np.abs(reference - background - clean)))
    identity_ok = identity_error < 1e-6
    return {
        "reference_sigma_clipped_median": float(reference_sky_median),
        "clean_sigma_clipped_median": float(clean_sky_median),
        "matched_star_count": len(matches),
        "median_centroid_shift_px": median_shift,
        "median_local_bg_subtracted_flux_relative_error": median_flux_error,
        "p95_local_bg_subtracted_flux_relative_error": p95_flux_error,
        "reference_minus_background_minus_clean_max_abs": identity_error,
        "thresholds": {
            "clean_sky_median_abs_max": float(config["audit_sky_median_abs_max"]),
            "centroid_shift_median_max_px": float(config["audit_centroid_shift_median_max"]),
            "flux_error_median_max": float(config["audit_flux_error_median_max"]),
            "flux_error_p95_max": float(config["audit_flux_error_p95_max"]),
            "identity_max_abs": 1e-6,
        },
        "photometry_checks": {
            "median_flux_error": bool(median_flux_ok),
            "p95_flux_error": bool(p95_flux_ok),
        },
        "checks": {
            "sky_zeroing": bool(sky_ok),
            "centroid_preservation": bool(centroid_ok),
            "local_photometry_preservation": bool(flux_ok),
            "additive_identity": bool(identity_ok),
        },
        "pass": bool(sky_ok and centroid_ok and flux_ok and identity_ok),
    }


def _nearest_matches(reference, predicted, max_distance: float):
    candidates = []
    for i, first in enumerate(reference):
        for j, second in enumerate(predicted):
            distance = math.hypot(float(first["x"]) - float(second["x"]), float(first["y"]) - float(second["y"]))
            if distance <= float(max_distance):
                candidates.append((distance, i, j))
    candidates.sort()
    used_i, used_j, matches = set(), set(), []
    for _, i, j in candidates:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        matches.append((i, j))
    return matches
