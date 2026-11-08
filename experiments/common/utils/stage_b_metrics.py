"""Stage10-B metrics extending Stage10-A with faint-star and protected-structure checks."""
from __future__ import annotations

import math

import numpy as np

from experiments.common.utils import metrics, scores
from experiments.configs.stage_b_experiment import STAGE_B_CONFIG


def evaluate_stage_b(sample: dict, prediction: dict) -> dict:
    background_true = np.asarray(sample["background_true"], dtype=np.float32)
    clean_true = np.asarray(sample["clean_true"], dtype=np.float32)
    base_clean = np.asarray(sample["base_clean"], dtype=np.float32)
    protected_true = np.asarray(sample["protected_true"], dtype=np.float32)
    background_pred = np.asarray(prediction["background_pred"], dtype=np.float32)
    residual_pred = np.asarray(prediction["residual_pred"], dtype=np.float32)

    result = {
        "bg_mae": scores.mae(background_pred, background_true),
        "bg_rmse": scores.rmse(background_pred, background_true),
        "bg_psnr": scores.psnr(background_pred, background_true),
        "bg_ssim": scores.ssim(background_pred, background_true),
        "residual_mae": scores.mae(residual_pred, clean_true),
        "residual_rmse": scores.rmse(residual_pred, clean_true),
        "residual_psnr": scores.psnr(residual_pred, clean_true),
        "residual_ssim": scores.ssim(residual_pred, clean_true),
    }

    reference_info = sample.get("metadata", {}).get("star_reference", {})
    if int(reference_info.get("evaluation_schema_version", 0)) != metrics.EVALUATION_SCHEMA_VERSION:
        raise ValueError("Stage-B star reference schema mismatch")
    reference_stars = reference_info.get("stars")
    if reference_stars is None:
        raise ValueError("Stage-B sample has no frozen star catalogue")
    photometry = reference_info.get("photometry", metrics.default_photometry_config())
    result.update(
        metrics.star_metrics(
            residual_pred,
            threshold=float(reference_info["threshold"]),
            fwhm=float(reference_info["fwhm"]),
            matching_radius=float(reference_info["matching_radius"]),
            reference_stars=reference_stars,
            reference_count_capped=bool(reference_info.get("count_capped", False)),
            photometry=photometry,
            detection_mode=metrics.FORMAL_DETECTION_MODE,
        )
    )

    result.update(_faint_star_metrics(residual_pred, reference_info))
    result.update(_protected_structure_metrics(base_clean, protected_true, residual_pred))
    return result


def _faint_star_metrics(residual_pred, reference_info):
    reference = list(reference_info["stars"])
    finite_positive = [
        star for star in reference
        if np.isfinite(float(star.get("raw_aperture_flux", np.nan)))
        and float(star.get("raw_aperture_flux", 0.0)) > 0.0
    ]
    finite_positive.sort(key=lambda star: float(star["raw_aperture_flux"]))
    fraction = float(STAGE_B_CONFIG["evaluation"]["faint_star_fraction"])
    faint_count = max(1, int(math.ceil(len(finite_positive) * fraction))) if finite_positive else 0
    faint_reference = finite_positive[:faint_count]

    predicted, _ = metrics.build_star_catalog(
        residual_pred,
        threshold=float(reference_info["threshold"]),
        fwhm=float(reference_info["fwhm"]),
        max_stars=metrics.STAR_COUNT_LIMIT,
        photometry=reference_info.get("photometry", metrics.default_photometry_config()),
        detection_mode=metrics.FORMAL_DETECTION_MODE,
    )
    matches = _greedy_matches(
        faint_reference,
        predicted,
        max_distance=float(reference_info["matching_radius"]),
    )
    recall = float(len(matches) / faint_count) if faint_count else float("nan")
    return {
        "faint_star_reference_count": int(faint_count),
        "faint_star_matched_count": int(len(matches)),
        "faint_star_recall": recall,
    }


def _protected_structure_metrics(base_clean, protected_true, residual_pred):
    threshold_fraction = float(STAGE_B_CONFIG["evaluation"]["protected_mask_threshold_fraction"])
    peak = float(np.max(protected_true))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError("Stage-B protected_true must have positive finite peak")
    mask = protected_true >= (threshold_fraction * peak)
    recovered_structure = np.asarray(residual_pred, dtype=np.float64) - np.asarray(base_clean, dtype=np.float64)
    protected = np.asarray(protected_true, dtype=np.float64)
    region_mae = float(np.mean(np.abs(recovered_structure[mask] - protected[mask])))
    denom = float(np.sum(protected * protected))
    projection_ratio = float(np.sum(recovered_structure * protected) / max(denom, 1e-20))
    return {
        "protected_region_mae": region_mae,
        "protected_projection_ratio": projection_ratio,
        "protected_projection_abs_error": float(abs(projection_ratio - 1.0)),
        "protected_mask_fraction": float(np.mean(mask)),
    }


def _greedy_matches(reference_stars, predicted_stars, max_distance):
    candidates = []
    for i, first in enumerate(reference_stars):
        for j, second in enumerate(predicted_stars):
            distance = math.hypot(float(first["x"])-float(second["x"]), float(first["y"])-float(second["y"]))
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
