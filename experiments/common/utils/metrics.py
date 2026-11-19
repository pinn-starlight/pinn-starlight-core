"""E1-E4 shared metrics with raw scientific arrays and auditable detection modes.

Current protocol
----------------
* scientific residual = observed - background_pred, kept as raw float;
* MAE/RMSE/PSNR/SSIM and aperture photometry never clip scientific arrays;
* formal star detection uses the raw residual, including negative values;
* debug-only detector views retain ``nonnegative`` and legacy ``[0,1]`` modes;
* the frozen reference catalogue is built once from the Stage-10 native-
  background-flattened camera reference (legacy variable name: ``clean_true``).
"""

from __future__ import annotations

import math

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

import experiments.common.utils.scores as scores

STAR_COUNT_LIMIT = 2000
EVALUATION_SCHEMA_VERSION = 4

DETECTION_MODES = ("raw", "nonnegative", "legacy_01")
FORMAL_DETECTION_MODE = "raw"
REFERENCE_DETECTION_MODE = "raw"
DETECTION_IMAGE_RULES = {
    "raw": "raw_unclipped",
    "nonnegative": "lower_clip_zero_only_no_upper_clip",
    "legacy_01": "legacy_clip_0_1_debug_only",
}
DETECTION_IMAGE_RULE = DETECTION_IMAGE_RULES[FORMAL_DETECTION_MODE]

MATCHED_STAR_FLUX_METRIC = "matched_star_mean_relative_flux_error"

DEFAULT_APERTURE_RADIUS = 4.5
DEFAULT_ANNULUS_INNER_RADIUS = 6.0
DEFAULT_ANNULUS_OUTER_RADIUS = 8.0


def default_photometry_config() -> dict:
    """Return the frozen Stage-10 raw-array aperture-photometry definition."""
    return {
        "method": "local_background_subtracted_circular_aperture_on_raw_array",
        "aperture_radius": DEFAULT_APERTURE_RADIUS,
        "annulus_inner_radius": DEFAULT_ANNULUS_INNER_RADIUS,
        "annulus_outer_radius": DEFAULT_ANNULUS_OUTER_RADIUS,
        "source_array_reference": "clean_true_native_background_flattened_raw_float",
        "source_array_prediction": "residual_pred_raw_float",
        "uses_detection_image_flux": False,
        "upper_clipping": False,
    }


def build_detection_image(raw_image, mode=FORMAL_DETECTION_MODE) -> np.ndarray:
    """Build a detector-only view without changing the scientific residual.

    ``raw`` is formal. ``nonnegative`` and ``legacy_01`` are validation/debug only.
    """
    raw = scores.get_image(raw_image, clip=False)
    mode = str(mode)
    if mode not in DETECTION_MODES:
        raise ValueError(f"unknown detection mode: {mode}; expected one of {DETECTION_MODES}")
    if mode == "raw":
        return raw.astype(np.float32, copy=False)
    if mode == "nonnegative":
        return np.maximum(raw, 0.0).astype(np.float32, copy=False)
    return np.clip(raw, 0.0, 1.0).astype(np.float32, copy=False)


def extract_stars(
    detection_image,
    threshold=0.03,
    fwhm=3.0,
    max_stars=STAR_COUNT_LIMIT,
) -> list[dict]:
    """Detect point sources from a detector-only image.

    ``detection_flux`` is a detector diagnostic only and is not used for the
    paper flux metric.
    """
    image = scores.get_image(detection_image, clip=False)
    if threshold <= 0:
        raise ValueError("threshold 必须大于 0")
    if fwhm <= 0:
        raise ValueError("fwhm 必须大于 0")
    if max_stars <= 0:
        raise ValueError("max_stars 必须大于 0")

    _, background_median, _ = sigma_clipped_stats(image, sigma=3.0)
    finder = DAOStarFinder(
        fwhm=float(fwhm),
        threshold=float(threshold),
        exclude_border=False,
    )
    detections = finder(image - float(background_median))
    if detections is None or len(detections) == 0:
        return []

    detections.sort("peak")
    detections = detections[::-1][:max_stars]
    stars = []
    for row in detections:
        stars.append(
            {
                "x": float(row["x_centroid"]),
                "y": float(row["y_centroid"]),
                "detection_flux": float(row["flux"]),
            }
        )
    return stars


def detect_stars_with_cap(
    detection_image,
    threshold=0.03,
    fwhm=3.0,
    max_stars=STAR_COUNT_LIMIT,
):
    """Detect up to max_stars and explicitly record whether the cap was exceeded."""
    stars = extract_stars(
        detection_image,
        threshold=threshold,
        fwhm=fwhm,
        max_stars=int(max_stars) + 1,
    )
    capped = len(stars) > int(max_stars)
    return stars[: int(max_stars)], bool(capped)


def measure_raw_aperture_flux(
    raw_image,
    x,
    y,
    *,
    aperture_radius=DEFAULT_APERTURE_RADIUS,
    annulus_inner_radius=DEFAULT_ANNULUS_INNER_RADIUS,
    annulus_outer_radius=DEFAULT_ANNULUS_OUTER_RADIUS,
) -> float:
    """Local-background-subtracted circular aperture flux on an unclipped array."""
    image = scores.get_image(raw_image, clip=False).astype(np.float64, copy=False)
    aperture_radius = float(aperture_radius)
    annulus_inner_radius = float(annulus_inner_radius)
    annulus_outer_radius = float(annulus_outer_radius)
    if not (0.0 < aperture_radius < annulus_inner_radius < annulus_outer_radius):
        raise ValueError(
            "photometry radii must satisfy 0 < aperture < annulus_inner < annulus_outer"
        )

    x = float(x)
    y = float(y)
    height, width = image.shape
    margin = int(math.ceil(annulus_outer_radius))
    xmin = max(0, int(math.floor(x)) - margin)
    xmax = min(width - 1, int(math.floor(x)) + margin)
    ymin = max(0, int(math.floor(y)) - margin)
    ymax = min(height - 1, int(math.floor(y)) + margin)
    if xmin > xmax or ymin > ymax:
        return float("nan")

    yy, xx = np.mgrid[ymin : ymax + 1, xmin : xmax + 1]
    radius_sq = (xx - x) ** 2 + (yy - y) ** 2
    aperture_mask = radius_sq <= aperture_radius**2
    annulus_mask = (radius_sq >= annulus_inner_radius**2) & (radius_sq <= annulus_outer_radius**2)
    cutout = image[ymin : ymax + 1, xmin : xmax + 1]
    if not np.any(aperture_mask):
        return float("nan")

    local_background = float(np.median(cutout[annulus_mask])) if np.any(annulus_mask) else 0.0
    flux = np.sum(cutout[aperture_mask] - local_background, dtype=np.float64)
    return float(flux)


def attach_raw_aperture_fluxes(raw_image, stars, photometry=None) -> list[dict]:
    """Attach paper photometry to detected stars using the raw source array."""
    config = dict(default_photometry_config() if photometry is None else photometry)
    if config.get("uses_detection_image_flux") is not False:
        raise ValueError("raw-array photometry must not use detection-image flux")
    if config.get("upper_clipping") is not False:
        raise ValueError("raw-array photometry must not upper-clip raw arrays")

    output = []
    for star in stars:
        item = dict(star)
        item["raw_aperture_flux"] = measure_raw_aperture_flux(
            raw_image,
            item["x"],
            item["y"],
            aperture_radius=float(config["aperture_radius"]),
            annulus_inner_radius=float(config["annulus_inner_radius"]),
            annulus_outer_radius=float(config["annulus_outer_radius"]),
        )
        output.append(item)
    return output


def build_star_catalog(
    raw_image,
    *,
    threshold=0.03,
    fwhm=3.0,
    max_stars=STAR_COUNT_LIMIT,
    photometry=None,
    detection_mode=FORMAL_DETECTION_MODE,
):
    """Detect in the requested detector view, then measure flux on the raw array."""
    detection_image = build_detection_image(raw_image, mode=detection_mode)
    stars, capped = detect_stars_with_cap(
        detection_image,
        threshold=threshold,
        fwhm=fwhm,
        max_stars=max_stars,
    )
    stars = attach_raw_aperture_fluxes(raw_image, stars, photometry=photometry)
    return stars, capped


def star_metrics(
    residual_pred,
    *,
    threshold=0.03,
    fwhm=3.0,
    matching_radius=3.0,
    reference_stars=None,
    reference_count_capped=False,
    photometry=None,
    detection_mode=FORMAL_DETECTION_MODE,
):
    """Compute star detection metrics and matched-star raw-array flux error."""
    if reference_stars is None:
        raise ValueError("缺少冻结的 clean_true 星点参考目录")
    raw_residual = scores.get_image(residual_pred, clip=False)
    predicted, predicted_capped = build_star_catalog(
        raw_residual,
        threshold=float(threshold),
        fwhm=float(fwhm),
        max_stars=STAR_COUNT_LIMIT,
        photometry=photometry,
        detection_mode=detection_mode,
    )
    reference = list(reference_stars)
    matches = _match_stars(reference, predicted, matching_radius)

    true_positive = len(matches)
    false_positive = len(predicted) - true_positive
    false_negative = len(reference) - true_positive
    if not reference and not predicted:
        precision = recall = f1 = 1.0
    else:
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    relative_flux_errors = []
    for reference_index, predicted_index in matches:
        true_flux = float(reference[reference_index].get("raw_aperture_flux", np.nan))
        pred_flux = float(predicted[predicted_index].get("raw_aperture_flux", np.nan))
        if not np.isfinite(true_flux) or not np.isfinite(pred_flux):
            continue
        if abs(true_flux) <= 1e-8:
            continue
        relative_flux_errors.append(abs(pred_flux - true_flux) / abs(true_flux))

    flux_metric = float(np.mean(relative_flux_errors)) if relative_flux_errors else float("nan")

    return {
        "star_precision": float(precision),
        "star_recall": float(recall),
        "star_f1": float(f1),
        MATCHED_STAR_FLUX_METRIC: flux_metric,
        "matched_star_count": int(len(matches)),
        "matched_star_flux_count": int(len(relative_flux_errors)),
        "reference_star_count": int(len(reference)),
        "predicted_star_count": int(len(predicted)),
        "reference_star_count_capped": bool(reference_count_capped),
        "predicted_star_count_capped": bool(predicted_capped),
        "prediction_detection_mode": str(detection_mode),
    }


def compare_star_detection_modes(
    residual_pred,
    *,
    threshold=0.03,
    fwhm=3.0,
    matching_radius=3.0,
    reference_stars=None,
    reference_count_capped=False,
    photometry=None,
    modes=DETECTION_MODES,
):
    """Compare detector preprocessing on one raw residual for validation/debug."""
    results = {}
    for mode in modes:
        mode = str(mode)
        results[mode] = star_metrics(
            residual_pred,
            threshold=threshold,
            fwhm=fwhm,
            matching_radius=matching_radius,
            reference_stars=reference_stars,
            reference_count_capped=reference_count_capped,
            photometry=photometry,
            detection_mode=mode,
        )
    return results


def evaluate_synthetic(sample: dict, prediction: dict):
    """Evaluate one synthetic prediction under the raw-detection protocol."""
    background_true = sample["background_true"]
    clean_true = sample["clean_true"]
    background_pred = prediction["background_pred"]
    residual_pred = prediction["residual_pred"]

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
    if int(reference_info.get("evaluation_schema_version", 0)) != EVALUATION_SCHEMA_VERSION:
        raise ValueError(
            "synthetic sample star_reference is not evaluation schema v4; "
            "regenerate the formal synthetic dataset before validation/formal test"
        )
    if reference_info.get("catalog_source") != "native_background_flattened_camera_reference":
        raise ValueError(
            "formal Stage-10 star_reference must be built from the native-background-"
            "flattened camera reference only"
        )
    detection_info = reference_info.get("detection_image", {})
    if detection_info.get("reference_mode") != REFERENCE_DETECTION_MODE:
        raise ValueError("formal star_reference reference detection mode mismatch")
    expected_reference_rule = DETECTION_IMAGE_RULES[REFERENCE_DETECTION_MODE]
    if detection_info.get("reference_rule") != expected_reference_rule:
        raise ValueError("formal star_reference reference detection rule mismatch")

    photometry = reference_info.get("photometry", {})
    expected_photometry = default_photometry_config()
    for key, expected in expected_photometry.items():
        if photometry.get(key) != expected:
            raise ValueError(f"formal star_reference photometry mismatch: {key}")

    reference_stars = reference_info.get("stars")
    if reference_stars is None:
        raise ValueError(
            "synthetic sample is missing frozen metadata.star_reference.stars; "
            "regenerate the formal synthetic dataset"
        )

    result.update(
        star_metrics(
            residual_pred,
            threshold=float(reference_info["threshold"]),
            fwhm=float(reference_info["fwhm"]),
            matching_radius=float(reference_info["matching_radius"]),
            reference_stars=reference_stars,
            reference_count_capped=bool(reference_info.get("count_capped", False)),
            photometry=photometry,
            detection_mode=FORMAL_DETECTION_MODE,
        )
    )
    return result


def evaluate_real(observed, background_pred, residual_pred):
    """Return real-image descriptive statistics without unavailable truth."""
    observed = scores.get_image(observed, clip=False)
    background_pred = scores.get_image(background_pred, clip=False)
    residual_pred = scores.get_image(residual_pred, clip=False)
    total_variation = np.mean(np.abs(np.diff(background_pred, axis=0)))
    total_variation += np.mean(np.abs(np.diff(background_pred, axis=1)))
    detection_image = build_detection_image(residual_pred, mode=FORMAL_DETECTION_MODE)
    stars, count_capped = detect_stars_with_cap(detection_image)
    return {
        "observed_mean": float(observed.mean()),
        "background_mean": float(background_pred.mean()),
        "background_total_variation": float(total_variation),
        "residual_mean": float(residual_pred.mean()),
        "residual_std": float(residual_pred.std()),
        "residual_negative_fraction": float(np.mean(residual_pred < 0.0)),
        "residual_above_one_fraction": float(np.mean(residual_pred > 1.0)),
        "detected_star_count": int(len(stars)),
        "detected_star_count_capped": bool(count_capped),
    }


def center_error(predicted_x, predicted_y, true_x, true_y):
    return float(
        math.hypot(
            float(predicted_x) - float(true_x),
            float(predicted_y) - float(true_y),
        )
    )


def _match_stars(reference_stars, predicted_stars, max_distance):
    possible_matches = []
    for true_index, true_star in enumerate(reference_stars):
        for pred_index, pred_star in enumerate(predicted_stars):
            dx = true_star["x"] - pred_star["x"]
            dy = true_star["y"] - pred_star["y"]
            distance = math.sqrt(dx * dx + dy * dy)
            if distance <= max_distance:
                possible_matches.append(
                    {
                        "distance": distance,
                        "true_index": true_index,
                        "pred_index": pred_index,
                    }
                )

    possible_matches.sort(key=lambda items: items["distance"])
    matched_true = set()
    matched_pred = set()
    matches = []
    for item in possible_matches:
        true_index = item["true_index"]
        pred_index = item["pred_index"]
        if true_index in matched_true or pred_index in matched_pred:
            continue
        matched_true.add(true_index)
        matched_pred.add(pred_index)
        matches.append((true_index, pred_index))
    return matches
