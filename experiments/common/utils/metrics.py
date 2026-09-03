"""E1-E4 共用的论文评价指标。"""

import math

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

import experiments.common.utils.scores as scores

STAR_COUNT_LIMIT = 2000


def extract_stars(
    image,
    threshold = 0.1,
    fwhm = 3.0,
    max_stars = STAR_COUNT_LIMIT,
) -> list[dict]:
    """使用 photutils.DAOStarFinder 检测点源并返回统一格式的星点。"""
    image = scores.get_image(image)
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
                "flux": float(row["flux"]),
            }
        )
    return stars


def star_metrics(
    residual_pred,
    threshold = 0.1,
    matching_radius= 3,
    reference_stars=None,
):
    """以 clean_true 提取的星点作为固定参考，计算匹配和光通量误差。"""
    reference = reference_stars
    reference_capped = bool(reference is not None and len(reference) >= STAR_COUNT_LIMIT)
    if reference is None:
        raise ValueError("原图中没有星点")
    predicted, predicted_capped = _extract_stars_with_cap(
        np.clip(residual_pred, 0.0, 1.0),
        threshold=threshold,
    )
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

    flux_errors = []
    for reference_index, predicted_index in matches:
        true_flux = reference[reference_index]["flux"]
        pred_flux = predicted[predicted_index]["flux"]
        flux_errors.append(abs(pred_flux - true_flux) / max(true_flux, 1e-8))

    if flux_errors:
        flux_error = float(np.mean(flux_errors))
    else:
        flux_error = 0.0 if not reference else 1.0

    return {
        "star_precision": float(precision),
        "star_recall": float(recall),
        "star_f1": float(f1),
        "flux_error": flux_error,
        "reference_star_count": float(len(reference)),
        "predicted_star_count": float(len(predicted)),
        "reference_star_count_capped": float(reference_capped),
        "predicted_star_count_capped": float(predicted_capped),
        "star_counts": int(len(reference_stars))
    }


def evaluate_synthetic(
    sample: dict,
    prediction: dict,
    star_threshold= 0.1,
    matching_radius= 3,
):
    """计算一个合成样本可直接进入论文表格的统一指标。"""
    background_true = sample["background_true"]
    clean_true = sample["clean_true"]
    background_pred = prediction["background_pred"]
    residual_pred = prediction["residual_pred"]

    result = {
        "bg_mae": scores.mae(background_pred, background_true),
        "bg_rmse": scores.rmse(background_pred, background_true),
        "bg_ssim": scores.ssim(background_pred, background_true),
        "residual_mae": scores.mae(residual_pred, clean_true),
        "residual_rmse": scores.rmse(residual_pred, clean_true),
        "residual_ssim": scores.ssim(residual_pred, clean_true),
    }
    reference_info = sample.get("metadata", {}).get("star_reference", {})

    result.update(
        star_metrics(
            residual_pred,
            threshold=star_threshold,
            matching_radius=matching_radius,
            reference_stars=reference_info.get("stars"),
        )
    )
    return result


def evaluate_real(observed, background_pred, residual_pred):
    """返回真实图无需背景真值即可计算的描述性统计。"""
    observed = scores.get_image(observed)
    background_pred = scores.get_image(background_pred, clip=False)
    residual_pred = scores.get_image(residual_pred, clip=False)
    total_variation = np.mean(np.abs(np.diff(background_pred, axis=0)))
    total_variation += np.mean(np.abs(np.diff(background_pred, axis=1)))
    stars, count_capped = _extract_stars_with_cap(np.clip(residual_pred, 0.0, 1.0))
    return {
        "observed_mean": float(observed.mean()),
        "background_mean": float(background_pred.mean()),
        "background_total_variation": float(total_variation),
        "residual_mean": float(residual_pred.mean()),
        "residual_std": float(residual_pred.std()),
        "residual_negative_fraction": float(np.mean(residual_pred < 0.0)),
        "residual_saturated_fraction": float(np.mean(residual_pred > 1.0)),
        "detected_star_count": float(len(stars)),
        "detected_star_count_capped": float(count_capped),
    }


def _extract_stars_with_cap(image, threshold=0.03, fwhm=3.0):
    """从数组中解包星星"""
    stars = extract_stars(image, threshold, fwhm, STAR_COUNT_LIMIT+1)
    capped = len(stars) > STAR_COUNT_LIMIT
    return stars[:STAR_COUNT_LIMIT], capped


def center_error(predicted_x, predicted_y, true_x, true_y):
    return float(math.hypot(float(predicted_x) - float(true_x), float(predicted_y) - float(true_y)))


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

        if true_index in matched_true:
            continue
        if pred_index in matched_pred:
            continue
        matched_true.add(true_index)
        matched_pred.add(pred_index)

        matches.append((true_index, pred_index))

    return matches
