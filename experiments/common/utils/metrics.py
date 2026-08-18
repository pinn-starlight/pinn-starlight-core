"""E1-E4 共用的论文评价指标。"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def mae(prediction, target) -> float:
    prediction, target = _pair(prediction, target)
    return float(np.mean(np.abs(prediction - target)))


def rmse(prediction, target) -> float:
    prediction, target = _pair(prediction, target)
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def psnr(prediction, target, data_range: float = 1.0) -> float:
    prediction, target = _pair(prediction, target)
    mse = float(np.mean((prediction - target) ** 2))
    return float(10.0 * np.log10(data_range**2 / max(mse, 1e-12)))


def ssim(prediction, target, data_range: float = 1.0) -> float:
    """使用 11x11 高斯窗口计算灰度 SSIM，data_range 固定为 1。"""
    prediction, target = _pair(prediction, target)
    height, width = prediction.shape
    window_size = min(11, height, width)
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        return _global_ssim(prediction, target, data_range)

    sigma = 1.5 * window_size / 11.0
    axis = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    kernel_1d = torch.exp(-(axis**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d).reshape(1, 1, window_size, window_size)

    x = torch.from_numpy(prediction).reshape(1, 1, height, width)
    y = torch.from_numpy(target).reshape(1, 1, height, width)
    padding = window_size // 2
    pad_mode = "reflect" if min(height, width) > padding else "replicate"
    x = F.pad(x, (padding, padding, padding, padding), mode=pad_mode)
    y = F.pad(y, (padding, padding, padding, padding), mode=pad_mode)

    mu_x = F.conv2d(x, kernel)
    mu_y = F.conv2d(y, kernel)
    mu_x_sq = mu_x**2
    mu_y_sq = mu_y**2
    mu_xy = mu_x * mu_y
    sigma_x = F.conv2d(x * x, kernel) - mu_x_sq
    sigma_y = F.conv2d(y * y, kernel) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    )
    return float(score.mean().item())


def extract_stars(
    image,
    threshold: float = 0.03,
    matching_radius: int = 3,
    flux_radius: int = 2,
    max_stars: int = 2000,
) -> list[dict]:
    """从干净参考图或残差图中提取局部亮点及其近邻光通量。"""
    image = _image(image)
    height, width = image.shape
    tensor = torch.from_numpy(image).reshape(1, 1, height, width)

    background_window = min(15, height, width)
    if background_window % 2 == 0:
        background_window -= 1
    if background_window >= 3:
        padding = background_window // 2
        padded = F.pad(
            tensor,
            (padding, padding, padding, padding),
            mode="reflect" if min(height, width) > padding else "replicate",
        )
        local_background = F.avg_pool2d(padded, background_window, stride=1)
        signal = tensor - local_background
    else:
        signal = tensor - tensor.mean()

    radius = max(1, int(matching_radius))
    local_max = F.max_pool2d(signal, 2 * radius + 1, stride=1, padding=radius)
    mask = (signal >= float(threshold)) & (signal >= local_max - 1e-8)
    coordinates = torch.nonzero(mask[0, 0], as_tuple=False)
    if coordinates.numel() == 0:
        return []

    strengths = signal[0, 0, coordinates[:, 0], coordinates[:, 1]]
    order = torch.argsort(strengths, descending=True)[:max_stars]
    signal_array = signal[0, 0].numpy()
    stars = []
    for index in order.tolist():
        y, x = coordinates[index].tolist()
        top = max(0, y - flux_radius)
        bottom = min(height, y + flux_radius + 1)
        left = max(0, x - flux_radius)
        right = min(width, x + flux_radius + 1)
        flux = float(np.maximum(signal_array[top:bottom, left:right], 0.0).sum())
        stars.append({"x": int(x), "y": int(y), "flux": flux})
    return stars


def star_metrics(
    clean_true,
    residual_pred,
    threshold: float = 0.03,
    matching_radius: int = 3,
    flux_radius: int = 2,
    reference_stars=None,
) -> dict[str, float]:
    """以 clean_true 提取的星点作为固定参考，计算匹配和光通量误差。"""
    reference = reference_stars
    if reference is None:
        reference = extract_stars(
            clean_true,
            threshold=threshold,
            matching_radius=matching_radius,
            flux_radius=flux_radius,
        )
    predicted = extract_stars(
        np.clip(residual_pred, 0.0, 1.0),
        threshold=threshold,
        matching_radius=matching_radius,
        flux_radius=flux_radius,
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
    }


def evaluate_synthetic(
    sample: dict,
    prediction: dict,
    star_threshold: float = 0.03,
    matching_radius: int = 3,
) -> dict[str, float]:
    """计算一个合成样本可直接进入论文表格的统一指标。"""
    background_true = sample["background_true"]
    clean_true = sample["clean_true"]
    background_pred = prediction["background_pred"]
    residual_pred = prediction["residual_pred"]

    result = {
        "bg_mae": mae(background_pred, background_true),
        "bg_rmse": rmse(background_pred, background_true),
        "bg_psnr": psnr(background_pred, background_true),
        "bg_ssim": ssim(background_pred, background_true),
        "residual_mae": mae(residual_pred, clean_true),
        "residual_psnr": psnr(residual_pred, clean_true),
        "residual_ssim": ssim(residual_pred, clean_true),
    }
    reference_info = sample.get("metadata", {}).get("star_reference", {})
    if reference_info:
        fixed_threshold = float(reference_info.get("threshold", star_threshold))
        fixed_radius = int(reference_info.get("matching_radius", matching_radius))
        if not np.isclose(fixed_threshold, star_threshold) or fixed_radius != matching_radius:
            raise ValueError(
                "星点阈值或匹配半径与合成数据 metadata 不一致；"
                "正式测试阶段不能临时修改"
            )
    result.update(
        star_metrics(
            clean_true,
            residual_pred,
            threshold=star_threshold,
            matching_radius=matching_radius,
            reference_stars=reference_info.get("stars"),
        )
    )
    return result


def evaluate_real(observed, background_pred, residual_pred) -> dict[str, float]:
    """返回真实图无需背景真值即可计算的描述性统计。"""
    observed = _image(observed)
    background_pred = _image(background_pred, clip=False)
    residual_pred = _image(residual_pred, clip=False)
    total_variation = np.mean(np.abs(np.diff(background_pred, axis=0)))
    total_variation += np.mean(np.abs(np.diff(background_pred, axis=1)))
    stars = extract_stars(np.clip(residual_pred, 0.0, 1.0))
    return {
        "observed_mean": float(observed.mean()),
        "background_mean": float(background_pred.mean()),
        "background_total_variation": float(total_variation),
        "residual_mean": float(residual_pred.mean()),
        "residual_std": float(residual_pred.std()),
        "residual_negative_fraction": float(np.mean(residual_pred < 0.0)),
        "residual_saturated_fraction": float(np.mean(residual_pred > 1.0)),
        "detected_star_count": float(len(stars)),
    }


def center_error(predicted_x, predicted_y, true_x, true_y) -> float:
    return float(math.hypot(float(predicted_x) - float(true_x), float(predicted_y) - float(true_y)))


def _match_stars(reference: list[dict], predicted: list[dict], radius: float):
    candidates = []
    for reference_index, true_star in enumerate(reference):
        for predicted_index, pred_star in enumerate(predicted):
            distance = math.hypot(
                true_star["x"] - pred_star["x"],
                true_star["y"] - pred_star["y"],
            )
            if distance <= radius:
                candidates.append((distance, reference_index, predicted_index))

    matches = []
    used_reference = set()
    used_predicted = set()
    for _, reference_index, predicted_index in sorted(candidates):
        if reference_index in used_reference or predicted_index in used_predicted:
            continue
        used_reference.add(reference_index)
        used_predicted.add(predicted_index)
        matches.append((reference_index, predicted_index))
    return matches


def _pair(prediction, target):
    prediction = _image(prediction, clip=False)
    target = _image(target, clip=False)
    if prediction.shape != target.shape:
        raise ValueError(f"指标输入尺寸不一致：{prediction.shape} 与 {target.shape}")
    return prediction, target


def _image(image, clip: bool = True):
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"指标输入必须是二维灰度图，实际为 {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("指标输入包含 NaN 或 Inf")
    return np.clip(array, 0.0, 1.0) if clip else array


def _global_ssim(prediction, target, data_range):
    mean_x = float(prediction.mean())
    mean_y = float(target.mean())
    variance_x = float(prediction.var())
    variance_y = float(target.var())
    covariance = float(np.mean((prediction - mean_x) * (target - mean_y)))
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    return float(
        ((2 * mean_x * mean_y + c1) * (2 * covariance + c2))
        / ((mean_x**2 + mean_y**2 + c1) * (variance_x + variance_y + c2))
    )
