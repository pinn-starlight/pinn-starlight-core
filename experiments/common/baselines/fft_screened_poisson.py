"""screened-Poisson 基线"""

from collections.abc import Iterable

import numpy as np
from scipy import fft


PHYSICS_WEIGHT_CANDIDATES = (0.5, 0.7, 1, 1.2, 1.5, 2)


def estimate_background(observed, weight: float, alpha: float = 0.5, source=None):
    """返回与observed一样大小的(H,W)矩阵"""
    input_image = np.asarray(observed, dtype=np.float32)
    if input_image.ndim != 2:
        raise ValueError("observed must be a 2-D grayscale image")
    if min(input_image.shape) < 2:
        raise ValueError("observed must be at least 2x2")
    if not np.isfinite(input_image).all():
        raise ValueError("observed must contain only finite values")

    physics_weight = float(weight)
    if physics_weight <= 0.0:
        raise ValueError("normalized_sigma must be positive")
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")

    image_height, image_width = input_image.shape
    source_image = np.zeros_like(input_image) if source is None else np.asarray(source, dtype=np.float32)
    if source_image.shape != input_image.shape:
        raise ValueError("source must have the same shape as observed")

    padding_pixels = 1
    padded_image = np.pad(input_image, padding_pixels, mode="reflect")
    padded_source = np.pad(source_image, padding_pixels, mode="reflect")
    image_spectrum = fft.rfft2(padded_image)
    source_spectrum = fft.rfft2(padded_source)

    frequency_y = fft.fftfreq(padded_image.shape[0])[:, None]
    frequency_x = fft.rfftfreq(padded_image.shape[1])[None, :]
    laplacian_symbol = (
        2.0 * np.cos(2.0 * np.pi * frequency_y)
        + 2.0 * np.cos(2.0 * np.pi * frequency_x)
        - 4.0
    )
    screened_operator = laplacian_symbol - float(alpha)
    denominator = 1.0 + physics_weight * screened_operator**2
    background_spectrum = (
        image_spectrum - physics_weight * screened_operator * source_spectrum
    ) / denominator
    padded_background = fft.irfft2(background_spectrum, s=padded_image.shape)
    return padded_background[
        padding_pixels : padding_pixels + image_height,
        padding_pixels : padding_pixels + image_width,
    ].astype(np.float32)


def select_weight(
    validation_samples: Iterable,
    candidates=PHYSICS_WEIGHT_CANDIDATES,
) -> float:
    samples = list(validation_samples)
    if not samples:
        raise ValueError("validation_samples must not be empty")

    candidates = tuple(float(candidate) for candidate in candidates)
    if not candidates:
        raise ValueError("candidates must not be empty")

    average_validation_errors = []
    for candidate_sigma in candidates:
        sample_errors = []
        for sample in samples:
            if isinstance(sample, dict):
                observed = sample["observed"]
                background_true = sample["background_true"]
            else:
                observed, background_true = sample

            observed = np.asarray(observed, dtype=np.float32)
            background_true = np.asarray(background_true, dtype=np.float32)
            if observed.shape != background_true.shape:
                raise ValueError(
                    "observed and background_true must have the same shape"
                )

            prediction = estimate_background(observed, candidate_sigma)
            sample_errors.append(float(np.mean((prediction - background_true) ** 2)))

        average_validation_errors.append(float(np.mean(sample_errors)))

    return candidates[int(np.argmin(average_validation_errors))]
