"""FFT-Gaussian baseline used by E0, E2, and E4."""

from collections.abc import Iterable

import numpy as np
from scipy import fft


SIGMA_CANDIDATES = (0.02, 0.04, 0.08, 0.16)

# TODO:需要让学长检查一下

def estimate_background(observed, normalized_sigma: float):
    """返回与observed一样大小的(H,W)矩阵"""
    image = np.asarray(observed, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("observed must be a 2-D grayscale image")
    if min(image.shape) < 2:
        raise ValueError("observed must be at least 2x2")
    if not np.isfinite(image).all():
        raise ValueError("observed must contain only finite values")

    sigma = float(normalized_sigma)
    if not 0.0 < sigma <= 0.5:
        raise ValueError("normalized_sigma must be in (0, 0.5]")

    spatial_sigma = 1.0 / (2.0 * np.pi * sigma)
    pad = max(1, int(np.ceil(3.0 * spatial_sigma)))
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode="reflect")

    frequency_y = fft.fftfreq(padded.shape[0])[:, None]
    frequency_x = fft.rfftfreq(padded.shape[1])[None, :]
    frequency_squared = frequency_y**2 + frequency_x**2
    gaussian = np.exp(-0.5 * frequency_squared / sigma**2)

    spectrum = fft.rfft2(padded)
    filtered = fft.irfft2(spectrum * gaussian, s=padded.shape)

    height, width = image.shape
    return filtered[pad : pad + height, pad : pad + width].astype(np.float32)


def select_sigma(
    validation_samples: Iterable,
    candidates=SIGMA_CANDIDATES,
) -> float:
    """MSE Loss选择最优sigma"""
    samples = list(validation_samples)
    if not samples:
        raise ValueError("validation_samples must not be empty")

    candidates = tuple(float(candidate) for candidate in candidates)
    if not candidates:
        raise ValueError("candidates must not be empty")

    mean_losses = []
    for candidate in candidates:
        losses = []
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

            prediction = estimate_background(observed, candidate)
            losses.append(float(np.mean((prediction - background_true) ** 2)))

        mean_losses.append(float(np.mean(losses)))

    return candidates[int(np.argmin(mean_losses))]
