"""FFT-Gaussian method implementation.

Experiment candidates and validation selection live in
``experiments/configs/formal_experiment.py`` and
``experiments/scripts/selection/gaussian/run.py`` respectively.
"""

import numpy as np
from scipy import fft


def estimate_background(observed, normalized_sigma: float):
    """Estimate a smooth background from observed + explicit normalized sigma."""
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
