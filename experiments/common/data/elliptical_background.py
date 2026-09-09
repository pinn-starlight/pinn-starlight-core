"""Legacy compatibility wrapper for the old elliptical background helper.

Formal synthetic experiments use :mod:`synthetic_backgrounds` directly.
"""

import numpy as np
from matplotlib import pyplot as plt

from experiments.common.data.synthetic_backgrounds import generate_background
from pinn_starlight_core.data.image_loader import ImageLoader
import experiments.common.utils.experiment_utils as utils


def generate_elliptical_background(H, W, seed, show_sources=False):
    """Return a two-source smooth background using the formal generator."""
    background, truth = generate_background(
        H,
        W,
        background_type="multi_source",
        intensity=0.20,
        scale=0.70,
        seed=seed,
    )
    if show_sources:
        for center in truth["source_centers"]:
            plt.scatter(center["x_pixel"], center["y_pixel"])
    return np.asarray(background, dtype=np.float32)


def generate_single_ellipse(H, W, seed, show_source=False):
    """Compatibility helper for callers that requested a single ellipse."""
    background, truth = generate_background(
        H,
        W,
        background_type="single_eccentric",
        intensity=0.20,
        scale=0.70,
        seed=seed,
    )
    if show_source:
        center = truth["source_centers"][0]
        plt.scatter(center["x_pixel"], center["y_pixel"])
    return np.asarray(background, dtype=np.float32)


def main():
    path = utils.NATIVE_TEST_IMAGE
    _, values, W, H = ImageLoader(path).get_gray_data()
    image = np.asarray(values, dtype=np.float32).reshape(H, W)
    background = generate_elliptical_background(H, W, seed=114514, show_sources=True)
    observed = image + background
    plt.imshow(np.clip(observed, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
