"""Legacy compatibility wrapper for the old exponential background helper.

Formal synthetic experiments use :mod:`synthetic_backgrounds` directly.
"""

import numpy as np
from matplotlib import pyplot as plt

from experiments.common.data.synthetic_backgrounds import generate_background
from pinn_starlight_core.data.image_loader import ImageLoader
import experiments.common.utils.experiment_utils as utils


def generate_exponential_background(H, W, seed=0, show_source=False):
    """Return a single radial smooth background using the formal generator."""
    background, truth = generate_background(
        H,
        W,
        background_type="single_radial",
        intensity=0.30,
        scale=0.70,
        seed=seed,
    )
    if show_source:
        center = truth["source_centers"][0]
        plt.scatter(center["x_pixel"], center["y_pixel"], label="source centre")
    return np.asarray(background, dtype=np.float32)


def show_example():
    path = utils.NATIVE_TEST_IMAGE
    _, values, W, H = ImageLoader(path).get_gray_data()
    image = np.asarray(values, dtype=np.float32).reshape(H, W)
    background = generate_exponential_background(H, W, seed=114514, show_source=True)
    observed = image + background
    plt.imshow(np.clip(observed, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.legend()
    plt.show()
