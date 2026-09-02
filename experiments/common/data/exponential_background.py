"""指数衰减光污染"""

import numpy as np
from matplotlib import pyplot as plt

from pinn_starlight_core.data.image_loader import ImageLoader
import experiments.common.utils.experiment_utils as utils


def generate_exponential_background(H, W, seed=0, show_source=False):
    """Generate a smooth background from a light source below the image."""
    y, x = np.mgrid[0:H, 0:W]

    rng = np.random.default_rng(seed)
    source_x = W * (0.5 + rng.uniform(-0.25, 0.25))
    source_y = H * (1.5 + rng.uniform(-0.25, 0.25))

    base_level = 0.02
    strength = 0.3
    decay_length = H * 0.5

    distance = np.sqrt(
        (x - source_x) ** 2
        + (y - source_y) ** 2
    )
    decay = np.exp(-distance / decay_length)
    background = base_level + strength * decay

    if show_source:
        plt.scatter(source_x, source_y)

    return np.clip(background, 0.0, 1.0).astype(np.float32)


def show_example():
    path = utils.PROJECT_ROOT / "data/test/native_test.tif"
    _, values, W, H = ImageLoader(path).get_gray_data()
    image = np.asarray(values, dtype=np.float32).reshape(H, W)

    background = generate_exponential_background(
        H, W, seed=114514, show_source=True
    )
    observed = np.clip(image + background, 0.0, 1.0)

    plt.imshow(observed, cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.xlim(0, W)
    plt.ylim(H * 1.7, 0)
    plt.legend()
    plt.show()
