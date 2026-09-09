"""多源椭圆光污染"""
import numpy as np
from matplotlib import pyplot as plt

from pinn_starlight_core.data.image_loader import ImageLoader
import experiments.common.utils.experiment_utils as utils

POINT_LIMIT = 30

def generate_elliptical_background(H, W, seed, show_sources=False):
    mixed_source = np.zeros((H, W), dtype=np.float32)
    for point in range(POINT_LIMIT):
        one_source = generate_single_ellipse(
            H, W, seed + point, show_sources
        )
        mixed_source += one_source

    return np.clip(mixed_source, 0.0, 1.0)


def generate_single_ellipse(H, W, seed, show_source=False):
    y, x = np.mgrid[0:H, 0:W]
    rng = np.random.default_rng(seed)
    bias = rng.uniform(-0.1, 0.3)

    a = W * (0.2 + bias)
    b = H * (0.3 + bias)
    source_x = W * (0.5 + rng.uniform(-0.5, 0.5))
    source_y = H * (1.3 + rng.uniform(-0.2, 0.2))
    q = ((x - source_x) / a) ** 2 + ((y - source_y)/b) ** 2

    strength = 0.005
    one_source = strength * np.clip((1.0-q)**2, 0.0, 1.0)

    if show_source:
        plt.scatter(source_x, source_y)

    return np.float32(one_source)


def main():
    path = utils.PROJECT_ROOT / "data/test/native_test.tif"
    _, values, W, H = ImageLoader(path).get_gray_data()
    image = np.asarray(values, dtype=np.float32).reshape(H, W)

    background = generate_elliptical_background(
        H, W, seed=114514, show_sources=True
    )
    observed = np.clip(image + background, 0.0, 1.0)

    plt.imshow(observed, cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.xlim(0, W)
    plt.ylim(H * 2, 0)
    plt.show()

if __name__ == "__main__":
    main()
