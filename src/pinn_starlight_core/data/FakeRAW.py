import numpy
import torch


class FakeRaw:
    def __init__(
        self,
        H = 512,
        W = 512,
        n_stars = 10,
        bg_amplitude = 0.3,
        star_brightness = 0.7,
        seed = 114514
    ):
        self.coords = None
        self.values = None
        self.stars = None
        self.background = None
        self.H = H
        self.W = W
        self.n_stars = n_stars
        self.bg_amplitude = bg_amplitude
        self.star_brightness = star_brightness
        self.seed = seed

    def get_raw_data(self):
        x = torch.linspace(0, 1, self.W)
        y = torch.linspace(0, 1, self.H)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        x_flat = xx.flatten()
        y_flat = yy.flatten()
        N = self.H * self.W

        rng = numpy.random.default_rng(self.seed)
        stars = torch.zeros(N)
        centers = rng.random((self.n_stars, 2))
        for cx, cy in centers:
            cx, cy = float(cx), float(cy)
            dist2 = (cx - x_flat) ** 2 + (cy - y_flat) ** 2
            stars += self.star_brightness * torch.exp(-dist2 / 0.0007)

        self.background = self.bg_amplitude * torch.cos(3.0 * x_flat) * torch.cos(3.0 * y_flat)
        self.stars = stars
        self.values = self.background + stars
        self.coords = torch.stack([x_flat, y_flat], dim=1)

        return self.coords, self.values

    def to_png(self, path="fake_sky.png"):
        import matplotlib.pyplot as plt
        img = self.values.reshape(self.H, self.W).numpy()
        plt.imsave(path, img, cmap='gray')