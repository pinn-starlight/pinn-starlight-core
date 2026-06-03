import numpy
import torch


class FakeRaw:
    def __init__(
        self,
        W = 1920,
        H = 1080,
        n_stars = 10,
        bg_amplitude = 0.3,
        star_brightness = 0.7,
        seed = 114514
    ):
        self.fake_raw = torch.Tensor(H, W)
        self.stars = torch.Tensor(H, W)
        self.background = torch.Tensor(H, W)
        self.H = H
        self.W = W
        self.n_stars = n_stars
        self.bg_amplitude = bg_amplitude
        self.star_brightness = star_brightness
        self.seed = seed

    def get_fake_raw(self):
        x = torch.linspace(0, 1, self.W)
        y = torch.linspace(0, 1, self.H)
        xx, yy = torch.meshgrid(x, y, indexing='xy')

        rng = numpy.random.default_rng(self.seed)
        stars = torch.zeros(self.H, self.W)
        centers = rng.random((self.n_stars, 2))
        for cx, cy in centers:
            cx, cy = float(cx), float(cy)
            dist2 = (cx - xx) ** 2 + (cy - yy) ** 2
            stars += self.star_brightness * torch.exp(-dist2 / 0.0007)
        self.stars = stars

        self.background = self.bg_amplitude * torch.cos(3.0 * xx) * torch.cos(3.0 * yy)
        self.fake_raw = self.background + stars

        return self.fake_raw

    def to_png(self, path="fake_sky.png"):
        import matplotlib.pyplot as plt
        plt.imsave(path, self.fake_raw.numpy(), cmap='gray')