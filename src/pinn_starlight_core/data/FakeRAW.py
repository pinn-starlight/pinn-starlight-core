import torch
import numpy as np


class FakeRaw:
    def __init__(self, H=128, W=128, n_stars=8, bg_amplitude=0.3, star_brightness=0.8, seed=0):
        self.H = H
        self.W = W
        self.n_stars = n_stars
        self.bg_amplitude = bg_amplitude
        self.star_brightness = star_brightness
        self.seed = seed
        self.coords = None
        self.values = None

    def get_raw_data(self):
        x = torch.linspace(0, 1, self.W)
        y = torch.linspace(0, 1, self.H)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        self.coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        N = self.H * self.W

        # cos 背景（所有图共享同一背景——代表同一天区的光污染）
        background = (self.bg_amplitude
                      * torch.cos(3.0 * self.coords[:, 0])
                      * torch.cos(3.0 * self.coords[:, 1]))

        # 随机星点（每张图不同 seed，星点位置不同）
        rng = np.random.default_rng(self.seed)
        stars = torch.zeros(N)
        centers = rng.random((self.n_stars, 2))
        for cx, cy in centers:
            cx, cy = float(cx), float(cy)
            dist2 = (self.coords[:, 0] - cx) ** 2 + (self.coords[:, 1] - cy) ** 2
            stars += self.star_brightness * torch.exp(-dist2 / 0.0005)

        self.values = background + stars
        return self.coords, self.values
