import torch
import numpy as np
import matplotlib.pyplot as plt


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
        self.background = None
        self.stars = None

    def get_raw_data(self):
        x = torch.linspace(0, 1, self.W)
        y = torch.linspace(0, 1, self.H)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        self.coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        N = self.H * self.W

        # cos 背景
        self.background = (self.bg_amplitude
                           * torch.cos(3.0 * self.coords[:, 0])
                           * torch.cos(3.0 * self.coords[:, 1]))

        # 随机星点
        rng = np.random.default_rng(self.seed)
        self.stars = torch.zeros(N)
        centers = rng.random((self.n_stars, 2))
        for cx, cy in centers:
            cx, cy = float(cx), float(cy)
            dist2 = (self.coords[:, 0] - cx) ** 2 + (self.coords[:, 1] - cy) ** 2
            self.stars += self.star_brightness * torch.exp(-dist2 / 0.0005)

        self.values = self.background + self.stars
        return self.coords, self.values

    def to_png(self, path="test_data/fake_raw/fake_sky.png"):
        """导出观测图为 PNG"""
        img = self.values.reshape(self.H, self.W).numpy()
        plt.imsave(path, img, cmap='gray')

    def to_png_decomposed(self, path="test_data/fake_raw/fake_sky_decomposed.png"):
        """导出背景 + 星点 + 观测 三栏对比"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ['Background', 'Stars', 'Observed']
        arrays = [self.background, self.stars, self.values]
        for ax, title, arr in zip(axes, titles, arrays):
            ax.imshow(arr.reshape(self.H, self.W).numpy(), cmap='gray')
            ax.set_title(title)
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
