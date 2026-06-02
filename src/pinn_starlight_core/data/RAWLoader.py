import torch
import numpy as np
import matplotlib.pyplot as plt
from pinn_starlight_core.utils.Rasterize import rasterize


class ImageLoader:
    def __init__(self):
        self.path = None
        self.data = None
        self.coords = None

    def load(self, path: str):
        self.path = path
        ext = path.lower().rsplit('.', 1)[-1]

        if ext in ('cr2', 'nef', 'dng', 'arw'):
            self._load_raw(path)
        elif ext in ('png', 'jpg', 'jpeg', 'tiff', 'tif'):
            self._load_png(path)
        else:
            raise ValueError(f'Unsupported format: {ext}')

    def _load_raw(self, path):
        import rawpy
        with rawpy.imread(path) as raw:
            img = raw.raw_image.astype(np.float32).copy()
        # 归一化（黑电平 + 白电平）
        bl = getattr(raw, 'black_level_per_channel', [0])[0]
        wl = getattr(raw, 'white_level', 65535)
        img = (img - bl) / (wl - bl)
        self.data = torch.from_numpy(np.clip(img, 0, 1))

    def _load_png(self, path):
        img = plt.imread(path)                              # (H, W) 或 (H, W, 3)
        if img.ndim == 3:
            img = img.mean(axis=2)                          # RGB → 灰度
        img = img.astype(np.float32)
        if img.max() > 1.0:                                 # 0-255 → 0-1
            img /= 255.0
        self.data = torch.from_numpy(img)

    def get_raw_data(self):
        self.coords, self.data = rasterize(self.data.numpy())
        return self.coords, self.data
