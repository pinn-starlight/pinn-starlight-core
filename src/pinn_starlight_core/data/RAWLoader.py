import numpy as np
import rawpy
import torch


class RAWLoader:
    def __init__(self):
        self.data = None
        self.coords = None
        self.values = None
        self.W = None
        self.H = None
        self.path = None

    def load(self, path):
        self.path = path
        ext = path.lower().rsplit('.', 1)[-1]

        if ext in ('cr2', 'nef', 'dng', 'arw'):
            with rawpy.imread(path) as raw:
                img = raw.raw_image.astype(np.float32).copy()
                bl = raw.black_level_per_channel[0]
                wl = raw.white_level
                img = (img - bl) / (wl - bl)
            self.data = torch.from_numpy(np.clip(img, 0, 1))
        elif ext in ('png', 'jpg', 'jpeg', 'tiff', 'tif'):
            import matplotlib.pyplot as plt
            img = plt.imread(path)
            if img.ndim == 3:
                img = img.mean(axis=2)
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img /= 255.0
            self.data = torch.from_numpy(img)
        else:
            raise ValueError(f'Unsupported format: {ext}')

        self.W, self.H = self.data.shape

    def from_array(self, fake_raw):
        if isinstance(fake_raw, torch.Tensor):
            fake_raw = fake_raw.numpy()
        self.data = torch.from_numpy(fake_raw.astype(np.float32))

        self.W, self.H = self.data.shape

    def get_raw_data(self, device = "cpu"):
        from pinn_starlight_core.utils.Rasterize import rasterize
        coords, values = rasterize(self.data, device)

        return coords, values, self.W, self.H
