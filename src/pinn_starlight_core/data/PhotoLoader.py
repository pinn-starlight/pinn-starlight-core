import numpy as np
import rawpy
import torch
from pathlib import Path
from matplotlib import pyplot as plt


# 暂时使用灰度图训练
class RAWLoader:
    def __init__(self, path: str):
        ext = Path(path).suffix.lower().lstrip('.')

        if ext in ('png', 'jpg', 'jpeg', 'tiff', 'tif'):
            data = plt.imread(path)
            if data.ndim == 3 and data.shape[-1] == 4:
                data = data[:, :, :3]
            self.rgb_data = data
            scale = 255.0
        elif ext in ('cr2', 'nef', 'dng', 'arw'):
            with rawpy.imread(path) as raw:
                self.rgb_data = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=16,
                    gamma=(1, 1)
                )
            scale = 65535.0
        else:
            raise ValueError(f'Unsupported format: .{ext}')

        rgb_data = self.rgb_data.astype(np.float32) / np.float32(scale)

        self.rgb_data = rgb_data[::2, ::2, :]
        self.H, self.W = self.rgb_data.shape[:2]

    def get_gray_data(self, device = "cpu"):
        rgb = self.rgb_data
        gray = (
            0.2126 * rgb[:, :, 0] +
            0.7152 * rgb[:, :, 1] +
            0.0722 * rgb[:, :, 2]
        )

        x = np.linspace(-1, 1, self.W)
        y = np.linspace(-1, 1, self.H)
        xx, yy = np.meshgrid(x, y)

        coords = torch.tensor(
            np.stack([xx.ravel(), yy.ravel()], axis=-1),
            dtype=torch.float32, device=device
        )
        brightness = torch.tensor(
            gray.ravel(), dtype=torch.float32, device=device
        )

        return coords, brightness, self.W, self.H

