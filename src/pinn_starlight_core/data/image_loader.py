import numpy as np
import rawpy
import torch
import tifffile as tif
from pathlib import Path
from matplotlib import pyplot as plt

_RASTER_EXTS = {'png', 'jpg', 'jpeg'}
_RAW_EXTS = {'cr2', 'nef', 'dng', 'arw'}
_TIFF_EXTS = {'tif', 'tiff'}


# 暂时使用灰度图训练
class ImageLoader:
    def __init__(self, path: str, device="cpu"):
        """
            此path为文件的路径而非文件夹
        """
        self.device = device
        ext = Path(path).suffix.lower().lstrip('.')
        scale = None

        if ext in _RAW_EXTS:
            with rawpy.imread(path) as raw:
                self.rgb_data = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=16,
                    gamma=(1, 1)
                )
            scale = 65535.0
        elif ext in _RASTER_EXTS:
            data = plt.imread(path)
            if data.ndim == 3 and data.shape[-1] == 4:
                data = data[:, :, :3]
            if data.ndim == 2:
                data = np.stack([data] * 3, axis=-1)
            self.rgb_data = data
            scale = np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else 1.0
        elif ext in _TIFF_EXTS:
            self.rgb_data = tif.imread(path).astype(np.float32)
            scale = self.rgb_data.max()
        else:
            raise ValueError(f'Unsupported format: .{ext}')

        if scale is None:
            raise ValueError('scale is None')

        rgb_data = self.rgb_data.astype(np.float32) / np.float32(scale)

        self.rgb_data = rgb_data[::2, ::2, :]
        self.H, self.W = self.rgb_data.shape[:2]

    def get_gray_data(self):
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
            dtype=torch.float32, device=self.device
        )
        brightness = torch.tensor(
            gray.ravel(), dtype=torch.float32, device=self.device
        )

        return coords, brightness, self.W, self.H

