from pathlib import Path

import numpy as np
import rawpy
import tifffile as tif
import torch
from matplotlib import pyplot as plt

_RASTER_EXTS = {'png', 'jpg', 'jpeg'}
_RAW_EXTS = {'cr2', 'nef', 'dng', 'arw'}
_TIFF_EXTS = {'tif', 'tiff'}


def coordinate_grid(height: int, width: int, device="cpu") -> torch.Tensor:
    """获取(x,y)坐标"""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    x = torch.linspace(-1.0, 1.0, width, device=device)
    y = torch.linspace(-1.0, 1.0, height, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)


# 暂时使用灰度图训练
class ImageLoader:
    def __init__(self, path: Path, device="cpu", downsample=2):
        """
            此path为文件的路径而非文件夹
        """
        if downsample <= 0:
            raise ValueError("downsample must be positive")
        if not path.exists():
            raise FileNotFoundError(f"图像不存在：{path}")
        self.device = device
        ext = path.suffix.lower().lstrip('.')
        scale_val, data = self._normalize_photo(ext, path)

        if scale_val is None:
            raise ValueError('scale is None')

        if data.ndim == 3 and data.shape[-1] == 4:
            data = data[:, :, :3]
        if data.ndim == 2:
            data = np.stack([data] * 3, axis=-1)
        if data.ndim != 3 or data.shape[-1] != 3:
            raise ValueError(f'Unsupported image shape: {data.shape}')

        rgb_data = data.astype(np.float32) / np.float32(scale_val)

        self.rgb_data = rgb_data[::downsample, ::downsample, :]
        self.H, self.W = self.rgb_data.shape[:2]

    @staticmethod
    def _normalize_photo(ext : str, path):
        if ext in _RAW_EXTS:
            with rawpy.imread(str(path)) as raw:
                data = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=16,
                    gamma=(1, 1)
                )
            scale_val = 65535.0
        elif ext in _RASTER_EXTS:
            data = plt.imread(path)
            scale_val = np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else 1.0
        elif ext in _TIFF_EXTS:
            data = tif.imread(path)
            scale_val = np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else 1.0
        else:
            raise ValueError(f'Unsupported format: .{ext}')

        return scale_val, data


    def get_gray_data(self):
        rgb = self.rgb_data
        gray = (
            0.2126 * rgb[:, :, 0] +
            0.7152 * rgb[:, :, 1] +
            0.0722 * rgb[:, :, 2]
        )

        coords = coordinate_grid(self.H, self.W, self.device)
        brightness = torch.tensor(
            gray.ravel(), dtype=torch.float32, device=self.device
        )

        return coords, np.clip(brightness, 0, 1), self.W, self.H

