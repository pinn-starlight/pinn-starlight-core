import numpy as np
import rawpy
import torch


class RAWLoader:
    def __init__(self):
        self.rgb_data = None
        self.coords = None
        self.brightness = None
        self.W = None
        self.H = None
        self.path = "none"

    def load(self, path):
        self.path = path
        ext = path.lower().rsplit('.', 1)[-1]

        if ext in ('png', 'jpg', 'jpeg', 'tiff', 'tif'):
            from matplotlib import pyplot as plt
            data = plt.imread(path)
            if data.ndim == 3 and data.shape[-1] == 4:
                data = data[:, :, :3]
            self.rgb_data = data
        elif ext in ('cr2', 'nef', 'dng', 'arw'):
            with rawpy.imread(path) as raw:
                self.rgb_data = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=16,
                    gamma=(1, 1),
                )
        else:
            raise ValueError(f'Unsupported format: .{ext}')

        self.rgb_data = self.rgb_data.astype(np.float32)
        if self.rgb_data.max() > 1.0:
            self.rgb_data /= self.rgb_data.max()

        self.H, self.W = self.rgb_data.shape[:2]


    def from_array(self, array):
        if isinstance(array, torch.Tensor):
            array = array.cpu().numpy()
        self.rgb_data = array.astype(np.float32)
        if self.rgb_data.max() > 1.0:
            self.rgb_data /= self.rgb_data.max()

        self.H, self.W = self.rgb_data.shape[:2]


    def get_gray_data(self, device="cpu"):
        gray = np.mean(self.rgb_data, axis=2)

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

