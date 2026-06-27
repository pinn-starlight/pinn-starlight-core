# TODO: 自己重写 RAWLoader
#
# 需要支持:
#   1. RAW 文件 (CR2/NEF/DNG/ARW): rawpy 读取 → demosaic → RGB
#   2. RGB 图片 (PNG/JPG/TIFF): plt.imread → 保留 RGB 三通道
#   3. 合成数据: from_array 接受 numpy/torch
#   4. 输出: get_raw_data → 分通道 coords + values
#
# 已知问题:
#   - 旧版 img.mean(axis=2) 丢掉颜色 → 需保留 C 维
#   - 旧版 W,H 命名反了 (plt.imread 返回 H,W,C)
#   - 旧版 np.clip(img, 0, 1) 硬截断高光
#
# 库参考:
#   rawpy.imread(path) → .raw_image (Bayer), .postprocess() (RGB)
#   plt.imread(path) → (H, W, C) uint8 [0,255] 或 float [0,1]
import numpy as np
import rawpy
import torch
from torch.nn.functional import selu_


class RAWLoader:
    def __init__(self):
        self.rgb_data = None
        self.coords = None
        self.brightness = None
        self.W = None
        self.H = None
        self.path = None

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
                self.rgb_data = raw.postprocess()

        else:
            raise ValueError(f'Unsupported format: .{ext}')

        self.rgb_data = self.rgb_data.astype(np.float32)
        if self.rgb_data.max() > 1.0:
            self.rgb_data /= 255.0

        self.H, self.W = self.rgb_data.shape[:2]


    def from_array(self, array):
        if isinstance(array, torch.Tensor):
            array = array.cpu().numpy()
        self.rgb_data = array.astype(np.float32)
        if self.rgb_data.max() > 1.0:
            self.rgb_data /= 255.0
        self.H, self.W = self.rgb_data.shape[:2]


    def get_raw_data(self, device):
        # TODO: rasterize 分通道, 或 coords 加一维通道标记
        # TODO: 返回 coords, values, W, H
        raise NotImplementedError
