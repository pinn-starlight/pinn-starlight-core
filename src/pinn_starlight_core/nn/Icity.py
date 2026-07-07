import numpy as np
import torch
import torchvision.transforms.functional as F
import torch.nn.functional as F_conv

import pinn_starlight_core.data.PhotoLoader as Loader


class Icity:
    def __init__(self, path, device, kernel_size = 21):
        loader = Loader.RAWLoader()
        loader.load(path)

        self.gray_image = np.mean(loader.rgb_data, axis=2).astype(np.float32)
        H, W = self.gray_image.shape
        img_tensor = torch.from_numpy(self.gray_image).unsqueeze(0).unsqueeze(0)

        self.kernel_size = kernel_size
        sigma = self.kernel_size / 3.0
        blurred_img = F.gaussian_blur(img_tensor,
                                      kernel_size=[self.kernel_size, self.kernel_size],
                                      sigma=[sigma, sigma],
                                      ).squeeze()

        laplacian_kernel = torch.tensor([[0., 1., 0.],
                                         [1., -4., 1.],
                                         [0., 1., 0.]])
        laplacian_img = F_conv.conv2d(blurred_img.unsqueeze(0).unsqueeze(0),
                                      weight=laplacian_kernel.unsqueeze(0).unsqueeze(0),
                                      padding=1).squeeze()

        bright_mask = (blurred_img > blurred_img.quantile(0.70)).float()
        y_axis = torch.linspace(0, 1, H)
        vertical_decay = (1 - torch.exp(-y_axis * 6))[:, None]

        gy, gx = torch.gradient(blurred_img)
        gradient_magnitude = torch.sqrt(gx**2 + gy**2)
        edge_mask = (gradient_magnitude < gradient_magnitude.quantile(0.95)).float()

        self.icity = (laplacian_img * edge_mask * vertical_decay * bright_mask).to(device)

    def get_icity(self):
        return self.icity






