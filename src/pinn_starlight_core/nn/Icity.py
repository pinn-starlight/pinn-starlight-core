import numpy as np
import torch
from torch import nn
import torchvision.transforms.functional as F

import pinn_starlight_core.data.PhotoLoader as Loader


# Fixed A,B (A,B = 1)
class Icity(nn.Module):
    def __init__(self, path, device, kernel_size=31, alpha=0.5):
        super().__init__()
        self.device = device
        self.alpha = torch.as_tensor(alpha)

        loader = Loader.RAWLoader()
        loader.load(path)
        gray_img = np.mean(loader.rgb_data, axis=2).astype(np.float32)
        H, W = gray_img.shape

        img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(0)
        sigma = kernel_size / 3.0
        blurred = F.gaussian_blur(img_tensor,
                                   kernel_size=[kernel_size, kernel_size],
                                   sigma=[sigma, sigma]).squeeze()

        bright_mask = blurred > blurred.quantile(0.95)
        ys, xs = torch.where(bright_mask)
        x_init = xs.float().mean().item() / W
        y_init = ys.float().mean().item() / H

        self.x = nn.Parameter(torch.tensor([x_init], device=device))
        self.y = nn.Parameter(torch.tensor([y_init], device=device))

    def forward(self, coords):
        x = coords[:, 0]
        y = coords[:, 1]
        r = torch.sqrt((x - self.x) ** 2 + (y - self.y) ** 2 + 1e-8)
        f = 2 * self.alpha * torch.cos(r * torch.sqrt(self.alpha)) + (torch.sqrt(self.alpha) / r) * torch.sin(r * torch.sqrt(self.alpha)) - 16 * (r**2) * (self.alpha**2) + (r**4) * (self.alpha ** 3)
        return f