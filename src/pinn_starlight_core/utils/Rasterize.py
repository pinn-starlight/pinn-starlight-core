import torch


def rasterize(data, device = "cpu"):
    H, W = data.shape

    x = torch.linspace(0, 1, W, device= device)
    y = torch.linspace(0, 1, H, device= device)
    xx, yy = torch.meshgrid(x, y, indexing='xy')

    coords = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)
    values = torch.as_tensor(data).flatten().float().to(device)

    return coords, values
