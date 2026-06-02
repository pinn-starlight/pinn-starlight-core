import torch


def rasterize(data):
    H, W = data.shape

    x = torch.linspace(0, 1, W)
    y = torch.linspace(0, 1, H)
    xx, yy = torch.meshgrid(x, y, indexing='xy')

    coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    values = torch.from_numpy(data).flatten().float()

    return coords, values
