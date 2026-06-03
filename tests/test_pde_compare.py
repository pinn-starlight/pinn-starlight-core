"""不修改源文件，对比 Helmholtz vs Screened Poisson 在真实图片上的效果"""
import torch
import os
import numpy as np
import pytest
from pinn_starlight_core.data.RAWLoader import ImageLoader
from pinn_starlight_core.nn.Layers import SkyglowMLP
from pinn_starlight_core.utils.Laplacian import laplacian


INPUT_DIR = "data/real_raw/origin"
OUTPUT_DIR = "data/real_raw/pde_compare"


@pytest.fixture(autouse=True)
def setup():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _physics_loss(I_pred, coords, alpha, I_city, equation: str):
    """手工算物理损失，不依赖 Losses.py"""
    lap = laplacian(I_pred, coords)
    if equation == "helmholtz":
        f = lap + alpha * I_pred - I_city
    elif equation == "screened_poisson":
        f = lap - alpha * I_pred + I_city
    return (f ** 2).mean()


def _train(fname: str, equation: str, steps: int):
    """用指定 PDE 训练一张真实图片"""
    torch.manual_seed(42)
    loader = ImageLoader()
    loader.load(os.path.join(INPUT_DIR, fname))
    H, W = loader.data.shape
    coords, I_obs = loader.get_raw_data()
    N = H * W

    alpha_param = torch.nn.Parameter(torch.tensor([5.0]))
    mlp = SkyglowMLP([2, 64, 32, 1])
    opt = torch.optim.Adam(list(mlp.parameters()) + [alpha_param], lr=0.001)
    I_city_const = 0.5

    for _ in range(steps):
        idx = torch.randint(0, N, (512,))
        xy = coords[idx].clone().requires_grad_(True)
        Ip = mlp.forward(xy).squeeze()

        data_loss = ((Ip - I_obs[idx]) ** 2).mean()
        phys_loss = _physics_loss(Ip, xy, alpha_param, I_city_const, equation)
        (data_loss + 0.1 * phys_loss).backward()
        opt.step()
        opt.zero_grad()

    with torch.no_grad():
        I_pred = torch.empty(N)
        for s in range(0, N, 50000):
            e = min(s + 50000, N)
            I_pred[s:e] = mlp.forward(coords[s:e]).squeeze()

    mse = ((I_obs - I_pred) ** 2).mean().item()
    res_max = (I_obs - I_pred).max().item()
    return I_pred, I_obs, mse, res_max, alpha_param.item(), H, W


class TestPDECompareOnReal:
    """真实图片上对比两种 PDE"""

    @pytest.mark.parametrize("fname", sorted(os.listdir(INPUT_DIR)))
    def test_helmholtz(self, fname):
        pred, obs, mse, res_max, alpha, H, W = _train(fname, "helmholtz", steps=2000)
        base = fname.rsplit(".", 1)[0]
        _save(f"{base}_helmholtz", pred, obs, H, W)
        assert mse < 0.05, f"MSE={mse:.4f}"
        assert alpha > 0

    @pytest.mark.parametrize("fname", sorted(os.listdir(INPUT_DIR)))
    def test_screened_poisson(self, fname):
        pred, obs, mse, res_max, alpha, H, W = _train(fname, "screened_poisson", steps=2000)
        base = fname.rsplit(".", 1)[0]
        _save(f"{base}_screened_poisson", pred, obs, H, W)
        assert mse < 0.05, f"MSE={mse:.4f}"
        assert alpha > 0

    def test_same_image_both_equations(self):
        """同一张图，两种方程对比（1.jpg，3000 步）"""
        results = {}
        for eq in ("helmholtz", "screened_poisson"):
            pred, obs, mse, res_max, alpha, H, W = _train("1.jpg", eq, steps=3000)
            results[eq] = dict(mse=mse, res_max=res_max, alpha=alpha,
                               pred=pred, obs=obs)
            _save(f"1_compare_{eq}", pred, obs, H, W)

        # 两种方程都应该收敛
        for eq in results:
            assert results[eq]["mse"] < 0.05, f"{eq}: MSE too high"
            assert results[eq]["alpha"] > 0, f"{eq}: alpha should be positive"


def _save(tag, pred, obs, H, W):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, arr in [("observed", obs), ("predicted", pred),
                       ("residual", (obs - pred).clamp(0, 1))]:
        plt.imsave(f"{OUTPUT_DIR}/{tag}_{name}.png",
                   arr.reshape(H, W).numpy(), cmap="gray")
