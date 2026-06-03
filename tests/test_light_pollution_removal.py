import torch
import os
import pytest

from pinn_starlight_core.data.RAWLoader import ImageLoader
from pinn_starlight_core.nn.Layers import SkyglowMLP
from pinn_starlight_core.nn.Losses import MSEData, MSEPhysics


INPUT_DIR = "data/real_raw/origin"
OUTPUT_DIR = "data/real_raw/origin_output"

# 每张图的测试参数: (文件名, 期望的最小残差max, 训练步数, I_city)
IMAGES = sorted(os.listdir(INPUT_DIR))


@pytest.fixture(autouse=True)
def setup_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _train_one_image(fname: str, steps: int, I_city: float):
    """训练一张图，返回 (alpha, mse, residual_max, pred_tensor, obs_tensor)"""
    loader = ImageLoader()
    loader.load(os.path.join(INPUT_DIR, fname))
    H, W = loader.data.shape
    coords, I_obs = loader.get_raw_data()
    N = H * W

    torch.manual_seed(42)
    mlp = SkyglowMLP([2, 64, 32, 1])
    alpha_param = torch.nn.Parameter(torch.tensor([5.0]))
    opt = torch.optim.Adam(list(mlp.parameters()) + [alpha_param], lr=0.001)
    ld = MSEData()
    lp = MSEPhysics()

    initial_loss = None
    for step in range(steps):
        idx = torch.randint(0, N, (512,))
        xy = coords[idx].clone().requires_grad_(True)
        Ip = mlp.forward(xy).squeeze()
        loss = ld.forward(I_obs[idx], Ip) + lp.forward(
            I_obs[idx], Ip, torch.full((512,), I_city), alpha_param, 0.1, xy
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            initial_loss = loss.item()

    with torch.no_grad():
        I_pred = torch.empty(N)
        for s in range(0, N, 50000):
            e = min(s + 50000, N)
            I_pred[s:e] = mlp.forward(coords[s:e]).squeeze()

    mse = ((I_obs - I_pred) ** 2).mean().item()
    residual = (I_obs - I_pred).clamp(0, 1)

    return alpha_param.item(), mse, residual.max().item(), I_pred, I_obs, initial_loss, H, W


class TestSingleImage:
    """单图训练 + 输出去光污染结果"""

    def test_image_1_large(self):
        """最大的那张图 (2394x1440), 3000 步"""
        alpha, mse, res_max, pred, obs, init_loss, H, W = _train_one_image(
            "1.jpg", steps=3000, I_city=0.5
        )
        base = "1.jpg".rsplit(".", 1)[0]
        _save_outputs(base, obs, pred, H, W)

        assert mse < 0.01, f"MSE={mse:.4f} too high"
        assert res_max > 0.1, f"residual max={res_max:.4f} too low (no stars recovered)"
        assert alpha > 0, "alpha should be positive"


class TestBatchImages:
    """批量测试 — 每张图 500 步快速验证"""

    @pytest.mark.parametrize("fname", IMAGES)
    def test_quick_training(self, fname):
        alpha, mse, res_max, pred, obs, init_loss, H, W = _train_one_image(
            fname, steps=500, I_city=0.5
        )
        base = fname.rsplit(".", 1)[0]
        _save_outputs(base, obs, pred, H, W)

        assert mse < 0.05, f"{fname}: MSE={mse:.4f} too high"
        assert alpha > 0, f"{fname}: alpha={alpha:.2f} should be positive"
        assert res_max > 0.05, f"{fname}: residual max={res_max:.4f} too low"


class TestQualityMetrics:
    """质量指标"""

    def test_loss_decreases(self):
        """训练后 loss 必须比初始值显著下降"""
        *_, init_loss, _, _ = _train_one_image("1.jpg", steps=500, I_city=0.5)
        *_, mse, _, _, _, _, _ = _train_one_image("1.jpg", steps=3000, I_city=0.5)
        assert mse < 0.05, f"final MSE={mse:.4f} should be significantly improved"

    def test_alpha_reasonable_range(self):
        """α 必须在合理物理范围内 (0~20)"""
        for fname in IMAGES[:3]:
            alpha, *_, _, _ = _train_one_image(fname, steps=500, I_city=0.5)
            assert 0 < alpha < 20, f"{fname}: alpha={alpha:.2f} out of range"


def _save_outputs(base, obs, pred, H, W):
    """保存 PNG 到 output_dir"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs_img = obs.reshape(H, W).numpy()
    pred_img = pred.reshape(H, W).numpy()
    res_img = (obs_img - pred_img).clip(0, 1)

    plt.imsave(f"{OUTPUT_DIR}/{base}_observed.png", obs_img, cmap="gray")
    plt.imsave(f"{OUTPUT_DIR}/{base}_predicted.png", pred_img, cmap="gray")
    plt.imsave(f"{OUTPUT_DIR}/{base}_residual.png", res_img, cmap="gray")
