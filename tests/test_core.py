import torch
import pytest

from pinn_starlight_core.data.FakeRAW import FakeRaw
from pinn_starlight_core.nn.Layers import SkyglowLinear, SkyglowMLP
from pinn_starlight_core.nn.Losses import MSEData
from pinn_starlight_core.utils.Laplacian import laplacian


# ============================================================
# FakeRaw
# ============================================================
class TestFakeRaw:
    def test_output_shapes(self):
        fake = FakeRaw(H=64, W=64, n_stars=3, seed=0)
        coords, values = fake.get_fake_raw()
        assert coords.shape == (64 * 64, 2)
        assert values.shape == (64 * 64,)

    def test_seed_reproducibility(self):
        a = FakeRaw(H=32, W=32, seed=42).get_fake_raw()[1]
        b = FakeRaw(H=32, W=32, seed=42).get_fake_raw()[1]
        assert torch.allclose(a, b)

    def test_diff_seeds_diff_stars(self):
        a = FakeRaw(H=32, W=32, seed=0).get_fake_raw()[1]
        b = FakeRaw(H=32, W=32, seed=1).get_fake_raw()[1]
        assert not torch.allclose(a, b)           # 不同 seed 星点不同


# ============================================================
# SkyglowLinear
# ============================================================
class TestSkyglowLinear:
    def test_forward_shape(self):
        layer = SkyglowLinear(2, 64)
        x = torch.randn(128, 2)
        out = layer.forward(x)
        assert out.shape == (128, 64)

    def test_gradient_flows(self):
        layer = SkyglowLinear(2, 1)
        x = torch.randn(32, 2)
        out = layer.forward(x)
        loss = out.mean()
        loss.backward()
        assert layer.W_l.grad is not None
        assert layer.W_l.grad.abs().sum() > 0       # 梯度非零

    def test_gradient_shapes(self):
        layer = SkyglowLinear(2, 64)
        x = torch.randn(128, 2)
        out = layer.forward(x)
        out.mean().backward()
        assert layer.W_l.grad.shape == layer.W_l.shape
        assert layer.b_l.grad.shape == layer.b_l.shape


# ============================================================
# SkyglowMLP
# ============================================================
class TestSkyglowMLP:
    def test_forward_shape(self):
        mlp = SkyglowMLP([2, 64, 32, 1])
        out = mlp.forward(torch.randn(256, 2))
        assert out.shape == (256, 1)

    def test_deep_network_forward(self):
        mlp = SkyglowMLP([2, 16, 16, 16, 1])
        out = mlp.forward(torch.randn(128, 2))
        assert out.shape == (128, 1)

    def test_parameters_not_empty(self):
        mlp = SkyglowMLP([2, 64, 32, 1])
        params = list(mlp.parameters())
        assert len(params) > 0

    def test_gradient_flows_through_all_layers(self):
        mlp = SkyglowMLP([2, 16, 8, 1])
        x = torch.randn(64, 2)
        out = mlp.forward(x)
        out.mean().backward()
        for layer in mlp.layers:
            if isinstance(layer, SkyglowLinear):
                assert layer.W_l.grad is not None
                assert layer.W_l.grad.abs().sum() > 0


# ============================================================
# MSEData
# ============================================================
class TestMSEData:
    def test_perfect_prediction(self):
        loss_fn = MSEData()
        t = torch.ones(100)
        loss = loss_fn.forward(t, t.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_positive(self):
        loss_fn = MSEData()
        loss = loss_fn.forward(torch.zeros(100), torch.ones(100))
        assert loss.item() > 0

    def test_backward_shape(self):
        loss_fn = MSEData()
        obs = torch.randn(64)
        pred = torch.randn(64)
        loss_fn.forward(obs, pred)
        grad = loss_fn.backward()
        assert grad.shape == obs.shape


# ============================================================
# Laplacian
# ============================================================
class TestLaplacian:
    def test_gradient_graph_connected(self):
        """autograd 能从 loss 经 ∇² 回传梯度到网络"""
        mlp = SkyglowMLP([2, 8, 1])
        xy = torch.randn(16, 2).requires_grad_(True)
        I = mlp.forward(xy).squeeze()
        lap = laplacian(I, xy)
        loss = lap.mean()
        loss.backward()
        # 每个 SkyglowLinear 都收到了梯度
        for layer in mlp.layers:
            if isinstance(layer, SkyglowLinear):
                assert layer.W_l.grad is not None
                assert layer.W_l.grad.abs().sum() > 0

    def test_quadratic(self):
        """f(x,y) = x² + y², ∇²f = 4"""
        xy = torch.randn(16, 2).requires_grad_(True)
        f = xy[:, 0]**2 + xy[:, 1]**2
        lap = laplacian(f, xy)
        assert lap.mean().item() == pytest.approx(4.0, abs=0.01)

    def test_output_shape(self):
        xy = torch.randn(64, 2).requires_grad_(True)
        I = (xy[:, 0]**2 + xy[:, 1]**2).requires_grad_(True)
        lap = laplacian(I, xy)
        assert lap.shape == (64,)


# ============================================================
# 端到端：合成数据训练一个 epoch
# ============================================================
def test_end_to_end_synthetic():
    torch.manual_seed(42)
    fake = FakeRaw(H=32, W=32, n_stars=3, seed=0)
    coords, I_obs = fake.get_fake_raw()

    bg = 0.3 * torch.cos(3.0 * coords[:, 0]) * torch.cos(3.0 * coords[:, 1])
    I_city = (18.0 + 9.0) * bg

    mlp = SkyglowMLP([2, 32, 1])
    opt = torch.optim.Adam(mlp.parameters(), lr=0.01)
    ld = MSEData()

    initial_loss = None
    for step in range(200):
        idx = torch.randint(0, len(coords), (256,))
        xy = coords[idx].clone().requires_grad_(True)
        Ip = mlp.forward(xy).squeeze()
        loss = ld.forward(I_obs[idx], Ip)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            initial_loss = loss.item()

    assert loss.item() < initial_loss                 # loss 下降
    assert loss.item() < 1.0                          # 收敛到合理范围
