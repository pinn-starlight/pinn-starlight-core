"""E0、E2、E4 共用的 U-Net 背景估计模型。

网络结构参考 Astro U-net：四次下采样、四次上采样、跳跃连接、每个阶段
两个三乘三卷积，最后使用一个一乘一卷积输出单通道背景图。

接口约定：
- 输入和目标都是二维灰度图，通常已经归一化到 [0, 1]。
- 训练函数返回最佳模型检查点路径。
- 预测函数只使用已经训练好的模型，不重新训练。
- 一个训练好的检查点应复用于所有测试图像。
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from experiments.common.utils import experiment_utils as utils

# TODO:需要让学长检查一下

class _ConvBlock(nn.Sequential):
    """两个三乘三卷积和两个 LeakyReLU 激活层。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )


class UNetSmall(nn.Module):
    """单通道输入、单通道输出的四层 U-Net。"""

    def __init__(self, base_channels: int = 32):
        super().__init__()
        if base_channels <= 0:
            raise ValueError("基础通道数必须大于 0")

        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]

        self.encoder1 = _ConvBlock(1, channels[0])
        self.encoder2 = _ConvBlock(channels[0], channels[1])
        self.encoder3 = _ConvBlock(channels[1], channels[2])
        self.encoder4 = _ConvBlock(channels[2], channels[3])
        self.bottom = _ConvBlock(channels[3], channels[4])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up1 = nn.ConvTranspose2d(channels[4], channels[3], 2, 2)
        self.decoder1 = _ConvBlock(channels[3] * 2, channels[3])
        self.up2 = nn.ConvTranspose2d(channels[3], channels[2], 2, 2)
        self.decoder2 = _ConvBlock(channels[2] * 2, channels[2])
        self.up3 = nn.ConvTranspose2d(channels[2], channels[1], 2, 2)
        self.decoder3 = _ConvBlock(channels[1] * 2, channels[1])
        self.up4 = nn.ConvTranspose2d(channels[1], channels[0], 2, 2)
        self.decoder4 = _ConvBlock(channels[0] * 2, channels[0])

        # 论文最后一层没有激活函数。
        self.output = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, image):
        skip1 = self.encoder1(image)
        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))
        skip4 = self.encoder4(self.pool(skip3))
        hidden = self.bottom(self.pool(skip4))

        hidden = self.decoder1(self._join(self.up1(hidden), skip4))
        hidden = self.decoder2(self._join(self.up2(hidden), skip3))
        hidden = self.decoder3(self._join(self.up3(hidden), skip2))
        hidden = self.decoder4(self._join(self.up4(hidden), skip1))
        return self.output(hidden)

    @staticmethod
    def _join(upsampled, skip):
        if upsampled.shape[-2:] != skip.shape[-2:]:
            upsampled = F.interpolate(
                upsampled,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return torch.cat([upsampled, skip], dim=1)


def build_model(base_channels: int = 32):
    """创建并返回一个尚未训练的 U-Net 模型。"""
    return UNetSmall(base_channels=base_channels)


def train(
    train_manifest,
    validation_manifest,
    output_dir,
    device="cpu",
    epochs: int = 20,
    steps_per_epoch: int = 100,
    batch_size: int = 4,
    patch_size: int = 256,
    learning_rate: float = 1e-3,
    base_channels: int = 32,
    patience: int = 5,
    seed: int = 20260728,
):
    """训练一次 U-Net，并返回最佳模型检查点路径。

    训练数据的每项可以是二元组，也可以是包含 ``observed`` 和
    ``background_true`` 字段的字典。训练目标是背景图。

    返回：
        检查点路径。测试阶段应加载并复用该检查点，不应再次训练。
    """
    if epochs <= 0 or steps_per_epoch <= 0:
        raise ValueError("训练轮数和每轮更新次数必须大于 0")
    if batch_size <= 0:
        raise ValueError("批次大小必须大于 0")
    if patience <= 0:
        raise ValueError("提前停止轮数必须大于 0")
    if patch_size < 16:
        raise ValueError("图像块边长至少需要 16，以支持四次下采样")

    _set_seed(seed)
    train_pairs = _load_pairs(train_manifest)
    validation_pairs = _load_pairs(validation_manifest)
    if not train_pairs or not validation_pairs:
        raise ValueError("训练集和验证集不能为空")

    model = build_model(base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "unet_small_best.pt"
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        _run_training_steps(
            model,
            train_pairs,
            optimizer,
            device,
            steps_per_epoch,
            batch_size,
            patch_size,
            generator,
            description=f"U-Net 第 {epoch + 1}/{epochs} 轮",
        )
        validation_loss = _validation_loss(
            model,
            validation_pairs,
            device,
            patch_size,
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    },
                    "base_channels": int(base_channels),
                    "validation_loss": float(validation_loss),
                    "epoch": int(best_epoch),
                    "seed": int(seed),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    return checkpoint_path


def predict(
    checkpoint,
    observed,
    device="cpu",
    tile_size: int = 256,
    overlap: int = 32,
):
    """加载检查点，对一张图像预测背景并返回二维数组。"""
    model, _ = load_checkpoint_model(checkpoint, device)
    observed_array = (
        utils.load_gray_image(observed)
        if isinstance(observed, (str, Path))
        else np.asarray(observed, dtype=np.float32)
    )
    return predict_with_model(model, observed_array, device, tile_size, overlap)


def load_checkpoint_model(checkpoint, device="cpu"):
    """加载检查点，返回已经切换到评估模式的模型和检查点信息。"""
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    if "model_state" not in checkpoint_data:
        raise ValueError("检查点格式不受支持")

    base_channels = int(checkpoint_data.get("base_channels", 32))
    model = build_model(base_channels)
    model.load_state_dict(checkpoint_data["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint_data


def predict_with_model(
    model,
    observed,
    device="cpu",
    tile_size: int = 256,
    overlap: int = 32,
):
    """使用已经加载的模型预测背景，不进行训练。

    返回二维浮点数组，尺寸与输入图像完全相同。
    """
    observed = np.asarray(observed, dtype=np.float32)
    return _predict_background(model, observed, device, tile_size, overlap)


def single_predict(
    checkpoint,
    input_path,
    device="cpu",
    tile_size: int = 256,
    overlap: int = 32,
):
    """使用检查点处理一张图像，返回输入图、背景图和残差图。"""
    observed = utils.load_gray_image(input_path)
    predicted = predict(
        checkpoint,
        observed,
        device=device,
        tile_size=tile_size,
        overlap=overlap,
    )
    return observed, predicted, observed - predicted


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_pairs(manifest):
    pairs = []
    for item in manifest:
        if isinstance(item, dict):
            observed_path = item["observed"]
            background_path = item["background_true"]
        else:
            observed_path, background_path = item

        observed = utils.load_gray_image(observed_path)
        background_true = utils.load_gray_image(background_path)
        _validate_pair(observed, background_true)
        pairs.append((observed, background_true))
    return pairs


def _validate_pair(observed: np.ndarray, background_true: np.ndarray) -> None:
    if observed.ndim != 2 or background_true.ndim != 2:
        raise ValueError("输入图和真实背景图都必须是二维灰度数组")
    if observed.shape != background_true.shape:
        raise ValueError("输入图和真实背景图的尺寸必须相同")
    if not np.isfinite(observed).all() or not np.isfinite(background_true).all():
        raise ValueError("训练图像不能包含 NaN 或 Inf")


def _run_training_steps(
    model,
    pairs,
    optimizer,
    device,
    steps,
    batch_size,
    patch_size,
    generator,
    description,
):
    model.train()
    progress = tqdm(range(steps), desc=description, file=sys.stdout)
    for step in progress:
        batch_observed, batch_target = _sample_batch(
            pairs,
            batch_size,
            patch_size,
            generator,
        )
        batch_observed = batch_observed.to(device)
        batch_target = batch_target.to(device)

        prediction = model(batch_observed)
        training_loss = torch.mean(torch.abs(prediction - batch_target))
        optimizer.zero_grad(set_to_none=True)
        training_loss.backward()
        optimizer.step()

        if step % 20 == 0:
            progress.set_postfix(loss=f"{training_loss.detach().item():.6f}")


def _sample_batch(pairs, batch_size, patch_size, generator):
    observed_patches = []
    target_patches = []

    for _ in range(batch_size):
        pair_index = int(torch.randint(len(pairs), (1,), generator=generator).item())
        observed, target = pairs[pair_index]
        observed = _pad_to_patch(observed, patch_size)
        target = _pad_to_patch(target, patch_size)

        max_y = observed.shape[0] - patch_size
        max_x = observed.shape[1] - patch_size
        top = int(torch.randint(max_y + 1, (1,), generator=generator).item())
        left = int(torch.randint(max_x + 1, (1,), generator=generator).item())

        observed_patch = observed[top : top + patch_size, left : left + patch_size]
        target_patch = target[top : top + patch_size, left : left + patch_size]
        observed_patches.append(torch.from_numpy(observed_patch).unsqueeze(0))
        target_patches.append(torch.from_numpy(target_patch).unsqueeze(0))

    return torch.stack(observed_patches), torch.stack(target_patches)


def _pad_to_patch(image: np.ndarray, patch_size: int) -> np.ndarray:
    pad_y = max(0, patch_size - image.shape[0])
    pad_x = max(0, patch_size - image.shape[1])
    if pad_y == 0 and pad_x == 0:
        return image
    mode = "reflect" if pad_y < image.shape[0] and pad_x < image.shape[1] else "edge"
    return np.pad(image, ((0, pad_y), (0, pad_x)), mode=mode)


def _validation_loss(model, pairs, device, tile_size):
    overlap = min(32, tile_size // 4)
    losses = [
        np.mean(np.abs(_predict_background(model, observed, device, tile_size, overlap) - target))
        for observed, target in pairs
    ]
    return float(np.mean(losses))


def _predict_background(model, observed, device, tile_size, overlap):
    observed = np.asarray(observed, dtype=np.float32)
    if observed.ndim != 2:
        raise ValueError("输入图必须是二维灰度数组")
    if tile_size < 16:
        raise ValueError("推理图块边长至少需要 16")
    if not 0 <= overlap < tile_size:
        raise ValueError("重叠宽度必须位于 [0, 推理图块边长) 范围内")

    original_height, original_width = observed.shape
    padded = _pad_to_patch(observed, tile_size)
    prediction_sum = np.zeros_like(padded, dtype=np.float32)
    prediction_count = np.zeros_like(padded, dtype=np.float32)
    stride = tile_size - overlap
    starts_y = _tile_starts(padded.shape[0], tile_size, stride)
    starts_x = _tile_starts(padded.shape[1], tile_size, stride)

    model.eval()
    with torch.no_grad():
        for top in starts_y:
            for left in starts_x:
                patch = padded[top : top + tile_size, left : left + tile_size]
                tensor = (
                    torch.from_numpy(np.ascontiguousarray(patch))
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                )
                patch_prediction = (
                    model(tensor).squeeze(0).squeeze(0).cpu().numpy()
                )
                prediction_sum[
                    top : top + tile_size,
                    left : left + tile_size,
                ] += patch_prediction
                prediction_count[
                    top : top + tile_size,
                    left : left + tile_size,
                ] += 1.0

    prediction = prediction_sum / np.maximum(prediction_count, 1.0)
    return prediction[:original_height, :original_width]


def _tile_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts
