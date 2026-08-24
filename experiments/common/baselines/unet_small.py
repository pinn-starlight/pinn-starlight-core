"""E0、E2、E4 共用的 U-Net-small 监督式背景估计 baseline。

职责约定：
- 输入 observed，监督目标为 background_true，训练损失使用 MSE。
- E0 可调用 single_train 在单个样本上短暂训练并检查完整流程。
- 正式实验调用 train，使用验证集早停并保存最优 checkpoint。
- 训练阶段随机裁剪、翻转和旋转；整图预测采用重叠分块以控制显存。
- 返回浮点 background_pred，不在本模块中裁剪 residual_pred 或保存展示图。
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from experiments.common.utils import experiment_utils as utils


class _ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )


class UNetSmall(nn.Module):
    """两层下采样的单通道 U-Net。"""

    def __init__(self, base_channels: int = 16):
        super().__init__()
        self.encoder_1 = _ConvBlock(1, base_channels)
        self.encoder_2 = _ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = _ConvBlock(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)

        self.up_2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2,
        )
        self.decoder_2 = _ConvBlock(base_channels * 4, base_channels * 2)
        self.up_1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2,
        )
        self.decoder_1 = _ConvBlock(base_channels * 2, base_channels)
        self.output = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, image):
        skip_1 = self.encoder_1(image)
        skip_2 = self.encoder_2(self.pool(skip_1))
        hidden = self.bottleneck(self.pool(skip_2))

        hidden = self.up_2(hidden)
        hidden = self.decoder_2(torch.cat([hidden, skip_2], dim=1))
        hidden = self.up_1(hidden)
        hidden = self.decoder_1(torch.cat([hidden, skip_1], dim=1))
        return torch.sigmoid(self.output(hidden))


def build_model(base_channels: int = 16):
    """创建最终需要在论文中准确描述的 U-Net-small。"""
    if base_channels <= 0:
        raise ValueError("base_channels 必须大于 0")
    return UNetSmall(base_channels=base_channels)


def single_train(
    observed_path,
    background_true_path,
    device,
    steps: int,
    batch_size: int,
    patch_size: int = 256,
    learning_rate: float = 1e-3,
    base_channels: int = 16,
    seed: int = 20260728,
):
    """在单个合成样本上短暂训练，供 E0 检查完整流程。"""
    _set_seed(seed)
    observed = _load_gray(observed_path)
    background_true = _load_gray(background_true_path)
    _validate_pair(observed, background_true)

    model = build_model(base_channels).to(device)
    _initialize_output_layer(model, [background_true])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    _run_training_steps(
        model,
        [(observed, background_true)],
        optimizer,
        device,
        steps,
        batch_size,
        patch_size,
        generator,
        description="U-Net-small",
    )

    overlap = min(32, patch_size // 4)
    predicted = _predict_background(model, observed, device, patch_size, overlap)
    residual = observed - predicted
    return observed, predicted, residual


def train(
    train_manifest,
    validation_manifest,
    output_dir,
    device=None,
    epochs: int = 20,
    steps_per_epoch: int = 100,
    batch_size: int = 4,
    patch_size: int = 256,
    learning_rate: float = 1e-3,
    base_channels: int = 16,
    patience: int = 5,
    seed: int = 20260728,
):
    """训练并返回验证集最优 checkpoint 的路径。"""
    if epochs <= 0 or steps_per_epoch <= 0:
        raise ValueError("epochs 和 steps_per_epoch 必须大于 0")
    if patience <= 0:
        raise ValueError("patience 必须大于 0")

    _set_seed(seed)
    train_pairs = _load_pairs(train_manifest)
    validation_pairs = _load_pairs(validation_manifest)
    if not train_pairs or not validation_pairs:
        raise ValueError("训练集和验证集都不能为空")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(base_channels).to(device)
    _initialize_output_layer(model, [background for _, background in train_pairs])
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
            description=f"U-Net epoch {epoch + 1}/{epochs}",
        )
        validation_loss = _validation_loss(
            model,
            validation_pairs,
            device,
            patch_size,
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    },
                    "base_channels": base_channels,
                    "validation_loss": validation_loss,
                    "epoch": epoch + 1,
                    "seed": seed,
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
    device=None,
    tile_size: int = 256,
    overlap: int = 32,
):
    """返回与 observed 同尺寸的 background_pred。"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint_model(checkpoint, device)

    observed_array = (
        _load_gray(observed)
        if isinstance(observed, (str, Path))
        else np.asarray(observed, dtype=np.float32)
    )
    return predict_with_model(model, observed_array, device, tile_size, overlap)


def load_checkpoint_model(checkpoint, device=None):
    """加载一次 checkpoint，供 E2/E4 对多张图重复推理。"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(checkpoint_data["base_channels"])
    model.load_state_dict(checkpoint_data["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint_data


def predict_with_model(
    model,
    observed,
    device=None,
    tile_size: int = 256,
    overlap: int = 32,
):
    """使用已加载模型预测一张二维 observed。"""
    device = device or next(model.parameters()).device
    observed = np.asarray(observed, dtype=np.float32)
    return _predict_background(model, observed, device, tile_size, overlap)


def single_predict(
    checkpoint,
    input_path,
    device=None,
    tile_size: int = 256,
    overlap: int = 32,
):
    """读取单张图片并返回 observed、background_pred 和 residual_pred。"""
    observed = _load_gray(input_path)
    predicted = predict(
        checkpoint,
        observed,
        device=device,
        tile_size=tile_size,
        overlap=overlap,
    )
    residual = observed - predicted
    return observed, predicted, residual


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _initialize_output_layer(model, targets) -> None:
    """Start near the target mean instead of sigmoid(0)=0.5."""
    target_mean = float(np.mean([np.mean(target) for target in targets]))
    ratio = np.clip(target_mean, 1e-3, 1.0 - 1e-3)
    bias = float(np.log(ratio / (1.0 - ratio)))
    with torch.no_grad():
        model.output.bias.fill_(bias)


def _load_gray(path) -> np.ndarray:
    return utils.load_gray_image(path)


def _load_pairs(manifest):
    pairs = []
    for item in manifest:
        if isinstance(item, dict):
            observed_path = item["observed"]
            background_path = item["background_true"]
        else:
            observed_path, background_path = item

        observed = _load_gray(observed_path)
        background_true = _load_gray(background_path)
        _validate_pair(observed, background_true)
        pairs.append((observed, background_true))
    return pairs


def _validate_pair(observed: np.ndarray, background_true: np.ndarray) -> None:
    if observed.ndim != 2 or background_true.ndim != 2:
        raise ValueError("observed 和 background_true 都必须是二维灰度数组")
    if observed.shape != background_true.shape:
        raise ValueError("observed 与 background_true 的尺寸不一致")
    if not np.isfinite(observed).all() or not np.isfinite(background_true).all():
        raise ValueError("训练图片包含 NaN 或 Inf")


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
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps 和 batch_size 必须大于 0")
    if patch_size <= 0 or patch_size % 4 != 0:
        raise ValueError("patch_size 必须是大于 0 的 4 的倍数")

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
        training_loss = torch.mean((prediction - batch_target) ** 2)
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
        observed_patch = torch.from_numpy(
            np.ascontiguousarray(observed[top : top + patch_size, left : left + patch_size])
        )
        target_patch = torch.from_numpy(
            np.ascontiguousarray(target[top : top + patch_size, left : left + patch_size])
        )

        rotations = int(torch.randint(4, (1,), generator=generator).item())
        observed_patch = torch.rot90(observed_patch, rotations, dims=(0, 1))
        target_patch = torch.rot90(target_patch, rotations, dims=(0, 1))
        if bool(torch.randint(2, (1,), generator=generator).item()):
            observed_patch = torch.flip(observed_patch, dims=(1,))
            target_patch = torch.flip(target_patch, dims=(1,))
        if bool(torch.randint(2, (1,), generator=generator).item()):
            observed_patch = torch.flip(observed_patch, dims=(0,))
            target_patch = torch.flip(target_patch, dims=(0,))

        observed_patches.append(observed_patch.unsqueeze(0))
        target_patches.append(target_patch.unsqueeze(0))

    return torch.stack(observed_patches), torch.stack(target_patches)


def _pad_to_patch(image: np.ndarray, patch_size: int) -> np.ndarray:
    pad_y = max(0, patch_size - image.shape[0])
    pad_x = max(0, patch_size - image.shape[1])
    if pad_y == 0 and pad_x == 0:
        return image
    mode = "reflect" if min(image.shape) > 1 else "edge"
    return np.pad(image, ((0, pad_y), (0, pad_x)), mode=mode)


def _validation_loss(model, pairs, device, tile_size):
    losses = []
    overlap = min(32, tile_size // 4)
    for observed, background_true in pairs:
        prediction = _predict_background(
            model,
            observed,
            device,
            tile_size,
            overlap,
        )
        losses.append(float(np.mean((prediction - background_true) ** 2)))
    return float(np.mean(losses))


def _predict_background(model, observed, device, tile_size, overlap):
    observed = np.asarray(observed, dtype=np.float32)
    if observed.ndim != 2:
        raise ValueError("observed 必须是二维灰度数组")
    if not 0 <= overlap < tile_size:
        raise ValueError("overlap 必须位于 [0, tile_size) 范围内")
    if tile_size % 4 != 0:
        raise ValueError("tile_size 必须是 4 的倍数")

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
                tensor = torch.from_numpy(np.ascontiguousarray(patch))
                tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
                patch_prediction = model(tensor).squeeze(0).squeeze(0).cpu().numpy()
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


def _tile_starts(length: int, tile_size: int, stride: int):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts
