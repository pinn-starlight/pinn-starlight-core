"""正式实验共用的数据、随机种子和结果保存工具。"""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pinn_starlight_core.data.image_loader import ImageLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "experiments/outputs"
SYNTHETIC_ROOT = PROJECT_ROOT / "data/collections/synthetic"
SYNTHETIC_MANIFEST = SYNTHETIC_ROOT / "manifest.csv"
SEEDS = (20260728, 20260729, 20260730)

_RAW_EXTENSIONS = {".cr2", ".nef", ".dng", ".arw"}


def set_seed(seed: int) -> None:
    """固定 Python、NumPy、PyTorch 和 CUDA 的随机数源。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path) -> Path:
    """把 manifest 中的项目相对路径转换为绝对路径。"""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path) -> str:
    """尽量返回便于移动项目的相对路径。"""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_gray_image(path, downsample: int = 1) -> np.ndarray:
    """读取一张线性灰度图，返回范围为 [0, 1] 的 float32 数组。"""
    path = resolve_path(path)
    if downsample <= 0:
        raise ValueError("downsample 必须大于 0")
    if not path.is_file():
        raise FileNotFoundError(f"图像不存在：{path}")

    if path.suffix.lower() in _RAW_EXTENSIONS:
        loader = ImageLoader(str(path), device="cpu", downsample=downsample)
        rgb = loader.rgb_data
        gray = (
            0.2126 * rgb[:, :, 0]
            + 0.7152 * rgb[:, :, 1]
            + 0.0722 * rgb[:, :, 2]
        )
    else:
        with Image.open(path) as image:
            data = np.asarray(image)
        if data.ndim == 3:
            data = data[:, :, :3]
            gray = (
                0.2126 * data[:, :, 0]
                + 0.7152 * data[:, :, 1]
                + 0.0722 * data[:, :, 2]
            )
        elif data.ndim == 2:
            gray = data
        else:
            raise ValueError(f"不支持的图像形状：{data.shape}")

        if np.issubdtype(data.dtype, np.integer):
            gray = gray.astype(np.float32) / float(np.iinfo(data.dtype).max)

    gray = np.asarray(gray, dtype=np.float32)
    if path.suffix.lower() not in _RAW_EXTENSIONS:
        gray = gray[::downsample, ::downsample]
    return np.clip(gray, 0.0, 1.0)


def load_manifest(path=SYNTHETIC_MANIFEST, split: str | None = None) -> list[dict]:
    """读取正式合成数据 manifest，可按 split 过滤。"""
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"合成数据 manifest 不存在：{path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if split is not None:
        rows = [row for row in rows if row.get("split") == split]
    return rows


def load_synthetic_sample(row: dict) -> dict:
    """从一行 manifest 读取 clean、background、observed 和元数据。"""
    metadata_path = resolve_path(row["metadata"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "sample_id": row["sample_id"],
        "split": row["split"],
        "background_type": row["background_type"],
        "clean_true": load_gray_image(row["clean_true"]),
        "background_true": load_gray_image(row["background_true"]),
        "observed": load_gray_image(row["observed"]),
        "metadata": metadata,
    }


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_run_metadata(output_dir: Path, config: dict) -> None:
    """保存配置、环境、Git commit 和工作区状态。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    commit = _git_output("rev-parse", "HEAD")
    dirty = bool(_git_output("status", "--porcelain"))
    environment = {
        "created_at": datetime.now().astimezone().isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cpu_count": os.cpu_count(),
        "git_commit": commit or "unknown",
        "git_dirty": dirty,
    }
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "environment.json", environment)


def save_float_tiff(path, image) -> None:
    """以 32 位浮点 TIFF 保存指标计算使用的原始数组。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image, dtype=np.float32)
    Image.fromarray(array, mode="F").save(path)


def save_display_png(path, image, clip: bool = True) -> None:
    """保存论文展示用的 8 位灰度 PNG。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image, dtype=np.float32)
    if clip:
        array = np.clip(array, 0.0, 1.0)
    else:
        low, high = np.percentile(array, [1.0, 99.0])
        array = (array - low) / max(float(high - low), 1e-8)
    Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="L").save(path)


def save_prediction_arrays(
    output_dir: Path,
    observed,
    background_pred,
    residual_pred,
) -> None:
    """保存浮点 TIFF，并另存同名 PNG 用于论文排版。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "observed": observed,
        "background_pred": background_pred,
        "residual_pred": residual_pred,
    }
    for name, image in arrays.items():
        save_float_tiff(output_dir / f"{name}.tif", image)
        save_display_png(output_dir / f"{name}.png", image)


def save_rows_csv(path, rows: list[dict]) -> None:
    """把一组同类结果写成 CSV；空结果也会写出空文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def summarize_rows(rows: list[dict], group_key: str) -> dict:
    """按方法或消融版本汇总数值列的均值、标准差和样本数。"""
    summary = {}
    for group in sorted({str(row[group_key]) for row in rows}):
        group_rows = [row for row in rows if str(row[group_key]) == group]
        numeric_keys = sorted(
            {
                key
                for row in group_rows
                for key, value in row.items()
                if key not in {group_key, "seed", "step"}
                and _is_finite_number(value)
            }
        )
        values: dict[str, int | float] = {"count": len(group_rows)}
        for key in numeric_keys:
            column = [float(row[key]) for row in group_rows if _is_finite_number(row.get(key))]
            if column:
                values[f"{key}_mean"] = float(np.mean(column))
                values[f"{key}_std"] = float(np.std(column, ddof=0))
        summary[group] = values
    return summary


def parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def reset_peak_vram(device) -> None:
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_vram_mb(device):
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        return float(torch.cuda.max_memory_allocated(device) / (1024**2))
    return None


def _git_output(*args) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return project_relative(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_value(value), ensure_ascii=False)
    if isinstance(value, Path):
        return project_relative(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float, np.number)) and np.isfinite(value)
