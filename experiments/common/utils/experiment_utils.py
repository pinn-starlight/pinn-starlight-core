"""正式实验共用的数据、随机种子和结果保存工具。"""
import csv
import json
import os
import platform
import random
import shutil
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


def set_seed(seed: int):
    """固定 Python、NumPy、PyTorch 和 CUDA 的随机数源。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def convert_absol_path(path):
    """把 manifest 中的项目相对路径转换为绝对路径。"""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path):
    """返回便于移动项目的相对路径。"""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def prepare_output_root(path, force = False) -> Path:
    """
        创建输出根目录，并且可以用参数强制覆盖
        return Path
    """
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return path
    if not path.is_dir():
        raise NotADirectoryError(f"输出路径不是目录：{path}")
    if not any(path.iterdir()):
        return path
    if not force:
        raise FileExistsError(
            f"输出目录非空，不会覆盖已有结果：{path}。"
            "如确认要重跑，请添加 --force，或指定新的 --output-root。"
        )
    # path.mkdir(parents=True, exist_ok=True)
    _remove_forced_output(path)
    return path


def _remove_forced_output(path: Path):
    if path.is_symlink():
        raise ValueError("--force 不允许清理符号链接输出目录")
    resolved_path = path.resolve()
    resolved_output_root = OUTPUT_ROOT.resolve()
    if (
        resolved_path == resolved_output_root
        or resolved_output_root not in resolved_path.parents
    ):
        raise ValueError(
            "--force 只允许清理 experiments/outputs 下的输出目录："
            f"{resolved_path}"
        )
    shutil.rmtree(resolved_path)


def load_gray_image(path, downsample = 1) -> np.ndarray:
    """读取一张线性灰度图，返回范围为 [0, 1] 的 float32 数组。"""
    path = convert_absol_path(path)

    loader = ImageLoader(path=path, downsample=downsample)
    _, gray, _, _ = loader.get_gray_data()

    return gray


def load_manifest(path=SYNTHETIC_MANIFEST, split: str | None = None):
    """读取正式合成数据 manifest，可按 split 过滤。"""
    path = convert_absol_path(path)
    filtered_rows = []
    if not path.is_file():
        raise FileNotFoundError(f"合成数据 manifest 不存在：{path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if split is not None:
        for row in rows:
            if row["split"] == split:
                filtered_rows.append(row)
        rows = filtered_rows
    return rows


def load_synthetic_sample(row: dict):
    """从一行 manifest 读取 clean、background、observed 和元数据。"""
    metadata_path = convert_absol_path(row["metadata"])
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


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_run_metadata(output_dir: Path, config: dict):
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


def save_display_png(path, image, clip = True):
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
):
    """保存浮点 TIFF，并另存同名 PNG 用于论文排版。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "observed": observed,
        "background_pred": background_pred,
        "residual_pred": residual_pred,
    }
    for name, image in arrays.items():
        save_display_png(output_dir / f"{name}.png", image)


def save_rows_csv(path, rows: list[dict]):
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


def summarize_rows(rows: list[dict], group_key: str):
    """按方法或消融版本汇总数值列的均值、标准差和样本数。"""
    summary = {}
    for group in sorted({str(row[group_key]) for row in rows}):
        group_rows = [row for row in rows if str(row[group_key]) == group]
        numeric_keys = sorted(
            {
                key
                for row in group_rows
                for key, value in row.items()
                if key not in {group_key, "seed", "step"} and _is_finite_number(value)
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


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def reset_peak_vram(device):
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_vram_mb(device):
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        return float(torch.cuda.max_memory_allocated(device) / (1024**2))
    return None


def _git_output(*args):
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


def _is_finite_number(value):
    return isinstance(value, (int, float, np.number)) and np.isfinite(value)
