"""正式实验共用的数据、随机种子和结果保存工具。"""
import argparse
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
import tifffile as tif
from PIL import Image

from pinn_starlight_core.data.image_loader import ImageLoader
from experiments.configs.formal_experiment import (
    FORMAL_SEEDS,
    FOURIER_LS_CONFIG,
    formal_config_snapshot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "experiments/outputs"


def _resolve_data_root() -> Path:
    """Resolve the external data root without changing the experiment protocol.

    Supported layouts, in priority order:
    1. ``PINN_STARLIGHT_DATA_ROOT`` points to a directory containing
       ``collections/`` and ``test/``;
    2. root-level ``collections/`` / ``test/`` beside the project files;
    3. legacy ``data/collections`` / ``data/test``.
    """
    configured = os.environ.get("PINN_STARLIGHT_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    root_layout = PROJECT_ROOT
    legacy_layout = PROJECT_ROOT / "data"
    if (root_layout / "collections").exists() or (root_layout / "test").exists():
        return root_layout
    if (legacy_layout / "collections").exists() or (legacy_layout / "test").exists():
        return legacy_layout

    # The user's current formal layout is root-level. Returning PROJECT_ROOT when
    # no data are mounted also keeps imports side-effect free; preflight gives a
    # precise error before any formal run if the data are elsewhere.
    return root_layout


DATA_ROOT = _resolve_data_root()
COLLECTION_ROOT = DATA_ROOT / "collections"
COLLECTION_MANIFEST = COLLECTION_ROOT / "manifest.csv"
TEST_ROOT = DATA_ROOT / "test"
NATIVE_TEST_IMAGE = TEST_ROOT / "native_test.tif"
REAL_HIGH_POLLUTION_ROOT = COLLECTION_ROOT / "real_high_pollution"

# ``collections/synthetic`` is the legacy 10-sample set in the user's current
# data tree. Formal Stage-7+ data are isolated in a new directory.
LEGACY_SYNTHETIC_ROOT = COLLECTION_ROOT / "synthetic"
SYNTHETIC_ROOT = COLLECTION_ROOT / "synthetic_formal"
SYNTHETIC_MANIFEST = SYNTHETIC_ROOT / "manifest.csv"
SELECTION_ROOT = OUTPUT_ROOT / "selection"
PINN_SELECTION_ROOT = SELECTION_ROOT / "pinn"
GAUSSIAN_SELECTION_ROOT = SELECTION_ROOT / "fft_gaussian"
FOURIER_LS_SELECTION_ROOT = SELECTION_ROOT / "fourier_ls"
UNET_SELECTION_ROOT = SELECTION_ROOT / "unet_small"
PINN_SELECTION_ARTIFACT = PINN_SELECTION_ROOT / "pinn_selected.json"
GAUSSIAN_SELECTION_ARTIFACT = GAUSSIAN_SELECTION_ROOT / "gaussian_selected.json"
FOURIER_LS_SELECTION_ARTIFACT = FOURIER_LS_SELECTION_ROOT / "fourier_ls_selected.json"
UNET_SELECTION_ARTIFACT = UNET_SELECTION_ROOT / "unet_selected.json"
GLOBAL_CONFIGURATION_LOCK = OUTPUT_ROOT / "configuration_lock/global_configuration_lock.json"
SEEDS = tuple(FORMAL_SEEDS)


def set_seed(seed: int):
    """固定 Python、NumPy、PyTorch 和 CUDA 的随机数源。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_absolute_path(path):
    """把 manifest 中的项目相对路径转换为绝对路径。"""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_collection_path(path) -> Path:
    """Resolve one ``collections/manifest.csv`` path in all supported layouts."""
    path = Path(path)
    if path.is_absolute():
        return path
    parts = path.parts
    if len(parts) >= 2 and parts[:2] == ("data", "collections"):
        return DATA_ROOT / Path(*parts[1:])
    if parts and parts[0] == "collections":
        return DATA_ROOT / path
    return COLLECTION_ROOT / path


def supported_image_files(directory: Path):
    """Return deterministic real-image candidates from one directory."""
    directory = Path(directory)
    extensions = {
        ".nef", ".cr2", ".cr3", ".arw", ".dng",
        ".tif", ".tiff", ".png", ".jpg", ".jpeg",
    }
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


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
    _remove_forced_output(path)
    return path


def load_gray_image(path, downsample = 1) -> np.ndarray:
    """读取一张线性灰度图，返回范围为 [0, 1] 的 float32 数组。"""
    path = to_absolute_path(path)

    loader = ImageLoader(path=path, downsample=downsample)
    _, gray, W, H = loader.get_gray_data()

    gray = np.asarray(gray, dtype=np.float32)
    gray = gray.reshape(H, W)

    return np.clip(gray, 0.0, 1.0).astype(np.float32)


def load_manifest(path=SYNTHETIC_MANIFEST, split: str | None = None):
    """读取正式合成数据 manifest，可按 split 过滤。"""
    path = to_absolute_path(path)
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


def load_float_tiff(path) -> np.ndarray:
    """Load a formal synthetic float TIFF without clipping or renormalizing it."""
    path = to_absolute_path(path)
    array = np.asarray(tif.imread(path), dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"synthetic TIFF 必须是二维灰度图，实际为 {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"synthetic TIFF 包含 NaN 或 Inf：{path}")
    return array


def load_observed(observed):
    """Validate the formal Method input without reading files or clipping values."""
    if not isinstance(observed, np.ndarray):
        raise TypeError(
            "formal Method input must be an observed np.ndarray; "
            "load files in the data/experiment layer before calling a Method"
        )
    array = np.asarray(observed, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"observed 必须是二维灰度图，实际为 {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("observed 包含 NaN 或 Inf")
    return array.astype(np.float32, copy=False)


def load_synthetic_sample(row: dict):
    """Load one synthetic sample; truth stays separate from the method input."""
    metadata_path = to_absolute_path(row["metadata"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "sample_id": row["sample_id"],
        "base_id": row.get("base_id", metadata.get("base_id")),
        "split": row["split"],
        "background_type": row["background_type"],
        "intensity_level": row.get("intensity_level", metadata.get("intensity_level")),
        "clean_true": load_float_tiff(row["clean_true"]),
        "background_true": load_float_tiff(row["background_true"]),
        "observed": load_float_tiff(row["observed"]),
        "metadata": metadata,
    }


def reconstruct_residual(observed, background_pred) -> np.ndarray:
    """Common reconstruction used by every method: R_hat = observed - B_hat."""
    observed_array = np.asarray(observed, dtype=np.float32)
    background_array = np.asarray(background_pred, dtype=np.float32)
    if observed_array.shape != background_array.shape:
        raise ValueError(
            f"reconstruction shape mismatch: {observed_array.shape} vs {background_array.shape}"
        )
    if observed_array.ndim != 2:
        raise ValueError("reconstruction inputs must be 2-D grayscale arrays")
    if not np.isfinite(observed_array).all() or not np.isfinite(background_array).all():
        raise ValueError("reconstruction inputs contain NaN or Inf")
    return (observed_array - background_array).astype(np.float32)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_json(path):
    """Read one JSON file from the experiment layer."""
    path = to_absolute_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path) -> str:
    """Return a stable SHA-256 fingerprint for one file."""
    import hashlib

    path = to_absolute_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_global_configuration_lock(
    path=GLOBAL_CONFIGURATION_LOCK,
    verify=True,
    require_formal_ready=False,
):
    """Load the Stage-10 frozen configuration and verify immutable dependencies."""
    path = to_absolute_path(path)
    lock = read_json(path)
    if lock.get("lock_state") != "FROZEN":
        raise ValueError(f"Global configuration lock is not FROZEN: {path}")
    if int(lock.get("schema_version", 0)) < 1:
        raise ValueError(f"Unsupported global configuration lock schema: {path}")
    if require_formal_ready and not bool(lock.get("formal_test_ready", False)):
        raise RuntimeError("Global Configuration Lock is not complete for formal test")

    methods = lock.get("methods", {})
    required_methods = ("fft_gaussian", "fourier_ls", "unet_small", "pinn")
    for method in required_methods:
        if method not in methods:
            raise ValueError(f"Global lock is missing method configuration: {method}")

    if require_formal_ready:
        if int(lock.get("schema_version", 0)) < 5:
            raise ValueError("Stage-10 formal test requires Global Lock schema >= 5")
        boundary = lock.get("boundary", {})
        if boundary.get("test_arrays_loaded_during_lock") is not False:
            raise RuntimeError("Global Lock must attest that no test arrays were loaded")
        if boundary.get("selection_or_training_performed_during_lock") is not False:
            raise RuntimeError("Stage-10 Global Lock must be a pure freeze gate")

    fourier_config = methods["fourier_ls"]
    if "weight" in fourier_config or "physics_weight" in fourier_config:
        raise ValueError("Legacy Fourier-LS weight key found; rebuild the Global Lock")
    try:
        lambda_ls = float(fourier_config["lambda_ls"])
        sparse_weight = float(fourier_config["sparse_weight"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Stage-10 Fourier-LS requires lambda_ls and sparse_weight") from exc
    if not np.isfinite(lambda_ls) or lambda_ls <= 0.0:
        raise ValueError("Formal Fourier-LS lambda_ls must be finite and positive")
    if not np.isfinite(sparse_weight) or sparse_weight < 0.0:
        raise ValueError("Formal Fourier-LS sparse_weight must be finite and non-negative")
    expected_fourier = FOURIER_LS_CONFIG["method"]
    if float(fourier_config.get("alpha", 0.0)) != float(expected_fourier["alpha"]):
        raise ValueError("Formal Fourier-LS alpha does not match central config")
    if fourier_config.get("boundary_condition") != expected_fourier["boundary_condition"]:
        raise ValueError("Formal Fourier-LS boundary condition does not match central config")

    if verify:
        from experiments.common.utils import selection_artifacts as artifact_utils

        formal = lock.get("formal_config", {})
        current_snapshot = formal_config_snapshot()
        current_config_sha = artifact_utils.stable_json_sha256(current_snapshot)
        if artifact_utils.stable_json_sha256(formal.get("snapshot")) != formal.get("sha256"):
            raise ValueError("Global Lock formal-config snapshot is internally inconsistent")
        if formal.get("sha256") != current_config_sha:
            raise ValueError(
                "Formal experiment config changed after Global Configuration Lock; "
                "rebuild affected selection artifacts and the lock before test"
            )

        manifest = to_absolute_path(lock["data_provenance"]["manifest"])
        expected_manifest_sha = lock["data_provenance"].get("manifest_sha256")
        if expected_manifest_sha and file_sha256(manifest) != expected_manifest_sha:
            raise ValueError(
                "Synthetic manifest changed after Global Configuration Lock; rebuild the lock"
            )

        protocol_path = to_absolute_path(
            lock.get("data_provenance", {}).get("dataset_protocol", "")
        )
        expected_protocol_sha = lock.get("data_provenance", {}).get("dataset_protocol_sha256")
        if not protocol_path.is_file() or not expected_protocol_sha:
            raise ValueError("Stage-10 Global Lock is missing dataset-protocol provenance")
        if file_sha256(protocol_path) != expected_protocol_sha:
            raise ValueError("Synthetic dataset protocol changed after Global Lock")

        for method, info in lock.get("selection_artifacts", {}).items():
            artifact_path = to_absolute_path(info["path"])
            if not artifact_path.is_file():
                raise FileNotFoundError(f"Frozen selection artifact missing: {artifact_path}")
            if file_sha256(artifact_path) != info.get("sha256"):
                raise ValueError(f"Frozen selection artifact changed: {artifact_path}")

        artifact_utils.verify_source_provenance(
            lock.get("source_provenance", {}),
            artifact_utils.FORMAL_SOURCE_DEPENDENCIES,
        )

        for candidate in methods["unet_small"].get("checkpoints", []):
            checkpoint = to_absolute_path(candidate["checkpoint"])
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Frozen U-Net checkpoint missing: {checkpoint}")
            expected_sha = candidate.get("sha256")
            if expected_sha and file_sha256(checkpoint) != expected_sha:
                raise ValueError(f"Frozen U-Net checkpoint changed: {checkpoint}")
    return lock


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
        save_float_tiff(output_dir / f"{name}.tif", image)
        save_display_png(output_dir / f"{name}.png", image)


def save_float_tiff(path, image):
    """保存tiff，合成数据用"""
    path = Path(path)
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("TIFF image must be a finite 2-D array")
    path.parent.mkdir(parents=True, exist_ok=True)
    tif.imwrite(path, image, dtype=np.float32)


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
