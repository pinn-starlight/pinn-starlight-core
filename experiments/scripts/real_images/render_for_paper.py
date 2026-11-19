"""Render E4 scientific TIFF outputs in each source photograph's own display style.

This command is deliberately post-processing only:
- it never reruns Gaussian / Cosine-LS / PINN / Final-PINN;
- it never overwrites formal E4 TIFFs or metrics;
- one display mapping is fitted independently per source image;
- all methods for that source image share exactly the same mapping;
- the top-row "background estimate" is the true ``background_pred.tif`` rendered
  in the source photograph's own NEF/camera-preview display style.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tif

from experiments.common.utils import e4_visualization as vis
from experiments.common.utils import experiment_utils as utils

METHODS = (
    "fft_gaussian",
    "fourier_ls",
    "pinn",
    "pinn_decoupled_gradient",
)
METHOD_TITLES = {
    "fft_gaussian": "Gaussian",
    "fourier_ls": "Cosine-LS",
    "pinn": "PINN",
    "pinn_decoupled_gradient": "Final-PINN",
}

DEFAULT_E4_ROOT = utils.OUTPUT_ROOT / "e4_real"
DEFAULT_REAL_ROOT = utils.REAL_HIGH_POLLUTION_ROOT


def main(
    *,
    e4_root: Path,
    real_root: Path,
    image_name: str | None,
    force: bool,
) -> None:
    e4_root = Path(e4_root)
    real_root = Path(real_root)

    if not e4_root.is_dir():
        raise FileNotFoundError(
            f"E4 output directory does not exist: {e4_root}. Run formal E4 first."
        )

    sources = _source_map(real_root)
    result_dirs = sorted(
        path
        for path in e4_root.iterdir()
        if path.is_dir() and all((path / method).is_dir() for method in METHODS)
    )

    if image_name is not None:
        wanted = Path(image_name).stem
        result_dirs = [path for path in result_dirs if path.name == wanted]
        if not result_dirs:
            raise FileNotFoundError(
                f"E4 result for image {image_name!r} was not found in {e4_root}"
            )

    if not result_dirs:
        raise RuntimeError(f"No complete E4 image result directories found in {e4_root}")

    for index, image_dir in enumerate(result_dirs, start=1):
        stem = image_dir.name
        source = sources.get(stem)
        if source is None:
            raise FileNotFoundError(
                f"No source photograph with stem {stem!r} exists in {real_root}"
            )
        print(
            f"E4 paper render [{index}/{len(result_dirs)}] {source.name}",
            flush=True,
        )
        _render_one(image_dir, source, force=force)

    print(f"E4 paper rendering complete: {len(result_dirs)} image(s) -> {e4_root}")


def _render_one(image_dir: Path, source: Path, *, force: bool) -> None:
    scientific: dict[str, np.ndarray] = {}
    observed: np.ndarray | None = None

    for method in METHODS:
        method_dir = image_dir / method
        observed_path = method_dir / "observed.tif"
        background_path = method_dir / "background_pred.tif"

        if not observed_path.is_file() or not background_path.is_file():
            raise FileNotFoundError(
                f"Missing formal E4 TIFFs for {image_dir.name}/{method}. "
                "This renderer never regenerates scientific outputs."
            )

        method_observed = _read_gray_tiff(observed_path)
        background = _read_gray_tiff(background_path)

        if observed is None:
            observed = method_observed
        elif not np.array_equal(observed, method_observed):
            raise ValueError(
                f"Observed TIFFs differ across methods for {image_dir.name}; "
                "paper rendering is aborted to preserve a fair comparison."
            )

        if background.shape != method_observed.shape:
            raise ValueError(
                f"Shape mismatch for {image_dir.name}/{method}: "
                f"observed={method_observed.shape}, background={background.shape}"
            )
        scientific[method] = background

    assert observed is not None

    preview_rgb, source_meta = vis.load_source_display_rgb(source, observed.shape)
    tone_mapping = vis.fit_per_image_tone_mapping(observed, preview_rgb)

    original_path = image_dir / "original_display.png"
    mapping_path = image_dir / "display_mapping.json"
    _ensure_writable((original_path, mapping_path), force)
    vis.save_rgb_png(original_path, preview_rgb)
    vis.write_mapping_json(mapping_path, source_meta, tone_mapping)

    rendered: dict[str, dict[str, np.ndarray]] = {}

    for method in METHODS:
        method_dir = image_dir / method
        result = vis.render_source_style_result(
            observed,
            scientific[method],
            preview_rgb,
            tone_mapping,
        )
        rendered[method] = result

        background_display_path = method_dir / "background_estimate_display.png"
        corrected_path = method_dir / "corrected_display.png"
        triptych_path = method_dir / "display_triptych.png"
        _ensure_writable(
            (background_display_path, corrected_path, triptych_path),
            force,
        )

        vis.save_rgb_png(
            background_display_path,
            result["background_estimate_rgb"],
        )
        vis.save_rgb_png(
            corrected_path,
            result["corrected_rgb"],
        )
        _save_triptych(
            triptych_path,
            preview_rgb,
            result["background_estimate_rgb"],
            result["corrected_rgb"],
            METHOD_TITLES[method],
        )

    main_path = image_dir / "paper_main_comparison.png"
    full_path = image_dir / "paper_full_comparison.png"
    notes_path = image_dir / "paper_figure_notes.md"
    _ensure_writable((main_path, full_path, notes_path), force)

    _save_main_comparison(main_path, preview_rgb, rendered)
    _save_full_comparison(full_path, preview_rgb, rendered)
    _save_notes(notes_path, source.name)


def _source_map(real_root: Path) -> dict[str, Path]:
    images = utils.supported_image_files(real_root)
    mapping: dict[str, Path] = {}

    for path in images:
        if path.stem in mapping:
            raise ValueError(
                f"Duplicate source stem {path.stem!r}: {mapping[path.stem]} and {path}"
            )
        mapping[path.stem] = path

    if not mapping:
        raise FileNotFoundError(f"No supported real images found in {real_root}")

    return mapping


def _read_gray_tiff(path: Path) -> np.ndarray:
    array = np.asarray(tif.imread(path), dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"Expected a finite 2-D TIFF: {path}")
    return array


def _ensure_writable(paths, force: bool) -> None:
    existing = [Path(path) for path in paths if Path(path).exists()]
    if existing and not force:
        raise FileExistsError(
            "Display output already exists; use --force to replace "
            "display-only files: " + ", ".join(str(path) for path in existing[:3])
        )


def _save_triptych(
    path: Path,
    original: np.ndarray,
    background_estimate: np.ndarray,
    corrected: np.ndarray,
    method_title: str,
) -> None:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
        constrained_layout=True,
    )

    for axis, image, title in zip(
        axes,
        (original, background_estimate, corrected),
        ("Original", "Background estimate", "Corrected"),
    ):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")

    figure.suptitle(method_title)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _save_main_comparison(
    path: Path,
    original: np.ndarray,
    rendered: dict[str, dict[str, np.ndarray]],
) -> None:
    column_count = 1 + len(METHODS)
    figure, axes = plt.subplots(
        1,
        column_count,
        figsize=(3.5 * column_count, 3.7),
        constrained_layout=True,
    )

    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for axis, method in zip(axes[1:], METHODS):
        axis.imshow(rendered[method]["corrected_rgb"])
        axis.set_title(METHOD_TITLES[method])
        axis.axis("off")

    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _save_full_comparison(
    path: Path,
    original: np.ndarray,
    rendered: dict[str, dict[str, np.ndarray]],
) -> None:
    column_count = 1 + len(METHODS)
    figure, axes = plt.subplots(
        2,
        column_count,
        figsize=(3.5 * column_count, 7.2),
        constrained_layout=True,
    )

    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original")
    axes[1, 0].imshow(original)
    axes[1, 0].set_title("Original")

    for axis in (axes[0, 0], axes[1, 0]):
        axis.axis("off")

    for column, method in enumerate(METHODS, start=1):
        axes[0, column].imshow(rendered[method]["background_estimate_rgb"])
        axes[0, column].set_title(f"{METHOD_TITLES[method]} background estimate")
        axes[1, column].imshow(rendered[method]["corrected_rgb"])
        axes[1, column].set_title(f"{METHOD_TITLES[method]} corrected")
        axes[0, column].axis("off")
        axes[1, column].axis("off")

    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _save_notes(path: Path, source_name: str) -> None:
    content = f"""# E4 论文展示建议：{source_name}

## 正文优先

使用 `paper_main_comparison.png`：原始照片 + Gaussian / Cosine-LS / PINN / Final-PINN 的处理后照片。它最直接回答“不同方法处理后看起来怎样”。

建议图注：

> 同一真实高光污染星空图像的背景抑制结果。为便于视觉比较，每张源照片都根据其自身 NEF/相机预览独立建立显示映射；同一张照片的五种方法共享完全相同的显示映射。显示映射只用于论文可视化，科学存档均使用未改变的线性 TIFF 数组。

## 需要同时展示“背景估计”和“校正结果”时

使用 `paper_full_comparison.png`，或各方法目录下的 `display_triptych.png`。其三联图依次为：

1. 原始照片；
2. 该方法的 **background estimate**（即 `background_pred.tif`），但显示时映射到这张源照片自己的 NEF 风格；
3. 该方法的 **corrected result**（即 `observed - background_pred`），同样显示为这张源照片自己的 NEF 风格。

## 命名原则

- 上排应写 **background estimate**，因为它对应模型/方法输出的 `background_pred.tif`；
- 不建议在主图里把上排写成 `removed`，因为那容易被理解成“原图与结果图的差分可视层”，而不是方法的背景估计本身；
- 如果确实想说明“视觉上移除了什么”，可以另做补充解释图，但不应替代主图中的背景估计。

## 公平性规则

- 每张源 NEF **单独**拟合自己的显示映射，不让不同 NEF 共用一条全局拉伸曲线；
- 对同一张 NEF，Gaussian / Cosine-LS / PINN / Final-PINN **必须共享同一条映射**，不能按方法分别调亮；
- `background_pred.tif`、`residual_pred.tif`、`metrics.csv` 等正式科学结果不被本脚本修改；
- `background_estimate_display.png` 是背景估计的 NEF-style 可视化，不是新的数值背景；
- `corrected_display.png` 是 `observed - background_pred` 的 NEF-style 可视化，仅用于论文展示。
"""
    Path(path).write_text(content, encoding="utf-8")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--e4-root",
        type=Path,
        default=DEFAULT_E4_ROOT,
    )
    parser.add_argument(
        "--real-image-root",
        type=Path,
        default=DEFAULT_REAL_ROOT,
    )
    parser.add_argument(
        "--image",
        dest="image_name",
        default=None,
        help="render only one source stem (for example real_high_001)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace display-only PNG/JSON/MD files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        e4_root=args.e4_root,
        real_root=args.real_image_root,
        image_name=args.image_name,
        force=args.force,
    )
