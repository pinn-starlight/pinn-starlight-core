"""Display-only helpers for E4 paper figures.

This module deliberately stays outside the formal E4 numerical path. It reads
already-saved scientific TIFF outputs and projects them back into the visual
style of each source photograph. Every source image gets its own display
mapping; all methods for that same image share the mapping.

Important distinction:
- "background estimate" means the method output ``background_pred.tif``;
- "corrected" means the observed image after subtracting that estimate;
- both are rendered only for paper display and never replace the scientific
  TIFFs or metrics.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

_LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
_RAW_EXTS = {".nef", ".cr2", ".cr3", ".arw", ".dng"}
_RASTER_EXTS = {".png", ".jpg", ".jpeg"}
_TIFF_EXTS = {".tif", ".tiff"}


def load_source_display_rgb(
    path: Path,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a display RGB derived from *this* source image and resize to target.

    RAW files prefer their embedded camera preview. If that is unavailable,
    rawpy is used for a display-only postprocess with the image's own camera
    white balance. This never replaces the formal linear grayscale input.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"source image does not exist: {path}")

    suffix = path.suffix.lower()
    backend = ""

    if suffix in _RAW_EXTS:
        try:
            import rawpy
        except ImportError as exc:
            raise RuntimeError(
                "RAW/NEF display rendering requires rawpy. "
                "Run inside the project Pixi environment."
            ) from exc

        with rawpy.imread(str(path)) as raw:
            try:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    with Image.open(io.BytesIO(thumb.data)) as image:
                        image = ImageOps.exif_transpose(image).convert("RGB")
                        rgb = np.asarray(image, dtype=np.float32) / 255.0
                    backend = "embedded_camera_preview_jpeg"
                else:
                    rgb = _normalize_rgb_array(np.asarray(thumb.data))
                    backend = "embedded_camera_preview_bitmap"
            except Exception:
                data = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=False,
                    output_bps=8,
                    gamma=(2.222, 4.5),
                )
                rgb = np.asarray(data, dtype=np.float32) / 255.0
                backend = "rawpy_display_postprocess_fallback"

    elif suffix in _RASTER_EXTS:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            rgb = np.asarray(image, dtype=np.float32) / 255.0
        backend = "raster_rgb"

    elif suffix in _TIFF_EXTS:
        try:
            import tifffile as tif
        except ImportError as exc:
            raise RuntimeError("TIFF display rendering requires tifffile") from exc
        rgb = _normalize_rgb_array(np.asarray(tif.imread(path)))
        backend = "tiff_rgb"

    else:
        raise ValueError(f"unsupported source display format: {suffix}")

    rgb = _resize_rgb(np.clip(rgb, 0.0, 1.0), target_shape)
    return rgb.astype(np.float32, copy=False), {
        "source_file": str(path),
        "source_suffix": suffix,
        "preview_backend": backend,
        "target_height": int(target_shape[0]),
        "target_width": int(target_shape[1]),
    }


def fit_per_image_tone_mapping(
    observed: np.ndarray,
    preview_rgb: np.ndarray,
    *,
    quantile_count: int = 129,
) -> dict[str, Any]:
    """Fit one monotone luminance mapping for a single source photograph.

    Quantile matching is intentionally per-image. It captures the broad tone
    response of that source preview without using a global stretch shared by
    unrelated NEF files. The same fitted curve must be shared by all methods
    for the source image to keep method comparisons fair.
    """
    observed = _as_gray(observed)
    preview_rgb = _as_rgb(preview_rgb, observed.shape)
    preview_luma = np.tensordot(preview_rgb, _LUMA, axes=([-1], [0])).astype(np.float32)

    obs = np.clip(observed, 0.0, 1.0).reshape(-1)
    lum = np.clip(preview_luma, 0.0, 1.0).reshape(-1)
    finite = np.isfinite(obs) & np.isfinite(lum)
    obs = obs[finite]
    lum = lum[finite]
    if obs.size < 32:
        raise ValueError("not enough finite pixels to fit the per-image display mapping")

    max_samples = 1_000_000
    if obs.size > max_samples:
        stride = max(1, obs.size // max_samples)
        obs = obs[::stride][:max_samples]
        lum = lum[::stride][:max_samples]

    quantile_count = max(17, int(quantile_count))
    q = np.linspace(0.0, 1.0, quantile_count, dtype=np.float64)
    x = np.quantile(obs, q).astype(np.float64)
    y = np.quantile(lum, q).astype(np.float64)

    x = np.concatenate(([0.0], x, [1.0]))
    y = np.concatenate(([0.0], y, [1.0]))
    x = np.clip(x, 0.0, 1.0)
    y = np.maximum.accumulate(np.clip(y, 0.0, 1.0))

    unique_x: list[float] = []
    unique_y: list[float] = []
    for x_value, y_value in zip(x.tolist(), y.tolist()):
        if unique_x and abs(x_value - unique_x[-1]) <= 1e-12:
            unique_y[-1] = max(unique_y[-1], float(y_value))
        else:
            unique_x.append(float(x_value))
            unique_y.append(float(y_value))

    if len(unique_x) < 2:
        unique_x = [0.0, 1.0]
        unique_y = [0.0, 1.0]

    unique_y_array = np.maximum.accumulate(np.asarray(unique_y, dtype=np.float64))
    unique_y_array[0] = 0.0
    unique_y_array[-1] = 1.0
    unique_y_array = np.maximum.accumulate(unique_y_array)

    return {
        "schema_version": 1,
        "mapping": "per_image_monotone_quantile_luminance",
        "description": (
            "Display-only mapping fitted independently for each source image from "
            "formal linear observed luminance to that source image's own preview "
            "luminance. All methods for the same image share this mapping; "
            "scientific TIFFs and metrics are unchanged."
        ),
        "quantile_count": quantile_count,
        "x_linear": unique_x,
        "y_display": [float(v) for v in unique_y_array.tolist()],
    }


def render_source_style_result(
    observed: np.ndarray,
    background_pred: np.ndarray,
    preview_rgb: np.ndarray,
    tone_mapping: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Project one method's E4 outputs into the source image's display style.

    Returns both:
    - ``background_estimate_rgb``: the method's ``background_pred.tif`` shown in
      the source image's own display/NEF style;
    - ``corrected_rgb``: the corrected image shown in the same display style.
    """
    observed = _as_gray(observed)
    background = _as_gray(background_pred)
    if observed.shape != background.shape:
        raise ValueError(
            f"shape mismatch: observed={observed.shape}, background={background.shape}"
        )
    preview_rgb = _as_rgb(preview_rgb, observed.shape)

    residual = np.maximum(observed - background, 0.0).astype(np.float32)
    mapped_observed = apply_tone_mapping(observed, tone_mapping)
    mapped_background = apply_tone_mapping(background, tone_mapping)
    mapped_residual = apply_tone_mapping(residual, tone_mapping)

    eps = 1e-6
    preview_luma = np.tensordot(preview_rgb, _LUMA, axes=([-1], [0])).astype(np.float32)
    chroma_basis = np.divide(
        preview_rgb,
        np.maximum(preview_luma[..., None], eps),
        out=np.zeros_like(preview_rgb, dtype=np.float32),
        where=np.isfinite(preview_luma[..., None]),
    )

    # Keep the background estimate's identity as background_pred.tif while
    # borrowing the source preview's color cast/chroma for paper display.
    background_estimate_rgb = np.clip(
        chroma_basis * mapped_background[..., None],
        0.0,
        1.0,
    )

    keep_ratio = np.divide(
        mapped_residual,
        np.maximum(mapped_observed, eps),
        out=np.zeros_like(mapped_residual, dtype=np.float32),
        where=np.isfinite(mapped_observed),
    )
    keep_ratio = np.clip(keep_ratio, 0.0, 1.0)

    corrected_rgb = np.clip(preview_rgb * keep_ratio[..., None], 0.0, 1.0)

    return {
        "background_estimate_rgb": background_estimate_rgb.astype(np.float32, copy=False),
        "corrected_rgb": corrected_rgb.astype(np.float32, copy=False),
        "keep_ratio": keep_ratio.astype(np.float32, copy=False),
    }


def apply_tone_mapping(
    values: np.ndarray,
    tone_mapping: dict[str, Any],
) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    x = np.asarray(tone_mapping["x_linear"], dtype=np.float64)
    y = np.asarray(tone_mapping["y_display"], dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        raise ValueError("invalid tone mapping knots")
    mapped = np.interp(values.astype(np.float64), x, y)
    return np.clip(mapped, 0.0, 1.0).astype(np.float32)


def save_rgb_png(path: Path, rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"RGB PNG expects HxWx3, got {rgb.shape}")
    Image.fromarray(np.round(rgb * 255.0).astype(np.uint8), mode="RGB").save(path)


def write_mapping_json(
    path: Path,
    source_meta: dict[str, Any],
    tone_mapping: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "display_only": True,
        "scientific_outputs_modified": False,
        "source": source_meta,
        "tone_mapping": tone_mapping,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _normalize_rgb_array(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim == 2:
        data = np.stack([data, data, data], axis=-1)
    if data.ndim != 3 or data.shape[-1] not in (3, 4):
        raise ValueError(f"unsupported display image shape: {data.shape}")
    data = data[..., :3]

    if np.issubdtype(data.dtype, np.integer):
        maximum = float(np.iinfo(data.dtype).max)
        if maximum <= 0:
            raise ValueError("invalid integer image range")
        return data.astype(np.float32) / maximum

    array = data.astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError("display image contains NaN or Inf")

    if float(np.min(array)) < 0.0 or float(np.max(array)) > 1.0:
        low, high = np.percentile(array, [0.1, 99.9])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.min(array))
            high = float(np.max(array))
        if high > low:
            array = (array - low) / (high - low)
        else:
            array = np.zeros_like(array, dtype=np.float32)
    return np.clip(array, 0.0, 1.0)


def _resize_rgb(rgb: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    rgb = _normalize_rgb_array(rgb)
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"invalid target shape: {target_shape}")
    if rgb.shape[:2] == (target_h, target_w):
        return rgb.astype(np.float32, copy=False)

    image = Image.fromarray(np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8))
    resampling = getattr(Image, "Resampling", Image)
    image = image.resize((target_w, target_h), resample=resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def _as_gray(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"expected 2-D grayscale array, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("grayscale array contains NaN or Inf")
    return array


def _as_rgb(array: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 RGB array, got {array.shape}")
    if array.shape[:2] != tuple(expected_shape):
        raise ValueError(
            f"RGB shape mismatch: expected {tuple(expected_shape)}, got {array.shape[:2]}"
        )
    if not np.isfinite(array).all():
        raise ValueError("RGB array contains NaN or Inf")
    return np.clip(array, 0.0, 1.0)
