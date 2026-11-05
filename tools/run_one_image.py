"""Run the full method chain on ONE image, live, on a CPU laptop.

This is the "judge points at an image" entry point.  Give it anything:

    python -m tools.run_one_image --image /path/to/photo.png
    python -m tools.run_one_image --image photo.jpg --max-side 384
    python -m tools.run_one_image --sample clean_single_003_single_radial_high

Accepted image formats: png, jpg/jpeg, tif/tiff, nef, cr2, dng, arw.

Behaviour
---------
* The long side is automatically capped (default 512 px) so the PINN stages stay
  fast on a CPU.  No GPU is used or required.
* Synthetic samples are located through the dataset manifest.  If the sample's
  TIFFs are not on disk yet, they are extracted automatically from the local
  server-transfer archive, so any of the 158 synthetic samples can be picked.
* If ground truth exists (synthetic sample) a real metrics table is printed.
  For an arbitrary photograph there is no truth, so only descriptive residual
  statistics are reported - nothing is invented.
* A per-method failure is reported but does not abort the run, so a live demo
  never dies on stage.

Outputs land in ``experiments/outputs/code_check_demo/``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from experiments.common.baselines import fft_gaussian as fft
from experiments.common.baselines import fft_screened_poisson as fourier_ls
from experiments.common.utils import experiment_utils as utils
from tools.code_check_demo import run_methods, stage1_config, stage3_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = utils.BASE_OUTPUT_ROOT / "code_check_demo"
ARCHIVES = PROJECT_ROOT / "artifacts/transfers/stage11_transfer_archives"
INVENTORY = ARCHIVES / "archive_inventory.csv"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".nef", ".cr2", ".dng", ".arw"}


# --------------------------------------------------------------------------
# synthetic sample support (with on-demand extraction)
# --------------------------------------------------------------------------
def _archive_for(path: str) -> str | None:
    if not INVENTORY.is_file():
        return None
    prefix = path if not path.endswith("/") else path
    with INVENTORY.open("r", encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            fields = line.rstrip("\n").split(",")
            if len(fields) >= 8 and fields[4].startswith(prefix):
                return fields[7].strip()
    return None


def _ensure_sample_arrays(row: dict) -> None:
    """Extract this sample's TIFFs from the local archive if they are missing."""
    missing = [k for k in ("observed", "background_true", "clean_true")
               if not utils.to_absolute_path(row[k]).is_file()]
    if not missing:
        return
    sample_dir = str(Path(row["observed"]).parent) + "/"
    archive = _archive_for(sample_dir)
    if archive is None:
        raise SystemExit(
            f"sample arrays missing and no archive index found.\n"
            f"  missing : {missing}\n"
            f"  archive : {ARCHIVES}"
        )
    archive_path = ARCHIVES / archive
    if not archive_path.is_file():
        raise SystemExit(
            f"archive {archive!r} is not on this machine.\n"
            f"Run bash tools/fetch_smoke_data.sh for the default sample, "
            f"or pick a sample that is already extracted."
        )
    print(f"extracting {Path(row['observed']).parent.name} from {archive} ...", flush=True)
    subprocess.run(
        ["unzip", "-o", "-q", str(archive_path), sample_dir + "*", "-d", str(PROJECT_ROOT)],
        check=True,
    )


def load_synthetic(name: str, max_side: int) -> dict:
    manifest = utils.SYNTHETIC_DATA_ROOT / "manifest.csv"
    if not manifest.is_file():
        raise SystemExit(
            "Stage-A manifest not found. Run: bash tools/fetch_smoke_data.sh"
        )
    row = None
    for split in ("validation", "train", "test"):
        rows = utils.load_manifest(manifest, split=split)
        row = next((r for r in rows if r["sample_id"] == name), None)
        if row is not None:
            break
    if row is None:
        raise SystemExit(
            f"{name!r} is not in {manifest}.\n"
            f"Pick one of the sample_id values from that CSV."
        )
    _ensure_sample_arrays(row)

    observed = utils.load_float_tiff(row["observed"])
    stride = max(1, int(np.ceil(max(observed.shape) / max_side)))
    load = lambda key: np.ascontiguousarray(
        utils.load_float_tiff(row[key])[::stride, ::stride], dtype=np.float32
    )
    return {
        "sample_id": name,
        "source": "synthetic",
        "observed": np.ascontiguousarray(observed[::stride, ::stride], dtype=np.float32),
        "background_true": load("background_true"),
        "clean_true": load("clean_true"),
        "metadata": utils.read_json(utils.to_absolute_path(row["metadata"])),
        "downsample": stride,
        "has_truth": True,
    }


# --------------------------------------------------------------------------
# arbitrary image support
# --------------------------------------------------------------------------
def load_image(path: Path, max_side: int) -> dict:
    from pinn_starlight_core.data.image_loader import ImageLoader

    if not path.is_file():
        raise SystemExit(f"image not found: {path}")
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise SystemExit(
            f"unsupported extension {path.suffix!r}; "
            f"expected one of {sorted(IMAGE_SUFFIXES)}"
        )

    # Probe dimensions first so we can cap the long side before training.
    probe = ImageLoader(path, device="cpu", downsample=1)
    stride = max(1, int(np.ceil(max(probe.H, probe.W) / max_side)))
    loader = ImageLoader(path, device="cpu", downsample=stride) if stride > 1 else probe
    coords, gray, width, height = loader.get_gray_data()
    observed = np.asarray(gray, dtype=np.float32).reshape(height, width)
    return {
        "sample_id": path.stem,
        "source": str(path),
        "observed": observed,
        "downsample": stride,
        "has_truth": False,
    }


# --------------------------------------------------------------------------
# archived paper reference (Stage-A formal benchmark)
# --------------------------------------------------------------------------
ARCHIVED_SOURCES = (
    utils.BASE_OUTPUT_ROOT / "e2_benchmark/metrics.csv",
    utils.BASE_OUTPUT_ROOT / "final_pinn_validation/metrics.csv",
)
ARCHIVED_METHOD_ALIASES = {
    "fft_gaussian": ("fft_gaussian",),
    "fourier_ls": ("fourier_ls",),
    "pinn": ("pinn",),
    "final_pinn": ("pinn_decoupled_gradient",),
}


def load_archived_reference(sample_id: str) -> dict:
    """Return {method: row} from the frozen Stage-A GPU benchmarks, if available.

    These CSVs are the paper's full-budget runs.  Printing them next to the live
    CPU numbers makes the budget gap explicit instead of looking like a bug.
    """
    import csv

    wanted = {
        archived: key
        for key, names in ARCHIVED_METHOD_ALIASES.items()
        for archived in names
    }
    reference: dict[str, dict] = {}
    for path in ARCHIVED_SOURCES:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("sample_id") != sample_id:
                    continue
                key = wanted.get(row.get("method"))
                if key and key not in reference:
                    reference[key] = row
    return reference


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def evaluate_synthetic(sample: dict, backgrounds: dict) -> dict:
    from experiments.common.utils import metrics

    full = {
        "observed": sample["observed"],
        "background_true": sample["background_true"],
        "clean_true": sample["clean_true"],
        "metadata": sample["metadata"],
    }
    table = {}
    for name, bg in backgrounds.items():
        prediction = {
            "background_pred": bg,
            "residual_pred": utils.reconstruct_residual(sample["observed"], bg),
        }
        try:
            result = metrics.evaluate_synthetic(full, prediction)
        except Exception as exc:  # noqa: BLE001 - never kill a live demo
            table[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        table[name] = {
            k: result.get(k)
            for k in ("bg_mae", "bg_rmse", "bg_psnr", "bg_ssim",
                      "residual_ssim", "star_f1")
        }
    return table


def evaluate_real(sample: dict, backgrounds: dict) -> dict:
    from experiments.common.utils import metrics

    table = {}
    for name, bg in backgrounds.items():
        residual = utils.reconstruct_residual(sample["observed"], bg)
        try:
            table[name] = metrics.evaluate_real(sample["observed"], bg, residual)
        except Exception as exc:  # noqa: BLE001
            table[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return table


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def draw_figure(path: Path, sample: dict, backgrounds: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observed = sample["observed"]
    has_truth = sample.get("has_truth", False)
    labels = [
        ("Observed", observed, observed if not has_truth else None),
    ]
    if has_truth:
        labels.append(("Truth", sample["background_true"], sample["clean_true"]))
    labels += [
        ("FFT-Gaussian", backgrounds.get("fft_gaussian"), None),
        ("Cosine-LS", backgrounds.get("fourier_ls"), None),
        ("Original PINN", backgrounds.get("pinn"), None),
        ("SGC-PINN", backgrounds.get("final_pinn"), None),
    ]

    fig, axes = plt.subplots(2, len(labels), figsize=(2.9 * len(labels), 6.2))
    if len(labels) == 1:
        axes = np.array(axes).reshape(2, 1)

    reference = sample["background_true"] if has_truth else observed
    bg_lo, bg_hi = np.percentile(reference, [1, 99])
    hi_lo, hi_hi = np.percentile(observed, [1, 99])

    for col, (title, bg, given) in enumerate(labels):
        top, bottom = axes[0, col], axes[1, col]
        if bg is None:
            for axis in (top, bottom):
                axis.axis("off")
            top.text(0.5, 0.5, "failed", ha="center", va="center", fontsize=9, color="crimson")
            top.set_title(title, fontsize=9)
            continue
        top.imshow(np.asarray(bg), cmap="gray", vmin=bg_lo, vmax=bg_hi)
        top.set_title(f"{title}\nbackground", fontsize=9)
        if given is None:
            given = utils.reconstruct_residual(observed, bg)
            subtitle = "residual"
        else:
            subtitle = "clean / observed"
        bottom.imshow(np.asarray(given), cmap="gray", vmin=hi_lo, vmax=hi_hi)
        bottom.set_title(subtitle, fontsize=9)

    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        f"{sample['sample_id']}  (x{sample['downsample']})  -  CPU live run",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="any image file to analyse")
    parser.add_argument("--sample", help="synthetic validation sample_id")
    parser.add_argument("--max-side", type=int, default=384,
                        help="cap the long image side (default 512)")
    parser.add_argument("--steps", type=int, default=500,
                        help="PINN steps per stage (CPU demo default 500)")
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)

    if bool(args.image) == bool(args.sample):
        parser.error("give exactly one of --image or --sample")

    import torch

    device = torch.device("cpu")
    print("=" * 78)
    print("PINN-Starlight - one-image CPU demo")
    print("=" * 78)
    print(f"device : {device}   (CUDA available: {torch.cuda.is_available()})")
    print(f"cap    : long side <= {args.max_side} px")
    print(f"steps  : {args.steps} per PINN stage")
    print("=" * 78)

    started = time.perf_counter()
    if args.sample:
        sample = load_synthetic(args.sample, args.max_side)
    else:
        sample = load_image(args.image, args.max_side)
    print(f"loaded {sample['observed'].shape} (x{sample['downsample']}) "
          f"in {time.perf_counter() - started:.1f}s", flush=True)

    started = time.perf_counter()
    try:
        run = run_methods(sample, args.steps, device)
        backgrounds = run["backgrounds"]
        timings = run["timings"]
    except Exception as exc:  # noqa: BLE001 - degrade instead of dying
        print(f"\nFAILED to run the method chain: {type(exc).__name__}: {exc}")
        print("This usually means torch or the scientific stack is missing.")
        return 1
    print(f"methods finished in {time.perf_counter() - started:.1f}s\n")

    table = (
        evaluate_synthetic(sample, backgrounds)
        if sample.get("has_truth")
        else evaluate_real(sample, backgrounds)
    )

    labels = {
        "fft_gaussian": "FFT-Gaussian",
        "fourier_ls": "Cosine-LS",
        "pinn": "Original PINN",
        "final_pinn": "SGC-PINN",
    }
    reference = (
        load_archived_reference(sample["sample_id"]) if sample.get("has_truth") else {}
    )

    def cell(row, name, spec, width):
        value = row.get(name)
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "n/a".rjust(width)
        try:
            return format(value, spec).rjust(width)
        except (TypeError, ValueError):
            return "n/a".rjust(width)

    if sample.get("has_truth"):
        header = (
            f"{'method':<16}{'bg MAE':>11}{'bg PSNR':>9}{'bg SSIM':>9}"
            f"{'res SSIM':>9}{'star F1*':>9}{'time':>9}"
            f"{'| paper bgMAE':>15}{'paper F1':>10}"
        )
        print(header)
        print("-" * len(header))
        for key, label in labels.items():
            row = table.get(key, {})
            if "error" in row:
                print(f"{label:<16}  ERROR: {row['error'][:56]}")
                continue
            archived = reference.get(key, {})
            try:
                paper_mae = f"{float(archived['bg_mae']):.6f}"
            except (KeyError, TypeError, ValueError):
                paper_mae = "n/a"
            try:
                paper_f1 = f"{float(archived['star_f1']):.3f}"
            except (KeyError, TypeError, ValueError):
                paper_f1 = "n/a"
            print(
                f"{label:<16}{cell(row,'bg_mae','.6f',11)}"
                f"{cell(row,'bg_psnr','.2f',9)}{cell(row,'bg_ssim','.4f',9)}"
                f"{cell(row,'residual_ssim','.4f',9)}{cell(row,'star_f1','.3f',9)}"
                f"{timings.get(key, float('nan')):>8.2f}s"
                f"{paper_mae:>15}{paper_f1:>10}"
            )
        print("-" * len(header))
        if sample["downsample"] > 4:
            print(
                f"* star F1 is not measurable at x{sample['downsample']}: the stellar PSF"
                " falls below one pixel."
            )
            print("  Use --max-side 2048 (much slower) if live photometry is required.")
        if reference:
            print("'paper' columns = archived Stage-A GPU benchmark at the FULL paper budget.")
            print("Deterministic methods reproduce it; learned methods differ because")
            print("this CPU run uses a tiny fraction of the training budget.")
    else:
        header = (
            f"{'method':<16}{'bg mean':>10}{'bg TV':>10}{'res std':>10}"
            f"{'res<0':>9}{'res>1':>9}{'stars':>8}{'time':>9}"
        )
        print(header)
        print("-" * len(header))
        for key, label in labels.items():
            row = table.get(key, {})
            if "error" in row:
                print(f"{label:<16}  ERROR: {row['error'][:56]}")
                continue
            print(
                f"{label:<16}{cell(row,'background_mean','.5f',10)}"
                f"{cell(row,'background_total_variation','.5f',10)}"
                f"{cell(row,'residual_std','.5f',10)}"
                f"{cell(row,'residual_negative_fraction','.4f',9)}"
                f"{cell(row,'residual_above_one_fraction','.4f',9)}"
                f"{cell(row,'detected_star_count','d',8)}"
                f"{timings.get(key, float('nan')):>8.2f}s"
            )
        print("-" * len(header))
        print("This image has no ground truth, so error metrics cannot be computed.")
        print("Columns are descriptive only: background smoothness (TV), residual spread,")
        print("how much of the residual falls outside [0,1], and detected star count.")

    outdir = args.out / sample["sample_id"]
    outdir.mkdir(parents=True, exist_ok=True)
    figure = outdir / "comparison.png"
    draw_figure(figure, sample, backgrounds)
    summary = {
        "sample_id": sample["sample_id"],
        "source": sample.get("source"),
        "has_truth": bool(sample.get("has_truth")),
        "downsample": sample["downsample"],
        "steps": args.steps,
        "device": "cpu",
        "timings_s": timings,
        "metrics": table,
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"\nfigure  : {figure}")
    print(f"summary : {outdir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
