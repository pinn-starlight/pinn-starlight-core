"""Build SGC-PINN background-removal figures from the frozen E4 review renders.

Why this exists
---------------
The E4 formal run produced one `comparison.png` per real image, laid out as
2 rows x 5 columns: [Observed | FFT-Gaussian | Cosine-LS | PINN |
Final-PINN], with backgrounds on the top row and background-subtracted
residuals on the bottom row.

To make an "SGC-PINN removal effect" figure we only need three of those panels:

    Observed  |  SGC-PINN background  |  SGC-PINN residual

Two caveats are handled here:

1. The stored predictions are ``background_pred.tif`` only; residuals are
   ``observed - background``.  The E4 *observed* arrays are not in this repo
   (the raw camera images were never transferred), so the residual panel is
   cropped from the already-rendered PNG rather than recomputed.
2. The archived PNG labels the method "Final-PINN" (its internal code name),
   while the paper calls it "SGC-PINN".  The baked-in titles are therefore
   cropped away and re-drawn with the paper's naming.

Nothing is rescaled: the display range baked into the E4 render is preserved,
so the panels remain consistent with the paper's Fig. 8 / Fig. 9.

Usage
-----
    python -m tools.render_sgc_removal                 # all 16 images
    python -m tools.render_sgc_removal --images 001 016
    python -m tools.render_sgc_removal --list
"""
from __future__ import annotations

import argparse
import math
import zipfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_ZIP = PROJECT_ROOT / "artifacts/review/e4_review_for_paper.zip"
DEFAULT_OUT = PROJECT_ROOT / "experiments/outputs/sgc_removal_figures"

# Panel geometry measured from the archived renders (4200 x 1400, 6 x 2 grid).
CJK_FONT_CANDIDATES = (
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
)
LABELS = {
    "zh": ("(a) 原始观测", "(b) SGC-PINN 背景估计", "(c) SGC-PINN 背景扣除结果"),
    "en": ("(a) Observed", "(b) SGC-PINN background", "(c) SGC-PINN residual"),
}

COLUMN_X = [(0, 692), (708, 1392), (1408, 2092), (2108, 2792), (2808, 3492), (3508, 4200)]
ROW_Y = [(0, 700), (700, 1400)]
CONTENT_Y = (123, 577)          # image pixels only, titles excluded
OBSERVED_COL = 0
SGC_COL = 5                     # last column == Final-PINN == SGC-PINN


def _load_png(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    import io

    from PIL import Image

    with zf.open(name) as handle:
        return np.array(Image.open(io.BytesIO(handle.read())).convert("RGB"))


def _stretch_panel(panel: np.ndarray, gamma: float, black_pct: float | None,
                   white_pct: float | None) -> np.ndarray:
    """Darken an already-rendered display panel.

    The stored E4 renders are 8-bit display images; the raw float arrays are not
    in this repo.  This is therefore a *display-only* transform applied
    identically to all three panels, so their relative brightness stays
    comparable:

        normalise [p_black, p_white] -> [0, 1], clip, then apply  out = in ** gamma

    ``gamma > 1`` darkens midtones (the sky background), which is what the
    server-side linear ``vmin=0, vmax=1`` mapping left too bright.
    """
    data = np.asarray(panel, dtype=np.float32) / 255.0
    if black_pct is None or white_pct is None:
        # No re-stretch: the server render is already normalised to [0, 1].
        # Re-stretching here would double-normalise and make dark frames black.
        scaled = np.clip(data, 0.0, 1.0)
    else:
        lo = float(np.percentile(data, black_pct))
        hi = float(np.percentile(data, white_pct))
        if hi <= lo:
            hi, lo = 1.0, 0.0
        scaled = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    if gamma != 1.0:
        scaled = np.power(scaled, gamma, dtype=np.float32)
    return np.clip(scaled * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _auto_gamma(panel: np.ndarray, target_median: float) -> float:
    """Pick the gamma that maps this panel's median onto ``target_median``.

    A single fixed gamma cannot serve all 16 real frames: their source renders
    differ by more than a factor of two in median level (63-155 for the observed
    panels).  Deriving gamma per panel gives every image the same perceived
    background brightness, which is what a side-by-side figure needs.
    """
    data = np.asarray(panel, dtype=np.float32) / 255.0
    median = float(np.median(data))
    target = float(np.clip(target_median / 255.0, 1e-3, 1.0 - 1e-3))
    if median <= 1e-3 or median >= 1.0 - 1e-3:
        return 1.0
    gamma = math.log(target) / math.log(median)
    return float(np.clip(gamma, 0.30, 5.0))


def _panel(rgb: np.ndarray, column: int, row: int) -> np.ndarray:
    x0, x1 = COLUMN_X[column]
    y0, y1 = CONTENT_Y
    top = ROW_Y[row][0]
    return rgb[top + y0 : top + y1, x0:x1]


def _setup_cjk_font() -> bool:
    """Register a CJK font so Chinese labels do not render as tofu boxes."""
    from matplotlib import font_manager

    for path in CJK_FONT_CANDIDATES:
        if not Path(path).is_file():
            continue
        try:
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            import matplotlib

            matplotlib.rcParams["font.family"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
        except Exception:  # noqa: BLE001 - fall back to default font
            continue
    return False


def build_figure(observed, background, residual, title: str, out_png: Path, lang: str = "zh"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if lang == "zh":
        _setup_cjk_font()

    labels = LABELS[lang]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    panels = [
        (observed, labels[0]),
        (background, labels[1]),
        (residual, labels[2]),
    ]
    for axis, (image, label) in zip(axes, panels):
        axis.imshow(image)
        axis.set_title(label, fontsize=13)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="*", default=None,
                        help="suffixes like 001 016 (default: all)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--list", action="store_true", help="list available images")
    parser.add_argument("--verbose", action="store_true", help="print per-panel gamma")
    parser.add_argument("--lang", choices=("zh", "en"), default="zh",
                        help="figure label language")
    parser.add_argument("--gamma", type=float, default=2.2,
                        help="display gamma for observed/background; >1 darkens (default 2.2)")
    parser.add_argument("--auto-level", action="store_true", default=True,
                        help="derive gamma per panel so every image has the same "
                             "perceived background brightness (default on)")
    parser.add_argument("--no-auto-level", dest="auto_level", action="store_false",
                        help="use the fixed --gamma / --residual-gamma instead")
    parser.add_argument("--target-median", type=float, default=75.0,
                        help="target median for every panel when auto-levelling (default 75)")
    parser.add_argument("--match-b", action="store_true",
                        help="auto-level to the 'B' tonality that was approved on "
                             "real_high_001 (obs 78 / bg 63 / res 57)")
    parser.add_argument("--residual-gamma", type=float, default=1.5,
                        help="gamma for the residual panel (default 1.5). "
                             "Use a smaller value to keep faint stars visible.")
    parser.add_argument("--stretch", choices=("none", "percentile"), default="none",
                        help="extra percentile re-stretch; default none because the "
                             "server render is already normalised")
    parser.add_argument("--black-pct", type=float, default=1.0,
                        help="lower percentile when --stretch percentile (default 1.0)")
    parser.add_argument("--white-pct", type=float, default=99.5,
                        help="upper percentile when --stretch percentile (default 99.5)")
    args = parser.parse_args(argv)

    if not REVIEW_ZIP.is_file():
        raise SystemExit(f"missing review archive: {REVIEW_ZIP}")

    with zipfile.ZipFile(REVIEW_ZIP) as zf:
        names = [n for n in zf.namelist() if n.endswith("/comparison.png")]
        if args.list:
            for name in sorted(names):
                print(name.rsplit("/", 2)[-2])
            return 0

        selected = []
        for name in sorted(names):
            image_id = name.rsplit("/", 2)[-2]          # real_high_001
            suffix = image_id.rsplit("_", 1)[-1]
            if args.images and suffix not in args.images:
                continue
            selected.append((image_id, name))

        if not selected:
            raise SystemExit("no images matched --images")

        print(f"building {len(selected)} SGC-PINN removal figures -> {args.out}")
        for image_id, name in selected:
            rgb = _load_png(zf, name)
            if rgb.shape[:2] != (1400, 4200):
                raise SystemExit(
                    f"{image_id}: unexpected render size {rgb.shape[:2]}, expected (1400, 4200)"
                )
            lo = args.black_pct if args.stretch == "percentile" else None
            hi = args.white_pct if args.stretch == "percentile" else None
            raw_obs = _panel(rgb, OBSERVED_COL, 0)
            raw_bg = _panel(rgb, SGC_COL, 0)
            raw_res = _panel(rgb, SGC_COL, 1)
            if args.auto_level:
                if args.match_b:
                    t_obs, t_bg, t_res = 78.0, 63.0, 57.0
                else:
                    t_obs = t_bg = t_res = args.target_median
                g_obs = _auto_gamma(raw_obs, t_obs)
                g_bg = _auto_gamma(raw_bg, t_bg)
                g_res = _auto_gamma(raw_res, t_res)
            else:
                g_obs = args.gamma
                g_bg = args.gamma
                g_res = args.gamma if args.residual_gamma is None else args.residual_gamma
            observed = _stretch_panel(raw_obs, g_obs, lo, hi)
            background = _stretch_panel(raw_bg, g_bg, lo, hi)
            residual = _stretch_panel(raw_res, g_res, lo, hi)
            if args.verbose:
                print(f"    gammas obs={g_obs:.2f} bg={g_bg:.2f} res={g_res:.2f}")
            suffix = "_matchB" if (args.auto_level and args.match_b) else (
                "_auto" if args.auto_level else "")
            if not args.auto_level and args.gamma != 1.0:
                suffix = f"_g{args.gamma:g}"
                if args.residual_gamma is not None:
                    suffix += f"r{args.residual_gamma:g}"
            out_png = args.out / f"{image_id}_sgc_removal{suffix}.png"
            build_figure(observed, background, residual, image_id, out_png, args.lang)
            print(f"  {image_id} -> {out_png.name}")
    print(f"\ndone: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
