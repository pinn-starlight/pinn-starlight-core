"""Optional PNG preview for ``pinn-starlight estimate``.

matplotlib is imported lazily and every failure is swallowed: a missing plotting
stack must never fail the estimation itself.
"""

from __future__ import annotations

from pathlib import Path

from experiments.cli.context import TOOL_NAME
from experiments.configs.demo_methods import DISPLAY_NAMES


def render_preview(path: Path, observed, backgrounds: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - preview is optional
        return

    from experiments.common.utils import experiment_utils as utils

    columns = [("Observed", observed, None)] + [
        (DISPLAY_NAMES.get(name, name), background, None)
        for name, background in backgrounds.items()
    ]
    figure, axes = plt.subplots(2, len(columns), figsize=(3.0 * len(columns), 6.2))
    axes = axes.reshape(2, len(columns))
    vmin, vmax = float(observed.min()), float(observed.max())
    for col, (title, background, _) in enumerate(columns):
        axes[0, col].imshow(background, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f"{title}\nbackground", fontsize=9)
        residual = observed if col == 0 else utils.reconstruct_residual(observed, background)
        axes[1, col].imshow(residual, cmap="gray", vmin=vmin, vmax=vmax)
        axes[1, col].set_title("observed" if col == 0 else "residual", fontsize=9)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"{TOOL_NAME} - CPU/demo budgets, not a paper result", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
