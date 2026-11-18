"""Method registry for the demo CLI / ``experiments.common.pipeline``.

Lightweight on purpose (no numpy/torch imports) so the CLI can read it at startup.
These names are *not* part of the frozen formal protocol; ``balanced`` is demo-only.
"""

from __future__ import annotations

# The four paper methods, in table order.
PAPER_METHODS = ("fft_gaussian", "fourier_ls", "pinn", "final_pinn")

# Everything the demo pipeline can run.
METHOD_NAMES = (*PAPER_METHODS, "balanced")

DISPLAY_NAMES = {
    "fft_gaussian": "FFT-Gaussian",
    "fourier_ls": "Cosine-LS",
    "pinn": "Original PINN",
    "final_pinn": "SGC-PINN",
    "balanced": "Balanced (anchored)",
}

# CLI ``--mode`` shortcuts.
MODE_TO_METHOD = {"star": "final_pinn", "nebula": "pinn", "balanced": "balanced"}
