"""Finite-value aggregation for the benchmark result tables."""

from __future__ import annotations

import numpy as np


def finite_values(rows, key):
    """Return the finite float values of ``key`` across ``rows``.

    Missing keys, ``None``, non-numeric text and non-finite numbers are skipped,
    so a metric that is undefined for some rows cannot abort a whole run.
    """
    values = []
    for row in rows:
        value = row.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def finite_mean(rows, key):
    """Mean of the finite values of ``key``, or ``nan`` when there are none."""
    values = finite_values(rows, key)
    return float(np.mean(values)) if values else float("nan")


def finite_std(rows, key):
    """Population std of the finite values of ``key``, or ``nan`` when none."""
    values = finite_values(rows, key)
    return float(np.std(values)) if values else float("nan")
