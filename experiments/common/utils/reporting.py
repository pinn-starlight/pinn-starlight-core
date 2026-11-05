"""Numeric aggregation helpers shared by the benchmark result tables.

Why a shared module
-------------------
``benchmark.py`` (Stage-A) and ``stage_b/formal_run.py`` (Stage-B) each grew
their own copy of "collect the finite values of one metric column, then take
mean/std".  The two copies were not equivalent:

* the Stage-A copy required ``isinstance(value, (int, float, np.number))``,
  which silently dropped numeric *strings*;
* the Stage-B copy used ``float(value)``, which accepts ``"0.5"`` as well.

Rows produced in memory carry Python/NumPy floats, so both worked there, but
rows read back from a CSV carry strings.  Using the ``float()`` form here keeps
a single behaviour that is correct for both cases.

Non-finite values (``nan``/``inf``), ``None``, missing keys and non-numeric text
are excluded rather than raising: a metric that is legitimately undefined for
some rows must not abort a whole run.
"""
from __future__ import annotations

import numpy as np


def finite_values(rows, key):
    """Return the finite float values of ``key`` across ``rows``.

    Missing keys, ``None``, non-numeric text and non-finite numbers are skipped.
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
