"""Stage11 camera-base split protocols.

The five camera-acquired reference bases are the independent grouping units.
All semi-synthetic variants derived from one base must remain in the same split.

Two protocols are supported:
- ``standard``: the historical fixed 3/1/1 split, retained for ordinary runs;
- ``cv5``: five grouped folds.  Each base is test exactly once and validation
  exactly once.  Fold 0 deliberately matches the standard 3/1/1 split so the
  ordinary protocol is also one well-defined fold of the rotation.
"""
from __future__ import annotations

import os
from copy import deepcopy

BASE_IDS = (
    "clean_single_001",
    "clean_single_002",
    "clean_single_003",
    "clean_stack_001",
    "clean_stack_002",
)

STANDARD_BASE_SPLIT = {
    "clean_single_003": "train",
    "clean_stack_002": "train",
    "clean_stack_001": "train",
    "clean_single_002": "validation",
    "clean_single_001": "test",
}

# Test -> validation pairing is fixed before formal CV runs.  It avoids using
# both stacked references as validation/test in the same fold and makes fold 0
# exactly equal to the standard split.
_CV5_TEST_VALIDATION = (
    ("clean_single_001", "clean_single_002"),
    ("clean_single_002", "clean_stack_001"),
    ("clean_single_003", "clean_stack_002"),
    ("clean_stack_001", "clean_single_003"),
    ("clean_stack_002", "clean_single_001"),
)

CV5_SEQUENCE: tuple[str, ...] = (
    "clean_single_001",
    "clean_single_002",
    "clean_stack_001",
    "clean_single_003",
    "clean_stack_002",
)

# Pre-specified balanced circular evaluation rule:
#   current item in CV5_SEQUENCE -> test
#   next item cyclically         -> validation
#   remaining three              -> train
#
# The sequence is fixed before formal evaluation.  It preserves the original
# standard Stage11-A split as fold 0, then alternates single-frame and stacked
# reference fields as far as possible.  No performance result is used to choose
# any validation/test pairing.
CV5_FOLDS: tuple[dict[str, str], ...] = tuple(
    {
        key: (
            "test"
            if key == CV5_SEQUENCE[fold]
            else "validation"
            if key == CV5_SEQUENCE[(fold + 1) % len(CV5_SEQUENCE)]
            else "train"
        )
        for key in CV5_SEQUENCE
    }
    for fold in range(len(CV5_SEQUENCE))
)

def validate_protocols() -> None:
    if set(STANDARD_BASE_SPLIT) != set(BASE_IDS):
        raise AssertionError("standard split does not cover exactly the five base IDs")
    if len(CV5_FOLDS) != len(BASE_IDS):
        raise AssertionError("cv5 must contain exactly five folds")

    test_counts = {base_id: 0 for base_id in BASE_IDS}
    validation_counts = {base_id: 0 for base_id in BASE_IDS}
    for fold_index, split in enumerate(CV5_FOLDS):
        if set(split) != set(BASE_IDS):
            raise AssertionError(f"cv5 fold {fold_index} has the wrong base-ID set")
        counts = {name: sum(value == name for value in split.values()) for name in ("train", "validation", "test")}
        if counts != {"train": 3, "validation": 1, "test": 1}:
            raise AssertionError(f"cv5 fold {fold_index} is not a 3/1/1 split: {counts}")
        for base_id, value in split.items():
            if value == "test":
                test_counts[base_id] += 1
            elif value == "validation":
                validation_counts[base_id] += 1
    if any(value != 1 for value in test_counts.values()):
        raise AssertionError(f"each base must be test once: {test_counts}")
    if any(value != 1 for value in validation_counts.values()):
        raise AssertionError(f"each base must be validation once: {validation_counts}")
    if CV5_FOLDS[0] != STANDARD_BASE_SPLIT:
        raise AssertionError("cv5 fold 0 must match the standard split")


def normalize_mode(mode: str | None) -> str:
    value = str(mode or "standard").strip().lower()
    aliases = {"ordinary": "standard", "normal": "standard", "5fold": "cv5", "fivefold": "cv5"}
    value = aliases.get(value, value)
    if value not in {"standard", "cv5"}:
        raise ValueError("split mode must be 'standard' or 'cv5'")
    return value


def validate_fold(fold: int | str | None) -> int:
    if fold is None:
        raise ValueError("cv5 mode requires a fold index in [0, 4]")
    value = int(fold)
    if not 0 <= value < 5:
        raise ValueError("cv5 fold index must lie in [0, 4]")
    return value


def base_split(mode: str = "standard", fold: int | None = None) -> dict[str, str]:
    mode = normalize_mode(mode)
    if mode == "standard":
        return deepcopy(STANDARD_BASE_SPLIT)
    return deepcopy(CV5_FOLDS[validate_fold(fold)])


def active_protocol_from_env() -> tuple[str, int | None]:
    mode = normalize_mode(os.environ.get("PINN_STARLIGHT_SPLIT_MODE", "standard"))
    if mode == "standard":
        return mode, None
    fold = validate_fold(os.environ.get("PINN_STARLIGHT_CV_FOLD"))
    return mode, fold


def active_base_split() -> dict[str, str]:
    mode, fold = active_protocol_from_env()
    return base_split(mode, fold)


validate_protocols()
