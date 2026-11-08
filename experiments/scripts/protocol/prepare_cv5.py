"""Create five grouped split *views* for Stage11-A and/or Stage11-B.

The scientific TIFF arrays are not duplicated or regenerated.  Each fold view
contains only a manifest whose ``split`` column is reassigned by camera base,
plus protocol/provenance JSON files.  Every variant from one camera base stays
in the same group.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

from experiments.configs.split_protocols import BASE_IDS, CV5_FOLDS

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _data_root() -> Path:
    configured = os.environ.get("PINN_STARLIGHT_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    if (PROJECT_ROOT / "collections").exists() or (PROJECT_ROOT / "test").exists():
        return PROJECT_ROOT
    if (PROJECT_ROOT / "data" / "collections").exists():
        return PROJECT_ROOT / "data"
    return PROJECT_ROOT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty fold manifest")
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _paths(stage: str) -> tuple[Path, Path]:
    collection_root = _data_root() / "collections"
    if stage == "a":
        root = collection_root / "synthetic_formal"
    elif stage == "b":
        root = collection_root / "synthetic_stage_b"
    else:
        raise ValueError("stage must be 'a' or 'b'")
    return root, root / "manifest.csv"


def prepare_stage(stage: str, *, force: bool = False) -> list[Path]:
    source_root, source_manifest = _paths(stage)
    if not source_manifest.is_file():
        generator = (
            "python -m experiments.common.data.generate_synthetic --force"
            if stage == "a"
            else "python -m experiments.common.data.generate_stage_b --force"
        )
        raise FileNotFoundError(
            f"Stage11-{stage.upper()} ordinary manifest is missing: {source_manifest}\n"
            f"Generate the ordinary dataset first with: {generator}"
        )

    source_rows = _read_csv(source_manifest)
    present = {str(row.get("base_id")) for row in source_rows}
    missing = [base_id for base_id in BASE_IDS if base_id not in present]
    if missing:
        raise ValueError(
            f"Stage11-{stage.upper()} source manifest is missing camera bases: "
            + ", ".join(missing)
        )

    protocol_path = source_root / "dataset_protocol.json"
    source_protocol = (
        json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol_path.is_file()
        else {}
    )
    outputs = []
    for fold, split_map in enumerate(CV5_FOLDS):
        fold_root = source_root / "cv5" / f"fold_{fold}"
        if fold_root.exists() and any(fold_root.iterdir()):
            if not force:
                existing_protocol_path = fold_root / "dataset_protocol.json"
                existing_manifest = fold_root / "manifest.csv"
                reusable = False
                if existing_protocol_path.is_file() and existing_manifest.is_file():
                    try:
                        existing_protocol = json.loads(existing_protocol_path.read_text(encoding="utf-8"))
                        provenance = existing_protocol.get("source_dataset_provenance", {})
                        reusable = (
                            provenance.get("manifest_sha256") == _sha256(source_manifest)
                            and existing_protocol.get("split_protocol", {}).get("base_split") == split_map
                        )
                    except (OSError, json.JSONDecodeError):
                        reusable = False
                if reusable:
                    outputs.append(existing_manifest)
                    continue
                raise FileExistsError(
                    f"CV5 view exists but is stale or unverifiable: {fold_root}. "
                    "Use --force to rebuild views."
                )
            shutil.rmtree(fold_root)
        fold_root.mkdir(parents=True, exist_ok=True)

        rows = []
        for original in source_rows:
            row = dict(original)
            base_id = str(row["base_id"])
            row["split"] = split_map[base_id]
            rows.append(row)

        manifest = fold_root / "manifest.csv"
        _write_csv(manifest, rows)
        counts = {
            name: sum(row["split"] == name for row in rows)
            for name in ("train", "validation", "test")
        }
        expected_factor = 6 if stage == "a" else 8
        expected_counts = {
            "train": 3 * expected_factor,
            "validation": expected_factor,
            "test": expected_factor,
        }
        if counts != expected_counts:
            raise AssertionError(
                f"Stage11-{stage.upper()} fold {fold} counts {counts} != {expected_counts}"
            )

        (fold_root / "base_split.json").write_text(
            json.dumps(split_map, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        view_protocol = dict(source_protocol)
        view_protocol.update(
            {
                "manifest": str(manifest),
                "split_protocol": {
                    "mode": "cv5_grouped_camera_base",
                    "fold": fold,
                    "base_split": split_map,
                    "independent_group": "camera_reference_base_id",
                    "test_once_per_base": True,
                    "validation_once_per_base": True,
                    "fold0_matches_standard_split": True,
                    "arrays_regenerated": False,
                },
                "base_split": split_map,
                "split_counts": counts,
                "source_dataset_provenance": {
                    "manifest": str(source_manifest),
                    "manifest_sha256": _sha256(source_manifest),
                    "dataset_protocol": str(protocol_path) if protocol_path.is_file() else None,
                    "dataset_protocol_sha256": _sha256(protocol_path) if protocol_path.is_file() else None,
                },
            }
        )
        (fold_root / "dataset_protocol.json").write_text(
            json.dumps(view_protocol, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        outputs.append(manifest)
    return outputs


def main(*, stage: str, force: bool = False) -> None:
    stages = ("a", "b") if stage == "all" else (stage,)
    for current in stages:
        outputs = prepare_stage(current, force=force)
        print(f"Stage11-{current.upper()} CV5 split views prepared:")
        for manifest in outputs:
            print(" ", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("a", "b", "all"), default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(stage=args.stage, force=args.force)
