"""``pinn-starlight samples`` - list validation samples on disk.

Standard library only; safe to run before the estimator extras are installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

from experiments.cli.context import EXIT_OK, apply_data_root, base_payload, print_json


def _list_manifest(path) -> list[dict]:
    from experiments.common.utils import paths

    if not Path(path).is_file():
        return []
    rows = paths.load_manifest(path, split="validation")
    return [
        {
            "sample_id": row["sample_id"],
            "scene_type": row.get("scene_type", row.get("background_type")),
            "intensity_level": row.get("intensity_level"),
        }
        for row in rows
    ]


def cmd_samples(args) -> int:
    apply_data_root(args)

    from experiments.common.utils import paths

    datasets = {}
    if args.stage in ("a", "all"):
        datasets["stage_a"] = _list_manifest(paths.SYNTHETIC_DATA_ROOT / "manifest.csv")
    if args.stage in ("b", "all"):
        datasets["stage_b"] = _list_manifest(
            paths.DATA_ROOT / "collections/synthetic_stage_b/manifest.csv"
        )
    payload = base_payload(
        "samples",
        data_root=str(paths.DATA_ROOT),
        data_root_source=paths.DATA_ROOT_SOURCE,
        datasets=datasets,
        warnings=[],
    )
    if not any(datasets.values()):
        payload["warnings"].append(paths.data_root_hint())
    if args.json:
        print_json(payload)
    else:
        for warning in payload["warnings"]:
            print(f"warning: {warning}", file=sys.stderr)
        for name, rows in datasets.items():
            print(f"{name}: {len(rows)} validation rows")
            for row in rows:
                print(
                    f"  {row['sample_id']:<52s} {row.get('scene_type') or row.get('background_type')}"
                )
    return EXIT_OK
