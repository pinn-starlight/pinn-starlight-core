"""Incrementally export Stage11 server results into priority ZIP parts.

Designed for a disposable one-day GPU server:
- keeps the generated synthetic_formal and synthetic_stage_b datasets;
- preserves every irreducible scientific prediction (background_pred.tif);
- preserves all model/checkpoint files;
- stores all source/config/CSV/JSON/history/log/provenance text;
- moves derivable duplicates (observed/residual TIFF and display PNG) to a
  lower-priority phase because they can be rebuilt locally;
- supports incremental batches, so completed work can be transferred and the
  ZIP deleted from the server without being repacked later.

Each ZIP is self-describing and contains an internal manifest.  The default
part target is 4 GiB; change with --part-gib.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PHASES = (
    ("P00_core", "source/config/text/provenance/runtime logs"),
    ("P01_synthetic_scientific", "synthetic_formal + synthetic_stage_b scientific content"),
    ("P02_formal_predictions", "formal background_pred.tif scientific predictions"),
    ("P03_models", "formal/selection checkpoints and final network states"),
    ("P04_selection_predictions", "selection-stage background predictions"),
    ("P05_secondary_predictions", "E3/E4/other non-primary scientific predictions"),
    ("P06_derived_duplicates", "server PNG previews + observed/residual TIFF duplicates"),
    ("P07_remaining", "remaining non-raw project result binaries"),
)
PHASE_ORDER = {name: i for i, (name, _) in enumerate(PHASES)}

TEXT_EXTS = {
    ".py", ".pyi", ".ps1", ".sh", ".bat",
    ".toml", ".lock", ".md", ".rst", ".txt", ".log",
    ".json", ".csv", ".tsv", ".yaml", ".yml", ".ini", ".cfg",
}
MODEL_EXTS = {".pt", ".pth", ".ckpt"}
IMAGE_PREVIEW_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
SCIENTIFIC_ARRAY_EXTS = {".tif", ".tiff", ".npy", ".npz"}

FORMAL_OUTPUT_MARKERS = (
    "/experiments/outputs/e2_benchmark/",
    "/experiments/outputs/final_pinn_validation/",
    "/experiments/outputs/stage_b/benchmark/",
    "/experiments/outputs/stage_b/final_pinn_validation/",
)
SELECTION_MARKERS = (
    "/experiments/outputs/selection/",
    "/experiments/outputs/stage_b/selection/",
)


def _norm_rel(path: Path) -> str:
    return "/" + path.relative_to(PROJECT_ROOT).as_posix()


def _is_under_cv5_formal(rel: str) -> bool:
    return (
        "/experiments/outputs/cv5/fold_" in rel
        and (
            "/e2_benchmark/" in rel
            or "/final_pinn_validation/" in rel
        )
    )


def _is_under_cv5_selection(rel: str) -> bool:
    return "/experiments/outputs/cv5/fold_" in rel and "/selection/" in rel


def _is_synthetic(rel: str) -> bool:
    return (
        rel.startswith("/collections/synthetic_formal/")
        or rel.startswith("/collections/synthetic_stage_b/")
    )


def _is_excluded_raw(rel: str) -> bool:
    # Original/raw data are intentionally not copied.  Generated synthetic
    # datasets are handled separately above.
    if _is_synthetic(rel):
        return False
    return (
        rel.startswith("/collections/")
        or rel.startswith("/test/")
        or rel.startswith("/data/")
    )


def _is_cache_or_admin(rel: str, archive_dir: Path) -> bool:
    parts = Path(rel.lstrip("/")).parts
    if any(x in parts for x in (".git", ".pixi", "__pycache__", ".pytest_cache", ".mypy_cache")):
        return True
    try:
        archive_rel = "/" + archive_dir.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        if rel == archive_rel or rel.startswith(archive_rel + "/"):
            return True
    except ValueError:
        pass
    return False


def classify(path: Path, archive_dir: Path) -> str | None:
    rel = _norm_rel(path)
    suffix = path.suffix.lower()
    name = path.name.lower()

    if _is_cache_or_admin(rel, archive_dir):
        return None
    if _is_excluded_raw(rel):
        return None

    # Generated synthetic scientific content is high priority.  Preview PNGs
    # are derivable and therefore intentionally deferred.
    if _is_synthetic(rel):
        if suffix in IMAGE_PREVIEW_EXTS:
            return "P06_derived_duplicates"
        return "P01_synthetic_scientific"

    # Keep the original collection manifest as provenance even though the raw
    # collection itself is not transferred.
    if rel == "/collections/manifest.csv":
        return "P00_core"

    # Models/checkpoints are reconstruction backup and go in their own bundle.
    if suffix in MODEL_EXTS:
        if "/experiments/outputs/" in rel:
            return "P03_models"
        # Source tree should not normally contain checkpoints, but retain any
        # non-raw project checkpoint as remaining data.
        return "P07_remaining"

    # All source/config/provenance/result tables/histories are tiny and crucial.
    if (
        suffix in TEXT_EXTS
        or "/experiments/outputs/" in rel and suffix in TEXT_EXTS
        or "/research_diagnostics/" in rel and suffix in TEXT_EXTS
    ):
        return "P00_core"

    if "/experiments/outputs/" in rel:
        # Irreducible scientific prediction.  observed/residual are derivable
        # from local synthetic observed + background_pred.
        if name == "background_pred.tif":
            if any(marker in rel for marker in FORMAL_OUTPUT_MARKERS) or _is_under_cv5_formal(rel):
                return "P02_formal_predictions"
            if any(marker in rel for marker in SELECTION_MARKERS) or _is_under_cv5_selection(rel):
                return "P04_selection_predictions"
            return "P05_secondary_predictions"

        # These are exactly reconstructible from dataset + background_pred.
        if name in {"observed.tif", "residual_pred.tif"} or suffix in IMAGE_PREVIEW_EXTS:
            return "P06_derived_duplicates"

        # Preserve any other scientific arrays; classify formal ones early.
        if suffix in SCIENTIFIC_ARRAY_EXTS:
            if any(marker in rel for marker in FORMAL_OUTPUT_MARKERS) or _is_under_cv5_formal(rel):
                return "P02_formal_predictions"
            if any(marker in rel for marker in SELECTION_MARKERS) or _is_under_cv5_selection(rel):
                return "P04_selection_predictions"
            return "P05_secondary_predictions"

        return "P07_remaining"

    # The rest of the project source snapshot is core unless it is a binary
    # artifact.  This keeps exact Stage11 code/docs/pixi files.
    if not suffix or suffix in TEXT_EXTS:
        return "P00_core"

    # Small non-output project resources are kept as a last-priority backup.
    return "P07_remaining"


def _signature(path: Path) -> dict:
    st = path.stat()
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_text(command: list[str]) -> str:
    try:
        proc = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
        )
        return proc.stdout
    except Exception as exc:
        return f"<unavailable: {type(exc).__name__}: {exc}>\n"


def _environment_snapshot(label: str) -> dict:
    info = {
        "label": label,
        "time_unix": time.time(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "project_root": str(PROJECT_ROOT.resolve()),
    }
    try:
        import torch
        info.update(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "cuda_devices": [
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                ] if torch.cuda.is_available() else [],
            }
        )
    except Exception as exc:
        info["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    info["nvidia_smi"] = _run_text(["nvidia-smi"])
    info["git_status"] = _run_text(["git", "status", "--short", "--branch"])
    info["git_rev_parse"] = _run_text(["git", "rev-parse", "HEAD"])
    info["pixi_list"] = _run_text(["pixi", "list"])
    return info


def _load_json(path: Path, default):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_inventory(path: Path, rows: list[dict]):
    fields = [
        "batch", "label", "phase", "part", "relative_path",
        "size_bytes", "mtime_ns", "archive_name",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _group_parts(files: list[Path], target_bytes: int) -> list[list[Path]]:
    parts = []
    current = []
    current_bytes = 0
    for path in files:
        size = path.stat().st_size
        if current and current_bytes + size > target_bytes:
            parts.append(current)
            current = []
            current_bytes = 0
        current.append(path)
        current_bytes += size
        if size >= target_bytes:
            parts.append(current)
            current = []
            current_bytes = 0
    if current:
        parts.append(current)
    return parts


def _zip_part(
    zip_path: Path,
    files: list[Path],
    *,
    batch: int,
    label: str,
    phase: str,
    part_number: int,
    compression: str,
    compresslevel: int,
    environment: dict,
):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if compression == "store":
        mode = ZIP_STORED
        kwargs = {}
    else:
        mode = ZIP_DEFLATED
        kwargs = {"compresslevel": int(compresslevel)}

    manifest_files = []
    with ZipFile(zip_path, "w", compression=mode, allowZip64=True, strict_timestamps=False, **kwargs) as zf:
        for path in files:
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            sig = _signature(path)
            zf.write(path, arcname=rel)
            manifest_files.append(
                {
                    "relative_path": rel,
                    "size_bytes": sig["size"],
                    "mtime_ns": sig["mtime_ns"],
                }
            )
        internal = {
            "schema": 2,
            "batch": batch,
            "label": label,
            "phase": phase,
            "part": part_number,
            "project_root_name": PROJECT_ROOT.name,
            "files": manifest_files,
            "environment": environment,
        }
        zf.writestr(
            "__stage11_archive__/manifest.json",
            json.dumps(internal, ensure_ascii=False, indent=2),
        )

    return {
        "archive_name": zip_path.name,
        "batch": batch,
        "label": label,
        "phase": phase,
        "part": part_number,
        "file_count": len(files),
        "uncompressed_bytes": sum(p.stat().st_size for p in files),
        "archive_bytes": zip_path.stat().st_size,
        "sha256": _sha256(zip_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=PROJECT_ROOT / "stage11_transfer_archives",
    )
    parser.add_argument("--label", default="snapshot")
    parser.add_argument("--part-gib", type=float, default=4.0)
    parser.add_argument(
        "--phases",
        nargs="*",
        choices=[x[0] for x in PHASES],
        default=None,
        help="default: all phases",
    )
    parser.add_argument("--compression", choices=("fast", "store"), default="fast")
    parser.add_argument("--compresslevel", type=int, default=1)
    parser.add_argument(
        "--full",
        action="store_true",
        help="ignore incremental state and archive all matching files again",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    archive_dir = args.archive_dir.expanduser()
    if not archive_dir.is_absolute():
        archive_dir = (PROJECT_ROOT / archive_dir).resolve()
    archive_dir.mkdir(parents=True, exist_ok=True)

    state_path = archive_dir / "archive_state.json"
    index_path = archive_dir / "archive_index.json"
    inventory_path = archive_dir / "archive_inventory.csv"

    state = _load_json(state_path, {"schema": 2, "next_batch": 1, "files": {}})
    index = _load_json(
        index_path,
        {
            "schema": 2,
            "project_root_name": PROJECT_ROOT.name,
            "phases": [{"name": n, "description": d} for n, d in PHASES],
            "archives": [],
            "batches": [],
        },
    )
    batch = int(state.get("next_batch", 1))
    selected_phases = set(args.phases or [x[0] for x in PHASES])

    all_files = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        phase = classify(path, archive_dir)
        if phase is None or phase not in selected_phases:
            continue
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        sig = _signature(path)
        if not args.full and state.get("files", {}).get(rel) == sig:
            continue
        all_files.append((phase, path, rel, sig))

    all_files.sort(key=lambda x: (PHASE_ORDER[x[0]], x[2]))
    by_phase = {name: [] for name, _ in PHASES}
    for phase, path, rel, sig in all_files:
        by_phase[phase].append(path)

    total_files = len(all_files)
    total_bytes = sum(path.stat().st_size for _, path, _, _ in all_files)
    print(f"Stage11 archive batch B{batch:03d} label={args.label!r}")
    print(f"new/changed files: {total_files}")
    print(f"uncompressed: {total_bytes / (1024**3):.3f} GiB")
    for phase, _ in PHASES:
        files = by_phase[phase]
        if files:
            size = sum(p.stat().st_size for p in files)
            print(f"  {phase}: {len(files)} files, {size / (1024**3):.3f} GiB")

    if args.dry_run:
        return
    if not all_files:
        print("Nothing new to archive.")
        return

    environment = _environment_snapshot(args.label)
    target_bytes = max(1, int(args.part_gib * (1024**3)))
    created = []
    inventory_rows = []
    compression = "store" if args.compression == "store" else "fast"

    for phase, _ in PHASES:
        files = by_phase[phase]
        if not files:
            continue
        parts = _group_parts(files, target_bytes)
        for part_no, part_files in enumerate(parts, 1):
            zip_name = f"B{batch:03d}_{phase}_part{part_no:03d}.zip"
            zip_path = archive_dir / zip_name
            if zip_path.exists():
                raise FileExistsError(f"archive already exists: {zip_path}")
            record = _zip_part(
                zip_path,
                part_files,
                batch=batch,
                label=args.label,
                phase=phase,
                part_number=part_no,
                compression=compression,
                compresslevel=args.compresslevel,
                environment=environment,
            )
            created.append(record)
            for path in part_files:
                rel = path.relative_to(PROJECT_ROOT).as_posix()
                sig = _signature(path)
                inventory_rows.append(
                    {
                        "batch": batch,
                        "label": args.label,
                        "phase": phase,
                        "part": part_no,
                        "relative_path": rel,
                        "size_bytes": sig["size"],
                        "mtime_ns": sig["mtime_ns"],
                        "archive_name": zip_name,
                    }
                )
                state.setdefault("files", {})[rel] = sig
            print(
                f"created {zip_name}: "
                f"{record['archive_bytes'] / (1024**3):.3f} GiB "
                f"sha256={record['sha256'][:16]}..."
            )

    index["archives"].extend(created)
    index["batches"].append(
        {
            "batch": batch,
            "label": args.label,
            "time_unix": time.time(),
            "environment": environment,
            "archive_names": [x["archive_name"] for x in created],
        }
    )
    state["next_batch"] = batch + 1

    # Merge old + new inventory if present.
    old_rows = []
    if inventory_path.is_file():
        with inventory_path.open("r", encoding="utf-8-sig", newline="") as f:
            old_rows = list(csv.DictReader(f))
    _write_inventory(inventory_path, old_rows + inventory_rows)
    _write_json(index_path, index)
    _write_json(state_path, state)

    order_lines = [
        "Stage11 transfer order (higher priority first within each batch)",
        "===============================================================",
        "",
        "Always transfer archive_index.json and archive_inventory.csv as tiny metadata.",
        "If time is limited, prioritize P00 -> P01 -> P02 -> P03 -> P04 -> P05 -> P06 -> P07.",
        "",
    ]
    for rec in sorted(index["archives"], key=lambda x: (x["batch"], PHASE_ORDER[x["phase"]], x["part"])):
        order_lines.append(
            f"{rec['archive_name']}  "
            f"{rec['archive_bytes']/(1024**3):.3f} GiB  "
            f"sha256={rec['sha256']}"
        )
    (archive_dir / "TRANSFER_ORDER.txt").write_text("\n".join(order_lines) + "\n", encoding="utf-8")

    summary = [
        f"batch=B{batch:03d}",
        f"label={args.label}",
        f"created_archives={len(created)}",
        f"new_files={total_files}",
        f"new_uncompressed_gib={total_bytes/(1024**3):.3f}",
        "",
        "You may transfer and then delete the Bxxx_*.zip files from the server.",
        "Do NOT delete archive_state.json/index/inventory until the very end.",
        "The incremental state prevents already-exported source files from being repacked.",
    ]
    (archive_dir / f"B{batch:03d}_SUMMARY.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("index:", index_path)
    print("order:", archive_dir / "TRANSFER_ORDER.txt")


if __name__ == "__main__":
    main()
