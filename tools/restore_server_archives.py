"""Restore Stage11 incremental archive ZIPs into a local project tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

INTERNAL_MANIFEST = "__stage11_archive__/manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_internal_manifest(path: Path) -> dict:
    with ZipFile(path, "r") as zf:
        if INTERNAL_MANIFEST not in zf.namelist():
            raise ValueError(f"not a Stage11 archive: {path}")
        return json.loads(zf.read(INTERNAL_MANIFEST).decode("utf-8"))


def safe_target(root: Path, member: str) -> Path:
    member_path = Path(member)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError(f"unsafe archive member: {member}")
    target = (root / member_path).resolve()
    root_resolved = root.resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise ValueError(f"archive member escapes target: {member}")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_dir", type=Path)
    parser.add_argument("target_root", type=Path)
    parser.add_argument("--phases", nargs="*", default=None)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="fail if a target file already exists",
    )
    args = parser.parse_args()

    archive_dir = args.archive_dir.expanduser().resolve()
    target_root = args.target_root.expanduser().resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    archives = []
    for path in archive_dir.glob("B*_P*_part*.zip"):
        manifest = read_internal_manifest(path)
        phase = manifest["phase"]
        if args.phases and phase not in set(args.phases):
            continue
        archives.append((manifest["batch"], phase, manifest["part"], path, manifest))
    archives.sort(key=lambda x: (x[0], x[1], x[2]))

    if not archives:
        raise FileNotFoundError("no Stage11 archive ZIPs found")

    index_path = archive_dir / "archive_index.json"
    expected_sha = {}
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        expected_sha = {r["archive_name"]: r["sha256"] for r in index.get("archives", [])}

    file_count = 0
    for batch, phase, part, path, manifest in archives:
        print(f"[B{batch:03d} {phase} part {part:03d}] {path.name}")
        if path.name in expected_sha:
            actual = sha256(path)
            if actual != expected_sha[path.name]:
                raise OSError(f"archive SHA256 mismatch: {path.name}")

        with ZipFile(path, "r") as zf:
            bad = zf.testzip()
            if bad:
                raise OSError(f"ZIP CRC failure in {path.name}: {bad}")
            if args.verify_only:
                file_count += len(manifest["files"])
                continue

            for info in manifest["files"]:
                member = info["relative_path"]
                target = safe_target(target_root, member)
                if target.exists() and args.no_overwrite:
                    raise FileExistsError(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member, "r") as src, target.open("wb") as dst:
                    while True:
                        chunk = src.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
                file_count += 1

    print(f"{'verified' if args.verify_only else 'restored'} files: {file_count}")
    print("target:", target_root)


if __name__ == "__main__":
    main()
