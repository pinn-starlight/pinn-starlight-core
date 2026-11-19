#!/usr/bin/env python3
"""Build and verify distributable artifacts for ``pinn-starlight``.

Two products:

* ``release`` (default) - wheel + sdist in ``build/dist/``, verified by
  installing the wheel into a throwaway venv *without* dependencies and
  checking that the CLI degrades gracefully instead of crashing;
* ``--portable`` - a relocatable folder (embedded CPython + ``--target``
  site-packages + launchers) handed out as a GitHub Release asset.  No venv is
  involved: ``python -m venv`` bakes absolute paths into the console-script
  shebang and ``pyvenv.cfg``, so a venv-based bundle dies as soon as it is
  unzipped somewhere else.

Usage
-----
    python tools/build_release.py
    python tools/build_release.py --portable
    python tools/build_release.py --portable --skip-deps   # structure only
    python tools/build_release.py --skip-verify

Exit codes: 0 success, 1 build/verify failure.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parents[1]
DIST = TOOL_ROOT / "build/dist"
WORK = TOOL_ROOT / "build/tmp/release"
MIRROR = "https://mirrors.ustc.edu.cn/pypi/simple"
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"

# Only what `estimate` imports: the CLI never touches torchvision, lightning,
# segmentation-models-pytorch, jupyter or jsonargparse.
RUNTIME_DEPS = (
    "torch",
    "numpy",
    "scipy",
    "tqdm",
    "astropy",
    "photutils",
    "scikit-image",
    "tifffile",
    "pillow",
    "matplotlib",
    "rawpy",
)

RUN_PY = '''\
"""Bundled entry point: put site-packages/ first, then run the real CLI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "site-packages"))

from experiments.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
'''

LAUNCHER_SH = """\
#!/usr/bin/env bash
here="$(cd "$(dirname "$0")" && pwd)"
exec "$here/python/bin/{exe}" -I "$here/run.py" "$@"
"""

LAUNCHER_BAT = """\
@echo off\r
"%~dp0python\\python.exe" -I "%~dp0run.py" %*\r
"""


def _run(
    cmd: list[str], *, cwd: Path = TOOL_ROOT, env: dict | None = None
) -> subprocess.CompletedProcess:
    print("  $ " + " ".join(str(part) for part in cmd))
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)


def _uv_env() -> dict:
    """Keep uv's cache and tool dirs inside the repo (the home cache may be read-only)."""
    env = dict(os.environ)
    env.setdefault("UV_CACHE_DIR", str(TOOL_ROOT / "build/tmp/uv_cache"))
    env.setdefault("UV_TOOL_DIR", str(TOOL_ROOT / "build/tmp/uv_tool"))
    env.setdefault("UV_TOOL_BIN_DIR", str(TOOL_ROOT / "build/tmp/uv_bin"))
    return env


# --------------------------------------------------------------------------
# wheel + clean-venv verification
# --------------------------------------------------------------------------
def build_artifacts(clean: bool = True) -> Path:
    """Build wheel + sdist into ``build/dist`` and return the wheel path."""
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required: install it or run `uv build` manually")
    if clean and WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)

    print("[1/4] building wheel + sdist")
    result = _run([uv, "build", "--out-dir", str(DIST)], env=_uv_env())
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit("uv build failed")

    wheels = sorted(DIST.glob("pinn_starlight_core-*.whl"))
    if not wheels:
        raise SystemExit(f"no wheel produced in {DIST}")
    print(f"  wheel: {wheels[-1].name} ({wheels[-1].stat().st_size / 1024:.1f} KiB)")
    return wheels[-1]


def verify_wheel(wheel: Path) -> bool:
    """Install the wheel without dependencies and check graceful degradation."""
    print("[2/4] verifying the wheel in a clean venv")
    WORK.mkdir(parents=True, exist_ok=True)
    venv = WORK / "venv"
    if venv.exists():
        shutil.rmtree(venv)
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)

    python = venv / "bin/python"
    exe = venv / "bin/pinn-starlight"
    if os.name == "nt":
        python = venv / "Scripts/python.exe"
        exe = venv / "Scripts/pinn-starlight.exe"

    install = subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", "--no-deps", str(wheel)],
        text=True,
        capture_output=True,
    )
    if install.returncode != 0:
        print(install.stderr, file=sys.stderr)
        return False

    checks = [
        ("version", [str(exe), "version"], {0}),
        ("doctor", [str(exe), "doctor"], {0, 3}),
        ("samples", [str(exe), "samples"], {0, 3}),
        ("estimate", [str(exe), "estimate", "--image", "missing.jpg"], {3, 4}),
    ]
    ok = True
    for name, cmd, allowed in checks:
        result = subprocess.run(cmd, text=True, capture_output=True)
        crashed = "Traceback (most recent call last)" in result.stderr
        passed = result.returncode in allowed and not crashed
        ok = ok and passed
        print(f"  {name:<10} exit={result.returncode} {'ok' if passed else 'FAIL'}")
        if not passed:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
    return ok


# --------------------------------------------------------------------------
# portable bundle
# --------------------------------------------------------------------------
SYSTEM_PREFIXES = {"/usr", "/usr/local", "/opt/homebrew", "/Library/Frameworks"}


def relocatable_python_root(override: Path | None = None) -> Path:
    """Locate a self-contained CPython tree that can be copied into the bundle."""
    base = Path(override or sys.base_prefix).resolve()
    if os.name != "nt" and str(base) in SYSTEM_PREFIXES:
        raise SystemExit(
            f"{base} is a system Python and cannot be bundled.\n"
            "Build with a standalone CPython (e.g. `uv python install 3.11`) or pass "
            "--python-dir /path/to/python."
        )
    posix_exe = next(
        (
            item
            for item in sorted((base / "bin").glob("python3.*"))
            if not item.name.endswith("-config")
        ),
        None,
    )
    if os.name == "nt":
        if not (base / "python.exe").is_file():
            raise SystemExit(f"{base} does not look like a Python installation")
    elif posix_exe is None:
        raise SystemExit(f"{base} has no bin/python3.* to copy")
    return base


def bundle_name(wheel: Path) -> str:
    package_version = wheel.name.split("-")[1]
    system = {"linux": "linux", "win32": "windows", "darwin": "macos"}.get(
        sys.platform, sys.platform
    )
    return f"pinn-starlight-{package_version}-{system}-{platform.machine().lower()}"


def assemble_portable(
    wheel: Path,
    out: Path | None = None,
    *,
    skip_deps: bool = False,
    python_dir: Path | None = None,
) -> Path:
    """Build a relocatable folder: embedded CPython + site-packages + launchers."""
    print("[3/4] assembling portable bundle")
    target = out or (DIST / bundle_name(wheel))
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    # 1. Embedded interpreter.  symlinks=False dereferences the copy, which is
    #    what makes the tree relocatable (uv's Python is symlinked internally).
    python_root = relocatable_python_root(python_dir)
    print(f"  python : {python_root}")
    shutil.copytree(
        python_root,
        target / "python",
        symlinks=False,
        ignore=shutil.ignore_patterns("__pycache__", "site-packages"),
    )
    posix_exe = next(
        (
            item
            for item in sorted((target / "python/bin").glob("python3.*"))
            if not item.name.endswith("-config")
        ),
        None,
    )
    exe_name = posix_exe.name if posix_exe else "python.exe"

    # 2. Dependencies and the package itself, installed with --target so no
    #    venv (and therefore no absolute shebang) is involved.
    site = target / "site-packages"
    site.mkdir()
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required to assemble the bundle")
    if skip_deps:
        print("  deps   : skipped (--skip-deps: only version/doctor/samples will work)")
    else:
        # torch must be resolved with the CPU index as the ONLY index.  If the
        # mirror is visible at the same time, uv prefers its CUDA build and the
        # bundle grows by ~2 GB of nvidia-* wheels.
        torch_install = _run(
            [
                uv,
                "pip",
                "install",
                "--target",
                str(site),
                "--python-version",
                "3.11",
                "--index-url",
                CPU_TORCH_INDEX,
                "torch",
            ],
            env=_uv_env(),
        )
        if torch_install.returncode != 0:
            print(torch_install.stdout)
            print(torch_install.stderr, file=sys.stderr)
            raise SystemExit("installing CPU torch failed")

        deps = [name for name in RUNTIME_DEPS if name != "torch"]
        install = _run(
            [
                uv,
                "pip",
                "install",
                "--target",
                str(site),
                "--python-version",
                "3.11",
                "--index-url",
                MIRROR,
                *deps,
            ],
            env=_uv_env(),
        )
        if install.returncode != 0:
            print(install.stdout)
            print(install.stderr, file=sys.stderr)
            raise SystemExit("installing the runtime dependencies failed")
    package = _run(
        [uv, "pip", "install", "--target", str(site), "--no-deps", str(wheel)],
        env=_uv_env(),
    )
    if package.returncode != 0:
        print(package.stdout)
        print(package.stderr, file=sys.stderr)
        raise SystemExit("installing the wheel into the bundle failed")

    # 3. Entry point and launchers (relative to their own location).
    (target / "run.py").write_text(RUN_PY, encoding="utf-8")
    launcher = target / "pinn-starlight.sh"
    launcher.write_text(LAUNCHER_SH.format(exe=exe_name), encoding="utf-8")
    launcher.chmod(0o755)
    (target / "pinn-starlight.bat").write_text(LAUNCHER_BAT, encoding="utf-8")
    (target / "README.txt").write_text(
        "PINN-Starlight portable bundle\n"
        "=============================\n\n"
        f"{bundle_name(wheel)} / Python {platform.python_version()}\n\n"
        "No Python installation is needed: the interpreter is inside this folder.\n\n"
        "  Linux : ./pinn-starlight.sh doctor\n"
        "          ./pinn-starlight.sh estimate --image photo.jpg --mode star\n"
        "  Windows: pinn-starlight.bat doctor\n"
        "           pinn-starlight.bat estimate --image photo.jpg --mode star\n\n"
        "Runs on CPU with demo budgets; it does not reproduce the paper numbers.\n"
        "`estimate --sample` additionally needs the synthetic dataset tree.\n",
        encoding="utf-8",
    )
    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    print(f"  bundle : {target} ({size / 1024**2:.0f} MiB)")

    # GitHub Release wants a single asset; zip next to the folder.
    archive = shutil.make_archive(str(target), "zip", root_dir=target.parent, base_dir=target.name)
    archive_path = Path(archive)
    print(f"  zip    : {archive_path.name} ({archive_path.stat().st_size / 1024**2:.0f} MiB)")
    return target


def verify_relocated(bundle: Path, image: Path | None = None) -> bool:
    """Copy the bundle elsewhere and run it there: catches non-relocatable output."""
    print("[4/4] verifying the bundle after relocation")
    with tempfile.TemporaryDirectory(prefix="pinn-relocate-") as tmp:
        moved = Path(tmp) / bundle.name
        shutil.copytree(bundle, moved, symlinks=True)
        exe = moved / ("pinn-starlight.bat" if os.name == "nt" else "pinn-starlight.sh")
        checks = [("version", [str(exe), "version"], {0}), ("doctor", [str(exe), "doctor"], {0, 3})]
        if image is not None and image.is_file():
            checks.append(
                (
                    "estimate",
                    [
                        str(exe),
                        "estimate",
                        "--image",
                        str(image),
                        "--mode",
                        "star",
                        "--steps",
                        "20",
                        "--max-side",
                        "32",
                        "--no-figure",
                        "--quiet",
                    ],
                    {0},
                )
            )
        ok = True
        for name, cmd, allowed in checks:
            result = subprocess.run(cmd, text=True, capture_output=True)
            passed = result.returncode in allowed
            ok = ok and passed
            print(f"  {name:<10} exit={result.returncode} {'ok' if passed else 'FAIL'}")
            if not passed:
                print(result.stdout[-2000:])
                print(result.stderr[-2000:], file=sys.stderr)
        return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build pinn-starlight release artifacts.")
    parser.add_argument("--portable", action="store_true", help="also assemble a portable folder")
    parser.add_argument("--skip-verify", action="store_true", help="skip the clean-venv check")
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="portable only: skip the runtime dependencies (structure check, no download)",
    )
    parser.add_argument("--out", type=Path, default=None, help="portable bundle output directory")
    parser.add_argument(
        "--python-dir",
        type=Path,
        default=None,
        help="relocatable CPython to embed (default: sys.base_prefix)",
    )
    args = parser.parse_args(argv)

    wheel = build_artifacts()
    if not args.skip_verify and not verify_wheel(wheel):
        return 1
    if args.portable:
        bundle = assemble_portable(
            wheel, args.out, skip_deps=args.skip_deps, python_dir=args.python_dir
        )
        image = None if args.skip_deps else TOOL_ROOT / "build/tmp/tiny.png"
        if not verify_relocated(bundle, image):
            return 1
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
