#!/usr/bin/env python3
"""Build and verify distributable artifacts for ``pinn-starlight``.

Two products:

* ``release`` (default) - wheel + sdist in ``build/dist/``, verified by
  installing the wheel into a throwaway venv *without* dependencies and
  checking that the CLI degrades gracefully instead of crashing;
* ``--portable`` - a self-contained folder (embedded venv + launchers) that a
  user can unzip and run.  This downloads the scientific stack, so it is opt-in.

Usage
-----
    python tools/build_release.py
    python tools/build_release.py --skip-verify
    python tools/build_release.py --portable

Exit codes: 0 success, 1 build/verify failure.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parents[1]
DIST = TOOL_ROOT / "build/dist"
WORK = TOOL_ROOT / "build/tmp/release"
DEFAULT_INDEX = "https://mirrors.ustc.edu.cn/pypi/simple"


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


def build_artifacts(clean: bool = True) -> Path:
    """Build wheel + sdist into ``build/dist`` and return the wheel path."""
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required: install it or run `uv build` manually")
    if clean and WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)

    print("[1/3] building wheel + sdist")
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
    print("[2/3] verifying the wheel in a clean venv")
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
        status = "ok" if passed else "FAIL"
        print(f"  {name:<10} exit={result.returncode} {status}")
        if not passed:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
    return ok


def assemble_portable(wheel: Path, out: Path | None = None) -> Path:
    """Build a self-contained folder: venv + wheel + launchers + README."""
    package_version = wheel.name.split("-")[1]
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    target = out or (
        TOOL_ROOT
        / "build/dist"
        / f"pinn-starlight-{sys.platform}-py{python_version.replace('.', '')}"
    )
    print(f"[3/3] assembling portable bundle -> {target}")
    if target.exists():
        shutil.rmtree(target)
    (target / "wheel").mkdir(parents=True)
    shutil.copy2(wheel, target / "wheel" / wheel.name)

    venv = target / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    install = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--quiet",
            "--index-url",
            DEFAULT_INDEX,
            f"{wheel}[estimator,cpu-torch,photo]",
        ],
        text=True,
        capture_output=True,
    )
    if install.returncode != 0:
        print(install.stdout)
        print(install.stderr, file=sys.stderr)
        raise SystemExit("installing the portable dependencies failed")

    launcher_sh = target / "pinn-starlight.sh"
    launcher_sh.write_text(
        "#!/usr/bin/env bash\n"
        'here="$(cd "$(dirname "$0")" && pwd)"\n'
        'exec "$here/venv/bin/pinn-starlight" "$@"\n',
        encoding="utf-8",
    )
    launcher_sh.chmod(0o755)
    (target / "pinn-starlight.bat").write_text(
        '@echo off\r\n"%~dp0venv\\Scripts\\pinn-starlight.exe" %*\r\n',
        encoding="utf-8",
    )
    (target / "README.txt").write_text(
        "PINN-Starlight portable bundle\n"
        "=============================\n\n"
        f"pinn-starlight-core {package_version} on {sys.platform}, "
        f"Python {python_version}.\n\n"
        "Usage (no Python installation needed):\n"
        "  Linux / macOS : ./pinn-starlight.sh estimate --image photo.jpg --mode star\n"
        "  Windows       : pinn-starlight.bat estimate --image photo.jpg --mode star\n\n"
        "Try `pinn-starlight.sh doctor` first.  The bundle runs on CPU with demo\n"
        "budgets; it does not reproduce the paper numbers.\n",
        encoding="utf-8",
    )
    print(f"  bundle: {target}")
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build pinn-starlight release artifacts.")
    parser.add_argument("--portable", action="store_true", help="also assemble a portable folder")
    parser.add_argument("--skip-verify", action="store_true", help="skip the clean-venv check")
    parser.add_argument("--out", type=Path, default=None, help="portable bundle output directory")
    args = parser.parse_args(argv)

    wheel = build_artifacts()
    if not args.skip_verify and not verify_wheel(wheel):
        return 1
    if args.portable:
        assemble_portable(wheel, args.out)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
