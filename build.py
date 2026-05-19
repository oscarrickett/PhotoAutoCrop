#!/usr/bin/env python3
"""Build a standalone PhotoAutoCrop executable for the current OS.

Usage:
    python build.py            # Windows .exe or Mac .app, single-file
    python build.py --onedir   # Faster startup, ships as a folder

PyInstaller can only target the OS it's running on. Run this script on
Windows to produce the Windows build; run it on macOS to produce the
Mac build.
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "PhotoAutoCrop"


def ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller not found in this Python — installing…")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pyinstaller>=6.0"]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PhotoAutoCrop bundle")
    parser.add_argument("--onedir", action="store_true", help="Folder bundle instead of single file")
    parser.add_argument("--console", action="store_true", help="Keep a console window (debug)")
    args = parser.parse_args()

    ensure_pyinstaller()

    web_dir = ROOT / "web"
    if not web_dir.is_dir():
        sys.exit(f"web/ folder not found at {web_dir}")

    entry = ROOT / "src" / "photo_autocrop" / "launcher.py"
    if not entry.exists():
        sys.exit(f"launcher entry not found at {entry}")

    # PyInstaller's --add-data uses ';' on Windows, ':' elsewhere.
    sep = ";" if os.name == "nt" else ":"

    is_mac = platform.system() == "Darwin"

    cmd: list[str] = [
        sys.executable, "-m", "PyInstaller",
        "--onedir" if args.onedir else "--onefile",
        "--name", NAME,
        "--add-data", f"{web_dir}{sep}web",
        # FastAPI / Uvicorn / Starlette have dynamic imports PyInstaller's
        # default analysis misses.
        "--collect-submodules", "uvicorn",
        "--collect-submodules", "fastapi",
        "--collect-submodules", "starlette",
        "--collect-submodules", "anyio",
        "--collect-submodules", "photo_autocrop",
        "--noconfirm",
        "--clean",
    ]

    # On Mac default to --windowed (proper .app, no terminal). On Windows
    # default to console so users can see crashes; pass --console=False
    # later if/when we want it fully silent.
    if is_mac and not args.console:
        cmd.append("--windowed")
    if not is_mac and not args.console:
        # On Windows --noconsole hides the console window. Comment out to
        # leave a console visible.
        # cmd.append("--noconsole")
        pass

    # Icon (optional). Pick the right format per OS if a file is present.
    icon_candidates = []
    if is_mac:
        icon_candidates = [web_dir / "logo.icns"]
    elif os.name == "nt":
        icon_candidates = [web_dir / "logo.ico"]
    for ic in icon_candidates:
        if ic.exists():
            cmd[2:2] = ["--icon", str(ic)]
            break

    cmd.append(str(entry))

    print("PyInstaller command:")
    print("  " + " ".join(cmd))
    print()
    subprocess.check_call(cmd, cwd=ROOT)

    print()
    print(f"Done. Output in: {ROOT / 'dist'}")
    if is_mac:
        print(f"  - macOS bundle:  dist/{NAME}.app")
        print(f"  - To distribute: right-click → Compress to get a .zip")
        print()
        print("  First time a user opens it macOS will block it (Gatekeeper).")
        print("  Have them right-click → Open the first time to bypass the warning,")
        print("  or sign it with a paid Apple Developer ID for a clean install.")
    elif os.name == "nt":
        print(f"  - Windows exe:   dist/{NAME}.exe")
    else:
        print(f"  - Linux binary:  dist/{NAME}")


if __name__ == "__main__":
    main()
