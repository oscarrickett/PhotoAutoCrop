"""Entry point used by PyInstaller bundles.

Starts the local FastAPI server, opens the user's default browser to the
folder picker, and stays alive until the user closes the process.
Optionally takes a folder path as the first positional argument — used
when the user drags a folder onto the executable.
"""
from __future__ import annotations

import os
import sys
import webbrowser
from multiprocessing import freeze_support
from pathlib import Path


def main() -> None:
    # CRITICAL on Windows when PyInstaller bundles use multiprocessing.
    # Without this, each spawned worker re-enters main() recursively and
    # the app deadlocks.
    freeze_support()

    # Optional first positional argument: a folder to open straight into.
    initial_folder: Path | None = None
    if len(sys.argv) > 1:
        cand = Path(sys.argv[1]).expanduser().resolve()
        if cand.is_dir():
            initial_folder = cand

    # Lazy imports so PyInstaller-bootloader output stays clean for users
    # who hit a missing-folder error before the heavy deps load.
    import uvicorn
    # Absolute import — relative imports fail when PyInstaller runs this
    # as a standalone script in the bundle.
    from photo_autocrop import server

    if initial_folder is not None:
        server.set_input_folder(initial_folder)

    port = int(os.environ.get("PHOTO_AUTOCROP_PORT", "8765"))
    host = "127.0.0.1"
    url = f"http://{host}:{port}"

    try:
        webbrowser.open(url)
    except Exception:
        pass

    print(f"PhotoAutoCrop running at {url}")
    print("Close this window to quit.")
    uvicorn.run(server.app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
