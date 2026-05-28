"""Entry point used by PyInstaller bundles.

Starts the local FastAPI server, waits for it to bind, opens the user's
default browser to the folder picker, and stays alive until the user
quits (Quit button in the UI, Ctrl-C, or closes the window).
Optionally takes a folder path as the first positional argument — used
when the user drags a folder onto the executable.
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from multiprocessing import freeze_support
from pathlib import Path


def _wait_for_port(host: str, port: int, timeout: float = 15.0) -> bool:
    """Block until `host:port` accepts a TCP connection, or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


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

    port = int(os.environ.get("PHOTO_AUTOCROP_PORT", "8765"))
    host = "127.0.0.1"
    url = f"http://{host}:{port}"

    print("Starting PhotoAutoCrop…", flush=True)

    # Pre-warm the heavy imports so the FIRST request the browser makes
    # doesn't pay the 5-10s of import time. Done before uvicorn binds so
    # the splash page in the user's browser stays in sync.
    import uvicorn  # noqa: F401
    import cv2  # noqa: F401
    import numpy  # noqa: F401
    # Absolute import — relative imports fail when PyInstaller runs this
    # as a standalone script in the bundle.
    from photo_autocrop import server

    if initial_folder is not None:
        server.set_input_folder(initial_folder)

    config = uvicorn.Config(server.app, host=host, port=port, log_level="warning")
    uv_server = uvicorn.Server(config)

    def _serve() -> None:
        uv_server.run()

    server_thread = threading.Thread(target=_serve, daemon=True)
    server_thread.start()

    if _wait_for_port(host, port):
        print(f"PhotoAutoCrop running at {url}", flush=True)
        print("Use the Quit button in the browser, or close this window.", flush=True)
        try:
            webbrowser.open(url)
        except Exception:
            pass
    else:
        print(f"Server failed to start on {url}", flush=True)
        sys.exit(1)

    # Block the main thread until the server stops (Quit button calls
    # os._exit, Ctrl-C raises KeyboardInterrupt).
    try:
        while server_thread.is_alive():
            server_thread.join(timeout=1.0)
    except KeyboardInterrupt:
        print("Stopping…", flush=True)
        uv_server.should_exit = True
        server_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
