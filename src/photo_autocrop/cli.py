from __future__ import annotations

import sys
import time
import webbrowser
from pathlib import Path

import typer

from .pipeline import run_batch
from .schemas import ManifestEntry


app = typer.Typer(add_completion=False, no_args_is_help=True, help="Auto-crop and review copy-stand photos.")


def _print_progress(i: int, total: int, entry: ManifestEntry) -> None:
    bar = f"[{i:>4}/{total}]"
    print(f"{bar} {entry.decision:<13} score={entry.score:>3}  {entry.filename}", flush=True)


@app.command()
def run(
    folder: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, help="Folder of JPGs to process."),
    workers: int = typer.Option(0, "--workers", "-w", help="Worker processes. 0 = auto (cpu-1)."),
) -> None:
    """Process all JPGs in FOLDER and write to approved/needs-review/rejected subfolders."""
    start = time.time()
    w = workers if workers > 0 else None
    manifest = run_batch(folder, workers=w, progress_callback=_print_progress)
    elapsed = time.time() - start
    counts: dict[str, int] = {}
    for e in manifest.entries:
        counts[e.decision] = counts.get(e.decision, 0) + 1
    print()
    print(f"Done in {elapsed:.1f}s. {sum(counts.values())} files.")
    for k in ("approved", "needs-review", "rejected", "no-detection"):
        if counts.get(k):
            print(f"  {k:<13} {counts[k]}")


@app.command()
def review(
    folder: Path = typer.Argument(None, file_okay=False, dir_okay=True, help="Folder to review. Omit to pick one in the browser."),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    no_browser: bool = typer.Option(False, "--no-browser"),
) -> None:
    """Launch the in-browser review UI. If no folder is provided, the
    browser opens a folder-picker first.
    """
    import uvicorn

    from . import server

    if folder is not None:
        server.set_input_folder(folder.resolve())
    url = f"http://{host}:{port}"
    if not no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    print(f"Review server: {url}")
    uvicorn.run(server.app, host=host, port=port, log_level="warning")


@app.command()
def process_and_review(
    folder: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    workers: int = typer.Option(0, "--workers", "-w"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    no_browser: bool = typer.Option(False, "--no-browser"),
) -> None:
    """Run the batch then immediately launch the review UI."""
    run(folder, workers)
    review(folder, host=host, port=port, no_browser=no_browser)


if __name__ == "__main__":
    app()
