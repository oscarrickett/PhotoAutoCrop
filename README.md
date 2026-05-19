# PhotoAutoCrop

Cross-platform tool that auto-crops and straightens scanned photos, then opens an in-browser review list where you can quickly tweak each one before delivery. Built for the Henley Scan workflow but works for any folder of JPGs.

## What it does

- Reads JPGs from a folder you point at.
- For each photo: detects the print's edges against the mat, straightens the rotation, and crops it tight (with a 2.5% inward safety margin so no mat sliver creeps in).
- Auto Tone (per-channel histogram stretch) is applied by default to fix dull colours and age-cast; you can disable per photo.
- Writes the user-approved results into a single `cropped/` sibling folder. The originals are never touched.
- Sidecar `_manifest.json` records everything: detected corners, crop box, rotation, decision, per-photo settings.

## Quick start (packaged build, no Python required)

Grab the latest release for your OS:

- **Windows** — `PhotoAutoCrop.exe`. Double-click to launch. Your default browser opens to the folder picker.
- **macOS** — `PhotoAutoCrop.app`. Double-click to launch. First-run only: right-click → Open to bypass Gatekeeper (the app is unsigned; signing requires a $99/yr Apple Developer ID).

The app starts a tiny local HTTP server on `127.0.0.1:8765` and points your browser at it. Close the window/terminal to quit.

You can also drag a folder onto the executable and it'll open straight into that folder.

## Running from source

You need Python 3.10+.

### Windows
```powershell
cd photo-autocrop
python -m venv .venv
.venv\Scripts\pip install -e .
.venv\Scripts\python -m photo_autocrop.cli review "C:\path\to\your\scans"
```

### macOS / Linux
```bash
cd photo-autocrop
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m photo_autocrop.cli review "/path/to/your/scans"
```

The browser opens automatically at `http://127.0.0.1:8765`. Pass `--port N` to use a different port; pass `--no-browser` if you don't want the browser opened for you.

CLI commands:
```
photo-autocrop run <folder>                # batch-process only (CLI, no UI)
photo-autocrop review <folder>             # open the review UI (folder optional)
photo-autocrop process-and-review <folder> # both
```

## Building a standalone executable

Bundles use PyInstaller. The bundle can only target the OS it's built on — so build on Windows for Windows, and on macOS for macOS.

```bash
# Set up the dev env first (see above), then:
python build.py             # single-file build
python build.py --onedir    # folder build (faster startup)
```

Output goes to `dist/`:
- Windows: `dist/PhotoAutoCrop.exe`
- macOS:   `dist/PhotoAutoCrop.app`

First-time PyInstaller install is automatic if it isn't already in the venv.

### macOS code signing

The default `--windowed` build is unsigned, so Gatekeeper will warn the first time a user opens it ("can't be opened because Apple cannot check it for malicious software"). Workaround:

- Right-click the .app → Open → confirm. After the first open it works normally.
- Or, with a paid Apple Developer ID, sign + notarise via `codesign` + `xcrun notarytool`.

### Windows code signing

Optional. Without it, SmartScreen may show a warning on first run that the user dismisses with "More info → Run anyway". For a clean install, use an EV code-signing certificate.

## Editor keyboard shortcuts

| Key | Action |
|---|---|
| `S` or `Enter` | Save the current crop |
| `Esc` | Back to the list without saving |
| `←` / `→` | Rotate ±0.5° (hold `Shift` for ±0.1°) |

## Folder layout

```
<your-folder>/
  IMG_1234.jpg          # your original scans, untouched
  ...
  cropped/              # auto-created, holds APPROVED crops only
    IMG_1234.jpg
  .cache/               # hidden, holds preview JPEGs + thumbnails
  _manifest.json        # crop boxes, rotations, decisions
```

Files in `cropped/` are what you deliver to the customer. Anything not in `cropped/` either wasn't reviewed yet, or you decided not to deliver it.

## Known limitations

- Auto-detection of the photo on a grey mat works best when the photo is rectangular and has a visible edge. Densely-textured mats or photos that nearly fill the frame may need a manual crop in the editor.
- Only JPG input is supported.
- One subject per scan. If your scanner captures multiple prints in one shot, they'll be detected as one.
- No perspective correction — assumes the camera/scanner is square to the mat. In-plane rotation only.
