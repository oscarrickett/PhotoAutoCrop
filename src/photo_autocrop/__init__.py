from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("photo-autocrop")
except PackageNotFoundError:
    # Not installed (e.g. running from source without `pip install -e .`).
    __version__ = "0.0.0+source"

