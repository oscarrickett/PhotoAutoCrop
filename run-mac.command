#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "Virtual environment not found at .venv/"
  echo "Set up once with:"
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/pip install -e ."
  exit 1
fi

if [ -z "$1" ]; then
  echo "Usage: ./run-mac.command <folder>"
  echo "Drop a folder of JPGs onto this script, or pass a folder path as the first argument."
  exit 1
fi

.venv/bin/python -m photo_autocrop.cli process-and-review "$1"
