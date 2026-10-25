#!/bin/bash
# Build the PyPI distribution `bantamtrain` from the library in ../zoomtrain.
#   packaging/release.sh            -> dist/ (sdist + wheel), twine check
#   TWINE_PASSWORD=<token> packaging/release.sh upload
# The import name on PyPI is `bantamtrain` (the name `bantam` was already taken).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="${BUILD_DIR:-$(mktemp -d)}"
cp "$HERE"/pyproject.toml "$HERE"/LICENSE "$HERE"/README.md "$BUILD"/
cp -r "$HERE"/../zoomtrain "$BUILD"/bantamtrain
find "$BUILD"/bantamtrain -name __pycache__ -prune -exec rm -rf {} +
find "$BUILD"/bantamtrain -name "*.py" -exec sed -i 's/zoomtrain/bantamtrain/g' {} +
cd "$BUILD" && python3 -m build --sdist --wheel && python3 -m twine check dist/*
if [ "$1" = "upload" ]; then
  TWINE_USERNAME=__token__ python3 -m twine upload --non-interactive dist/*
fi
echo "built in $BUILD/dist"
