#!/bin/bash
# Build the PyPI distribution `bantamtrain` (import name `bantamtrain`, command `bantam`)
# from the project directory: pyproject.toml lives one level up and points its readme and
# license at packaging/README.md and packaging/LICENSE.
#   packaging/release.sh            -> sdist + wheel in a temp dir ($BUILD_DIR/dist if set),
#                                      twine check --strict, README/LICENSE checked in both
#   TWINE_PASSWORD=<token> packaging/release.sh upload
#   BUILD_ARGS=--no-isolation packaging/release.sh   (offline: use the installed setuptools>=77)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
OUT="${BUILD_DIR:-$(mktemp -d)}"
rm -rf "$OUT/dist"                     # never upload stale files from a reused BUILD_DIR
had_egg=0
if [ -e "$ROOT/bantamtrain.egg-info" ]; then had_egg=1; fi
# no --sdist/--wheel: the wheel is built from the sdist, which proves the sdist is complete
python3 -m build ${BUILD_ARGS:-} --outdir "$OUT/dist" "$ROOT"
if [ "$had_egg" = 0 ]; then rm -rf "$ROOT/bantamtrain.egg-info"; fi
python3 -m twine check --strict "$OUT"/dist/*
python3 - "$OUT/dist" <<'EOF'
import glob, sys, tarfile, zipfile
d = sys.argv[1]
whl, = glob.glob(f"{d}/*.whl")
sdist, = glob.glob(f"{d}/*.tar.gz")
with zipfile.ZipFile(whl) as z:
    names = z.namelist()
    meta = z.read(next(n for n in names if n.endswith(".dist-info/METADATA"))).decode()
assert any(n.endswith("licenses/packaging/LICENSE") for n in names), "LICENSE not in the wheel"
head, _, body = meta.partition("\n\n")
assert "License-File: packaging/LICENSE" in head and len(body.strip()) > 100, \
    "wheel METADATA has no README body or License-File"
with tarfile.open(sdist) as t:
    names = t.getnames()
for f in ("packaging/README.md", "packaging/LICENSE", "pyproject.toml"):
    assert any(n.endswith("/" + f) for n in names), f"{f} not in the sdist"
print("README and LICENSE present in", whl.rsplit("/", 1)[-1], "and", sdist.rsplit("/", 1)[-1])
EOF
if [ "${1:-}" = "upload" ]; then
  TWINE_USERNAME=__token__ python3 -m twine upload --non-interactive "$OUT"/dist/*
fi
echo "built in $OUT/dist"
