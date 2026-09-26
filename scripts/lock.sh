#!/usr/bin/env sh
# Regenerates the hash-pinned lockfiles in locks/ from requirements.txt and
# requirements-dev.txt, one per CPU architecture the image is built on
# (x86_64: CI/servers, aarch64: Apple-silicon Docker). Run after changing a
# top-level pin, then rebuild the image and run the tests.
set -e
cd "$(dirname "$0")/.."
docker run --rm -v "$PWD:/src" -w /src python:3.12-slim sh -c '
  pip install -q uv
  for arch in x86_64 aarch64; do
    uv pip compile requirements.txt --python-platform ${arch}-manylinux_2_28 --python-version 3.12 \
      --generate-hashes --no-header -o locks/runtime-${arch}.txt
    uv pip compile requirements-dev.txt --python-platform ${arch}-manylinux_2_28 --python-version 3.12 \
      --generate-hashes --no-header -o locks/dev-${arch}.txt
  done'
