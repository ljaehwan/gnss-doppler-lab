#!/usr/bin/env bash
set -euo pipefail
COMMIT=28ca29a6719475195e3aabd5930c4ed02d67190f
DEST="${1:-$PWD/.tools/gps-sdr-sim-src}"
mkdir -p "$(dirname "$DEST")"
if [[ ! -d "$DEST/.git" ]]; then git clone https://github.com/osqzss/gps-sdr-sim.git "$DEST"; fi
git -C "$DEST" fetch origin "$COMMIT"
git -C "$DEST" checkout --detach "$COMMIT"
make -C "$DEST"
printf 'Built %s/gps-sdr-sim at pinned commit %s\n' "$DEST" "$COMMIT"
