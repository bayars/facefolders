#!/usr/bin/env bash
# Usage: ./run.sh /path/to/photos /path/to/output [extra sort_faces.py args]
set -euo pipefail

PHOTOS=${1:?usage: run.sh PHOTOS_DIR OUTPUT_DIR [args...]}
OUT=${2:?usage: run.sh PHOTOS_DIR OUTPUT_DIR [args...]}
shift 2
mkdir -p "$OUT"

GPU_FLAG=(--gpus all)
[[ "${NO_GPU:-}" == 1 ]] && GPU_FLAG=()

docker build -t face-sorter "$(dirname "$(realpath "$0")")"
docker run --rm "${GPU_FLAG[@]}" --user "$(id -u):$(id -g)" \
  -v "$(realpath "$PHOTOS")":/photos:ro \
  -v "$(realpath "$OUT")":/out \
  face-sorter "$@"
