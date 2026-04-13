#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
git submodule update --init --recursive || {
  echo "Warning: recursive submodule init failed (often private tore-eval)." >&2
  echo "Run: git submodule update --init third_party/sglang-fast-rotation third_party/sglang-kmeans" >&2
  git submodule update --init third_party/sglang-fast-rotation third_party/sglang-kmeans
}
