#!/usr/bin/env bash
# Template: start sglang-fast-rotation server with a KV config, then run bench_serving.
# See docs/05-throughput-benchmarking.md and upstream:
# https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/benchmark_and_profiling.md
#
# Usage:
#   # terminal 1
#   KV_MODE=bf16 ./scripts/run_bench_serving_example.sh server
#   KV_MODE=bdr ./scripts/run_bench_serving_example.sh server
#   # terminal 2
#   ./scripts/run_bench_serving_example.sh client
#
# Env:
#   MODEL_PATH                (default meta-llama/Meta-Llama-3.1-8B-Instruct)
#   PORT                      (default 30000)
#   KV_MODE                   bf16 | int4 | bdr  (default bf16)
#   PREFILL_ATTENTION_BACKEND (default fa3)
#   DECODE_ATTENTION_BACKEND  (default triton)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FR="$ROOT/third_party/sglang-fast-rotation/python"
MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
PORT="${PORT:-30000}"
KV_MODE="${KV_MODE:-bf16}"
PREFILL_ATTENTION_BACKEND="${PREFILL_ATTENTION_BACKEND:-fa3}"
DECODE_ATTENTION_BACKEND="${DECODE_ATTENTION_BACKEND:-triton}"
ROLE="${1:-}"

if [[ ! -d "$FR" ]]; then
  echo "Missing $FR — init submodules" >&2
  exit 1
fi

if [[ "$ROLE" == "server" ]]; then
  unset HADAMARD ROTATE_V HADAMARD_ORDER || true
  case "$KV_MODE" in
    bf16)
      DTYPE="auto"
      export HADAMARD=0
      export ROTATE_V=0
      ;;
    int4)
      DTYPE="int4"
      export HADAMARD=0
      export ROTATE_V=0
      ;;
    bdr)
      DTYPE="int4"
      export HADAMARD=1
      export ROTATE_V=0
      export HADAMARD_ORDER="${HADAMARD_ORDER:-16}"
      ;;
    *)
      echo "KV_MODE must be bf16|int4|bdr" >&2
      exit 1
      ;;
  esac
  echo "Starting server KV_MODE=$KV_MODE dtype=$DTYPE on port $PORT (prefill=$PREFILL_ATTENTION_BACKEND decode=$DECODE_ATTENTION_BACKEND)"
  cd "$FR"
  exec python -m sglang.launch_server \
    --prefill-attention-backend "$PREFILL_ATTENTION_BACKEND" \
    --decode-attention-backend "$DECODE_ATTENTION_BACKEND" \
    --model-path "$MODEL_PATH" --port "$PORT" --kv-cache-dtype "$DTYPE"
fi

if [[ "$ROLE" == "client" ]]; then
  cd "$FR"
  exec python -m sglang.bench_serving \
    --backend sglang \
    --model "$MODEL_PATH" \
    --num-prompts 80 \
    --max-concurrency 16 \
    --random-input-len 256 \
    --random-output-len 32 \
    --dataset-name random
fi

echo "Usage: $0 server|client" >&2
echo "  KV_MODE=bf16|int4|bdr $0 server" >&2
exit 1
