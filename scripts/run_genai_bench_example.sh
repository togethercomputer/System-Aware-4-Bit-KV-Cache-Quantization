#!/usr/bin/env bash
# Start sglang-fast-rotation with a KV config, then run [genai-bench](https://github.com/sgl-project/genai-bench)
# against the OpenAI-compatible HTTP API. Prerequisite: pip install genai-bench (see main README).
#
# Usage:
#   # terminal 1
#   KV_MODE=bf16 ./scripts/run_genai_bench_example.sh server
#   KV_MODE=bdr ./scripts/run_genai_bench_example.sh server
#   # terminal 2 (after pip install genai-bench)
#   ./scripts/run_genai_bench_example.sh client
#
# Env:
#   MODEL_PATH                (default Qwen/Qwen3-8B — throughput track)
#   PORT                      (default 30000)
#   KV_MODE                   bf16 | int4 | bdr  (default bf16)
#   PREFILL_ATTENTION_BACKEND (default fa3)
#   DECODE_ATTENTION_BACKEND  (default triton)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FR="$ROOT/third_party/sglang-fast-rotation/python"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
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
  if ! command -v genai-bench >/dev/null 2>&1; then
    echo "genai-bench not found. Install: pip install genai-bench" >&2
    echo "See README.md#prepare-genai-bench and https://docs.sglang.ai/genai-bench/getting-started/installation/" >&2
    exit 1
  fi
  exec genai-bench benchmark \
    --api-backend sglang \
    --api-base "http://127.0.0.1:${PORT}" \
    --api-key "${OPENAI_API_KEY:-dummy}" \
    --api-model-name "$MODEL_PATH" \
    --model-tokenizer "$MODEL_PATH" \
    --task text-to-text \
    --traffic-scenario "D(256,32)" \
    --num-concurrency 16 \
    --max-time-per-run 5 \
    --max-requests-per-run 200 \
    --server-engine "SGLang" \
    --server-gpu-type "local" \
    --server-version "custom" \
    --server-gpu-count 1
fi

echo "Usage: $0 server|client" >&2
echo "  KV_MODE=bf16|int4|bdr $0 server" >&2
exit 1
