# Speed evaluation (throughput / latency)

Part of **primary evaluation** (BF16 / INT4 / BDR): server = **`third_party/sglang-fast-rotation`** only. For primary **accuracy** logs, see [../eval_primary/README.md](../eval_primary/README.md).

**Default checkpoint for speed:** `Qwen/Qwen3-8B` (`MODEL_PATH` on [../scripts/run_genai_bench_example.sh](../scripts/run_genai_bench_example.sh)). Primary **accuracy** uses `Qwen/Qwen3-4B-Thinking-2507` + GPQA (see main README).

This folder is the **hub for throughput experiments**: conventions and where to store **genai-bench** experiment folders / exports.

**Canonical instructions:** [../README.md#throughput-and-latency-primary](../README.md#throughput-and-latency-primary) (install client: [Prepare (genai-bench)](../README.md#prepare-genai-bench))  
**Helper script:** [../scripts/run_genai_bench_example.sh](../scripts/run_genai_bench_example.sh)  
**Tool:** [genai-bench](https://github.com/sgl-project/genai-bench) — [docs](https://docs.sglang.ai/genai-bench/getting-started/)

## Server build

Use **[third_party/sglang-fast-rotation](../third_party/sglang-fast-rotation)** (fused INT4 KV + BDR). Install from `third_party/sglang-fast-rotation/python` per [../README.md#how-to-run-bdr](../README.md#how-to-run-bdr). Use **MHA** models and **Flash Attention prefill + Triton decode** (see [../README.md#server-requirements](../README.md#server-requirements)).

## Client (genai-bench)

Install **`pip install genai-bench`** (see main README [Prepare (genai-bench)](../README.md#prepare-genai-bench)). Then run a benchmark against the running server; see [Run benchmark](https://docs.sglang.ai/genai-bench/user-guide/run-benchmark/) for flags, traffic scenarios, and multi-worker load.

**Terminal 1 — server** (example: BDR, K-only):

```bash
cd third_party/sglang-fast-rotation/python
export HADAMARD=1
export ROTATE_V=0
export HADAMARD_ORDER=16
python -m sglang.launch_server \
  --prefill-attention-backend fa3 \
  --decode-attention-backend triton \
  --model-path "Qwen/Qwen3-8B" \
  --port 30000 \
  --kv-cache-dtype int4
```

**Terminal 2 — client** (same shape as the main README example; align `--api-base` / port with the server):

```bash
genai-bench benchmark --api-backend sglang \
  --api-base "http://127.0.0.1:30000" \
  --api-key "dummy" \
  --api-model-name "Qwen/Qwen3-8B" \
  --model-tokenizer "Qwen/Qwen3-8B" \
  --task text-to-text \
  --traffic-scenario "D(256,32)" \
  --num-concurrency 16 \
  --max-time-per-run 5 \
  --max-requests-per-run 200 \
  --server-engine "SGLang" \
  --server-gpu-type "local" \
  --server-version "custom" \
  --server-gpu-count 1
```

Sweep **BF16** / **INT4** / **BDR** by changing server env and `--kv-cache-dtype` only; keep genai-bench flags fixed for comparability.

## Results

Store experiment folders, logs, and summarized tables under **[results/](results/)**. Use the table template there; copy a one-row summary into the main [README.md](../README.md) when you publish numbers.
