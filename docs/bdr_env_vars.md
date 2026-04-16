# BDR environment variable reference

Set these in the shell **before** `python -m sglang.launch_server` on **`third_party/sglang-fast-rotation`** (read in `memory_pool.py`). Always combine with `--kv-cache-dtype`.

| Variable | Values | Role |
|----------|--------|------|
| **`HADAMARD`** | `0` / `1` | `0` = no rotation; `1` = block Hadamard on **K** before INT4 KV write, with matching **Q** correction at decode. |
| **`ROTATE_V`** | `0` / `1` | `0` = rotate K only (default BDR); `1` = also rotate **V** and apply inverse rotation to the attention output. |
| **`HADAMARD_ORDER`** | integer | Block size (e.g. `16`); **must divide head dim**; ignored when `HADAMARD=0`. |
| **`--kv-cache-dtype`** | `auto` / `int4` | `auto` = BF16 KV baseline; `int4` = 4-bit KV (with or without `HADAMARD=1` for BDR). |

## Mode matrix

| Mode | `HADAMARD` | `ROTATE_V` | `HADAMARD_ORDER` | `--kv-cache-dtype` |
|------|------------|------------|------------------|---------------------|
| BF16 KV (baseline) | `0` | `0` | unset | `auto` |
| INT4 KV (no rotation) | `0` | `0` | unset | `int4` |
| INT4 + BDR (K only) | `1` | `0` | e.g. `16` | `int4` |
| INT4 + BDR (K + V) | `1` | `1` | e.g. `16` | `int4` |

`HADAMARD_ORDER` must divide the model's head dim. For Qwen3 models with 128-dim heads, `16` (or any power-of-two divisor of 128) works.

## BDR + K+V example

```bash
cd third_party/sglang-fast-rotation/python
HADAMARD=1 ROTATE_V=1 HADAMARD_ORDER=16 python -m sglang.launch_server \
  --prefill-attention-backend fa3 \
  --decode-attention-backend triton \
  --model-path "Qwen/Qwen3-8B" \
  --port 30000 \
  --kv-cache-dtype int4
```
