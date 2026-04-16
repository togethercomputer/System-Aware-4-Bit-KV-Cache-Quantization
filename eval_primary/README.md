# Primary evaluation — accuracy (BF16, INT4, BDR)

This folder holds **accuracy** logs and summary tables for the **primary** paper track: **BF16**, **INT4**, and **BDR** on **`third_party/sglang-fast-rotation`** only.

**Default checkpoint:** `Qwen/Qwen3-4B-Thinking-2507` (override with `MODEL_PATH=…`). **Default benchmark:** **GPQA** via simple-evals `--eval gpqa` (see main README for registering the model in `simple_evals.py`).

**Throughput** for the same track lives under [../eval_speed/README.md](../eval_speed/README.md) and uses **`Qwen/Qwen3-8B`**.

**Canonical instructions:** [../README.md](../README.md#primary-accuracy-and-throughput) — GPQA client [Prepare](../README.md#prepare) under [Accuracy (primary)](../README.md#accuracy-primary).  
**Script:** [../scripts/run_primary_eval_matrix.sh](../scripts/run_primary_eval_matrix.sh)  
**Client:** [openai/simple-evals](https://github.com/openai/simple-evals) — included as submodule at [`third_party/simple-evals`](../third_party/simple-evals/); no separate clone needed

## Workflow

1. `cd third_party/sglang-fast-rotation/python` and install per [../README.md#how-to-run-bdr](../README.md#how-to-run-bdr).
2. From repo root: `./scripts/run_primary_eval_matrix.sh bf16` (or `int4`, `bdr`, `bdr_kv`) — start the printed server, then run simple-evals with `OPENAI_BASE_URL` pointing at `/v1` and **`--eval gpqa`** as in the main [README.md](../README.md#accuracy-primary).
3. Store outputs under [results/](results/) and mirror headline scores in the main [README.md](../README.md).
