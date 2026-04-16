# Primary evaluation — accuracy (BF16, INT4, BDR)

This folder holds **accuracy** logs and summary tables for the **primary** paper track: **BF16**, **INT4**, and **BDR** on **`third_party/sglang-fast-rotation`** only.

**Default checkpoint:** `Qwen/Qwen3-4B-Thinking-2507` (override with `MODEL_PATH=…`). **Default benchmark:** **GPQA** via simple-evals `--eval gpqa` (see main README for registering the model in `simple_evals.py`).

**Throughput** for the same track lives under [../eval_speed/README.md](../eval_speed/README.md) and uses **`Qwen/Qwen3-8B`**.

**Canonical instructions:** [../README.md](../README.md#primary-accuracy-and-throughput) — GPQA client [Prepare](../README.md#prepare) under [Accuracy (primary)](../README.md#accuracy-primary).  
**Script:** [../scripts/run_primary_eval_matrix.sh](../scripts/run_primary_eval_matrix.sh)  
**Client:** [openai/simple-evals](https://github.com/openai/simple-evals) — included as submodule at [`third_party/simple-evals`](../third_party/simple-evals/); no separate clone needed

## Workflow

1. `cd third_party/sglang-fast-rotation/python` and install per [../README.md#how-to-run-bdr](../README.md#how-to-run-bdr).
2. Start the SGLang server in the mode you want to evaluate (`BF16`, `INT4`, or `BDR`) as described in the main [README.md](../README.md#how-to-run-bdr).
3. From repo root, run GPQA from `third_party/simple-evals` as shown in the main [README.md](../README.md#accuracy-primary). Override `OPENAI_BASE_URL` if your server is not on the default `http://127.0.0.1:30000/v1`.
4. If you want a slightly more configurable wrapper, use `SIMPLE_EVALS_MODEL=<your_registered_simple_evals_model> ./scripts/run_primary_eval_matrix.sh`.
5. Store outputs under [results/](results/) and mirror headline scores in the main [README.md](../README.md).
