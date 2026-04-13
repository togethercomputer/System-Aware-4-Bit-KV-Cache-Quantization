# System-Aware 4-Bit KV Cache Quantization

Official companion repository for the paper **System-Aware 4-Bit KV Cache Quantization** (Together; citation and arXiv/DOI links will be added at publication time).

## What this repo contains

- **Documentation** under [docs/](docs/) for environment setup, **BDR** inference, **accuracy** evaluation, **KV calibration**, and **throughput** benchmarking.  
- **Git submodules** pointing at our SGLang forks (same remote, different branches):  
  - [third_party/sglang-fast-rotation](third_party/sglang-fast-rotation) — fused INT4 KV + BDR; use for **speed** (`bench_serving`).  
  - [third_party/sglang-kmeans](third_party/sglang-kmeans) — INT4 + dump + k-means centroids + BDR flags; use for the full **accuracy** matrix.  
- **Tools:** [tools/fit_kv_centroids.py](tools/fit_kv_centroids.py) turns dumped KV tensors into centroid files expected by `SGLANG_KV_CENTROIDS_PATH`.  
- **Scripts:** [scripts/](scripts/) helpers for submodules, eval env printing, and benchmark templates.

Pinned commits are recorded in [SUBMODULE_VERSIONS.md](SUBMODULE_VERSIONS.md).

## BDR is SGLang-based

**Block-diagonal rotation (BDR)** before 4-bit KV quantization is **not** a separate inference engine. It is implemented inside our **SGLang fork** as a **block Hadamard** transform on keys (and optionally values), combined with INT4 KV cache kernels and matching decode-side transforms so attention remains correct. See [docs/02-bdr-inference.md](docs/02-bdr-inference.md) and [third_party/sglang-fast-rotation/EVAL_NOTES.md](third_party/sglang-fast-rotation/EVAL_NOTES.md).

## Quick start: run BDR with rotation on **K** only

From [third_party/sglang-fast-rotation/python](third_party/sglang-fast-rotation/python), after `pip install -e ".[all]"` and `pip install fast_hadamard_transform`:

```bash
export HADAMARD=1
export ROTATE_V=0
export HADAMARD_ORDER=16

python -m sglang.launch_server \
  --model-path "Qwen/Qwen3-8B" \
  --port 30000 \
  --kv-cache-dtype int4
```

- **BF16 KV:** `HADAMARD=0`, `--kv-cache-dtype auto`.  
- **INT4 without rotation:** `HADAMARD=0`, `--kv-cache-dtype int4`.  
- **Rotate V as well:** set `ROTATE_V=1` (ablation / alternate setting).

## Accuracy evaluation (OpenAI simple-evals)

We report accuracy using **[OpenAI simple-evals](https://github.com/openai/simple-evals)** against SGLang’s **OpenAI-compatible HTTP API**. Build the **sglang-kmeans** submodule, launch `sglang.launch_server`, set `OPENAI_BASE_URL` to `http://<host>:<port>/v1`, then run simple-evals. Full matrix (BF16, INT4, BDR, k-means, k-means + rotation): [docs/03-evaluation-matrix.md](docs/03-evaluation-matrix.md).

Helper: [scripts/run_eval_matrix.sh](scripts/run_eval_matrix.sh).

## Throughput evaluation (SGLang benchmarks)

Speed uses the **fast-rotation** fork and SGLang’s official tools, primarily **`python -m sglang.bench_serving`**. See [docs/05-throughput-benchmarking.md](docs/05-throughput-benchmarking.md) and the upstream guide: [Benchmark and Profiling](https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/benchmark_and_profiling.md).

Helper: [scripts/run_bench_serving_example.sh](scripts/run_bench_serving_example.sh).

## Representative results (paper)

Fill this table from the camera-ready paper (models and benchmarks must match the paper text). Until release, you can point readers to the PDF tables.

| Model | Method | Benchmark | Score |
|-------|--------|-----------|-------|
| — | BF16 | — | — |
| — | INT4 | — | — |
| — | BDR (K-only) | — | — |
| — | K-means + INT4 | — | — |
| — | K-means + BDR | — | — |

## Full evaluation reproduction

Large raw logs, exact simple-evals command lines, and frozen conda environments may live in a **separate artifact** (for example Hugging Face dataset, Zenodo archive, or an internal lab URL). **Add the canonical link here when it is published:**

- **Full reproduction bundle:** *TBD — add URL*

Submodule SHAs to match the paper: [SUBMODULE_VERSIONS.md](SUBMODULE_VERSIONS.md).

## Clone

```bash
git clone --recurse-submodules https://github.com/togethercomputer/System-Aware-4-Bit-KV-Cache-Quantization.git
cd System-Aware-4-Bit-KV-Cache-Quantization
```

If `tore-eval` fails to initialize under `sglang-fast-rotation` (private submodule), you can still follow this README using **simple-evals** only; see [SUBMODULE_VERSIONS.md](SUBMODULE_VERSIONS.md).

## License

See [LICENSE](LICENSE).
