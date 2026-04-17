# Throughput benchmark metrics

Metrics are produced by [genai-bench](https://github.com/sgl-project/genai-bench) and stored in the per-run JSON files under `results/<timestamp>/`.  
Full upstream definitions: [SGLang genai-bench — Metrics Definition](https://docs.sglang.io/genai-bench/getting-started/metrics-definition/).

---

## Request timeline

Each request goes through two phases — **prefill** (prompt processing) and **decode** (token generation):

```
submit          first token                       last token
  │                  │                                 │
  ├── TTFT ──────────┤──── Output Latency ─────────────┤
  │                                                    │
  └──────────────── E2E Latency ───────────────────────┘
```

- **TTFT** (Time to First Token): `time_at_first_token − start_time`  
- **Output Latency**: time from first token to last token  
- **E2E Latency**: `end_time − start_time` = TTFT + Output Latency

---

## Metrics reported in the results tables

### Latency (request-level)

| Column | JSON field | Formula | Unit |
|--------|-----------|---------|------|
| **mean_ttft(req)** | `aggregated_metrics.stats.ttft.mean` | `time_at_first_token − start_time`, averaged over requests | ms |
| **E2E mean(req)** | `aggregated_metrics.stats.e2e_latency.mean` | `end_time − start_time`, averaged | s |
| **E2E p75(req)** | `aggregated_metrics.stats.e2e_latency.p75` | 75th-percentile E2E latency | s |
| **E2E p90(req)** | `aggregated_metrics.stats.e2e_latency.p90` | 90th-percentile E2E latency | s |

TTFT is dominated by **prefill time + queuing delay**. At high concurrency, queuing inflates TTFT even when the prefill kernel itself is fast.

### Throughput (request-level)

| Column | JSON field | Formula | Unit |
|--------|-----------|---------|------|
| **mean_input_tps(req)** | `aggregated_metrics.stats.input_throughput.mean` | `input_tokens ÷ TTFT`, averaged over requests | tok/s |
| **mean_output_tps(req)** | `aggregated_metrics.stats.output_inference_speed.mean` | `1 ÷ TPOT` = `(output_tokens − 1) ÷ output_latency`, averaged | tok/s |

Where **TPOT** (Time Per Output Token) = `(E2E − TTFT) ÷ (output_tokens − 1)`.

- `mean_input_tps(req)` approximates per-request **prefill speed**; it decreases with concurrency because TTFT grows with queuing.  
- `mean_output_tps(req)` is the token streaming speed a single user perceives once generation starts.

### Throughput (job-level)

| Column | JSON field | Formula | Unit |
|--------|-----------|---------|------|
| **output_tps(job)** | `aggregated_metrics.mean_output_throughput_tokens_per_s` | total output tokens ÷ wall-clock run duration | tok/s |

This is the **server's aggregate decode capacity** — the right metric for comparing configurations.  
Relationship to the per-request metric:

```
output_tps(job)  ≈  mean_output_tps(req)  ×  concurrency
```

Use `output_tps(job)` to compare configs; use `mean_output_tps(req)` to understand single-user experience.

### Run metadata

| Column | JSON field | Notes |
|--------|-----------|-------|
| **total requests** | `aggregated_metrics.num_completed_requests` | Requests that finished within the run cap |
| **Wall (s)** | `aggregated_metrics.run_duration` | Elapsed wall time (s); runs are capped by `--max-time-per-run` or `--max-requests-per-run`, whichever fires first |

---

## Traffic scenario notation

`D(N, M)` — the load generator draws request lengths from a distribution with mean **N** input tokens and mean **M** output tokens.

| Scenario | Input tokens | Output tokens | Character |
|----------|-------------|--------------|-----------|
| `D(256, 1024)` | ~256 | ~1024 | Short context, decode-heavy |
| `D(16384, 1024)` | ~16 384 | ~1024 | Long context, prefill-heavy |
