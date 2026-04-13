# Speed results (raw + summary)

Place artifacts here, for example:

- **genai-bench** experiment directories (logs, config)  
- `genai-bench excel` / `genai-bench plot` exports  
- A short `SUMMARY.md` per hardware / git SHA

## Summary table (template)

| Model | KV config | Output tok/s | TTFT (ms) | TPOT (ms) | ITL (ms) | Workload | Date | Git commit |
|-------|-----------|--------------|-----------|-----------|----------|----------|------|------------|
| Qwen/Qwen3-8B | BF16 / auto | — | — | — | — | genai-bench D(256,32), conc=16 | — | — |
| Qwen/Qwen3-8B | INT4 | — | — | — | — | same | — | — |
| Qwen/Qwen3-8B | INT4 + BDR K-only | — | — | — | — | same | — | — |

Replace columns with the metrics genai-bench reports (or your Excel export); keep workload columns identical across rows.
