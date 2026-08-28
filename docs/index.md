# Pantheon

Universal GPU stress and diagnostics suite for CUDA, ROCm/HIP, and mock-mode validation.

## Inference-path diagnostics

Pantheon includes an `inference` suite for the GPU-side patterns behind LLM
serving: autoregressive decode, long-context prefill, and paged KV-cache
maintenance. The workloads are diagnostic stress tests, not end-to-end model
or model-quality benchmarks.

```bash
pantheon --test inference --duration 60 --gpu 0 --mem 50 --verify --profile
```

| Workload | Focus | Result |
| --- | --- | --- |
| `llm_decode` | Dependent KV-cache gathers and projection-like work | `tokens/s` |
| `llm_prefill` | Causal context scans and projection-like work | `prompt-tokens/s` |
| `kv_cache_churn` | Paged/ragged cache reads and updates | `cache-updates/s` |

