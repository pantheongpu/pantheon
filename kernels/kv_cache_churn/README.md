# KV Cache Churn

`kv_cache_churn` targets the cache-management path of LLM serving. It performs
sparse reads and updates across synthetic pages, with changing offsets intended
to resemble ragged requests and paged KV-cache maintenance.

```bash
pantheon --test kv_cache_churn --gpu 0 --duration 60 --mem 50 --verify
```

| Item | Behavior |
| --- | --- |
| Working set | A mutable synthetic KV-cache sized from `--mem` percent of free VRAM. |
| Access pattern | Pseudo-random page selection, sparse reads, and token-like appends. |
| Result | Cache updates per second, reported as `cache-updates/s`. |
| Verification | `--verify` detects invalid cache entries after the update phase. |

Use this workload to compare memory-management and cache behavior across
drivers, power limits, and GPU configurations. It does not model a specific
runtime's allocator, eviction policy, request queue, or model architecture.
