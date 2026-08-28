# AI Workload Suites

Pantheon provides controlled GPU stress workloads for AI execution paths. The
reported rates are synthetic diagnostic rates, not model quality, framework, or
end-to-end service benchmarks.

| Suite | Workloads | Use |
| --- | --- | --- |
| `inference` | `llm_decode`, `llm_prefill`, `kv_cache_churn`, `fused_attention`, `rope_stress`, `quantized_gemm`, `serving_mix`, `speculative_decode`, `moe_router` | Transformer serving-path pressure. |
| `training` | `transformer_train_step` | Forward/backward and optimizer-style pressure. |
| `runtime` | `allocation_fragmentation`, `graph_replay` | Runtime allocation and graph-replay pressure. |
| `ai_auxiliary` | `rag_embedding`, `vision_encoder` | Embedding and vision-encoder projection pressure. |

```bash
pantheon --test inference --duration 60 --gpu 0 --mem 50 --verify --profile
pantheon --test training --duration 60 --gpu 0 --mem 50 --verify --profile
pantheon --test runtime --duration 60 --gpu 0 --mem 25 --verify --profile
pantheon --test ai_auxiliary --duration 60 --gpu 0 --mem 50 --verify --profile
```

`--mem` reserves a percentage of currently free VRAM for the synthetic working
set. Start at 25% when the GPU is serving other work. `--profile` captures the
same per-workload counter, timeline, telemetry, and HTML report artifacts as
the existing Pantheon workloads.

## Scope

The workloads isolate useful GPU-side patterns, including causal scans,
rotations, packed low-precision arithmetic, sparse routing, and state updates.
They do not replace a specific model and runtime benchmark. Validate production
throughput, time-to-first-token, inter-token latency, queueing, and accuracy
with the deployed serving stack as a separate measurement.

## What The Reported Rate Means

Nine of these workloads share one harness and report `ai-ops/s`, which counts
**thread-iterations per second**. It is a synthetic diagnostic rate, not a
serving or training rate: `ai-ops/s` is not tokens, requests, or steps.

Each workload stresses a **different bottleneck**. That distinction is the point
of having more than one of them, and it is load profile rather than arithmetic
that creates it — ALU work inside a memory-fed loop is hidden behind load
latency and changes nothing measurable.

| Workload | Bottleneck | Typical rate |
| --- | --- | --- |
| `rag_embedding` | Random gather across the whole allocation; latency bound | ~5.4e7 |
| `vision_encoder` | Strided patch access, a new page per patch; TLB pressure | ~1.1e8 |
| `serving_mix` | Warp divergence — lanes fetch different numbers of tiles | ~2.1e8 |
| `moe_router` | Routed gather to a per-expert row, plus routing atomics | ~8.2e8 |
| `quantized_gemm` | Integer pipe; no float in the inner loop | ~8.5e8 |
| `speculative_decode` | Branch divergence with a data-dependent early exit | ~1.3e9 |
| `fused_attention` | SFU bound on `exp`, plus a cross-lane reduction | ~1.6e9 |
| `rope_stress` | SFU bound on `sincos` across the head dimension | ~1.7e9 |
| `transformer_train_step` | FP pipe; one load feeds a long dependent FMA chain | ~3.1e9 |

Rates are from one RTX 3060 and are there to show the workloads are genuinely
distinct — a 58x spread end to end — not as targets. Run-to-run noise is under
0.4%, so neighbouring entries differ by well over ten times the error bar.

`allocation_fragmentation` is not part of this harness at all. It allocates and
frees device memory in a fragmenting pattern and reports `alloc-events/s`,
stressing the driver's allocator rather than the GPU pipelines.

!!! note "History"
    Through v1.0.19 these workloads shared one loop: six were byte-identical and
    all ten measured within 0.9% of each other, so running the whole suite told
    you no more than running one of them. `allocation_fragmentation` performed no
    allocation despite its name. Results recorded before that change are not
    comparable with results recorded after it.
