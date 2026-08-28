# LLM Decode

`llm_decode` is a latency-oriented inference-path stress workload. Each GPU
thread performs a dependent gather through a synthetic KV-cache window while
executing small projection-like FMA work. It is designed to reveal the combined
cache, DRAM, scheduling, clock, and power response of autoregressive token
generation.

```bash
pantheon --test llm_decode --gpu 0 --duration 60 --mem 50 --verify
```

| Item | Behavior |
| --- | --- |
| Working set | A synthetic KV-cache store sized from `--mem` percent of free VRAM. |
| Access pattern | Dependent, strided gathers through a bounded history window. |
| Compute | Small projection-like fused multiply-add chains. |
| Result | Synthetic token iterations per second, reported as `tokens/s`. |
| Verification | `--verify` checks that the read-only KV-cache pattern is unchanged. |

This is a subsystem stress test, not an end-to-end model benchmark. It does
not include a tokenizer, model weights, a serving queue, network latency, or
time-to-first-token. Compare its telemetry and relative changes between driver,
power, cooling, and hardware configurations, rather than treating its absolute
throughput as model-serving performance.
