# LLM Prefill

`llm_prefill` is a throughput-oriented inference-path stress workload. It
models long-prompt work by combining projection-like FMA chains with a causal
scan across a synthetic context store.

```bash
pantheon --test llm_prefill --gpu 0 --duration 60 --mem 50 --verify
```

| Item | Behavior |
| --- | --- |
| Working set | A synthetic context store sized from `--mem` percent of free VRAM. |
| Access pattern | Tiled causal scans over a context window. |
| Compute | Projection and attention-score-like fused multiply-add work. |
| Result | Synthetic prompt token iterations per second, reported as `prompt-tokens/s`. |
| Verification | `--verify` confirms that the read-only context pattern remains intact. |

The workload is intended to expose relative pressure from longer contexts and
their interaction with cache, DRAM, clocks, power, and thermals. It is not a
replacement for a framework benchmark using a specific model, attention
implementation, precision, or batching policy.
