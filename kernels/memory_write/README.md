# Memory Write Aggressive

`memory_write` is a VRAM write-bandwidth stress test. It writes deterministic, entropy-mixed `uint4` patterns over the requested memory budget using coalesced grid-stride stores.

`memory_write_agg` is the same workload with a different data background, not a
separate kernel. It was previously a duplicate source file identical apart from
one constant, and is now a registry alias for
`memory_write --init_pattern 2`.

| `--init_pattern` | Background | Selected by |
| :--- | :--- | :--- |
| 0 | Solid zeros | `--init_pattern 0` |
| 1 | Solid ones | `--init_pattern 1` |
| 2 | Crosstalk `0xAAAAAAAA`/`0x55555555` + entropy | `memory_write_agg` |
| 3 | Rail-to-rail `0x00000000`/`0xFFFFFFFF` + entropy | **default**, `memory_write` |

Note the defaults differ from `memory_read`, which defaults to crosstalk. Each
workload keeps the background it has always used so published scores stay
comparable.

## What It Stresses

| Area | Stress mechanism |
| :--- | :--- |
| VRAM write bandwidth | Repeated full-buffer coalesced vector stores. |
| Memory controller | Large write bursts across the allocated memory region. |
| Data bus crosstalk | Alternating `0xAAAAAAAA` / `0x55555555` style payloads. |
| Compression avoidance | Address-derived entropy makes the payload hard to compress. |
| Write correctness | Optional verification recomputes the expected payload. |

## How It Works

1. The test allocates `--mem` percent of free VRAM.
2. Each active launch walks the allocation with coalesced `uint4` stores.
3. `init_pattern=0` writes zeroes, `1` writes ones, and `2` writes entropy/crosstalk patterns.
4. `kernel_loops` controls how many full-buffer write passes happen per launch.
5. With `--verify`, every vector lane is checked against the deterministic expected pattern.

## Command Examples

```bash
pantheon --test memory_write --gpu 0 --duration 30 --mem 90
pantheon --test memory_write --gpu 0 --duration 30 --mem 90 --verify   # crosstalk background
```

## Runtime Parameters

| Parameter | Default | Effect |
| :--- | ---: | :--- |
| `block_size` | `256` | Threads per block. |
| `grid_size` | `0` | Number of blocks. `0` means auto-calculate from occupancy. |
| `kernel_loops` | `10` | Full-buffer write passes per launch. |
| `warmup_iters` | `5` | Warmup launches before telemetry. |
| `sync_mode` | `2` | `0=Spin`, `1=Yield`, `2=Block`. |
| `init_pattern` | `2` | `0=Zeroes`, `1=Ones`, `2=Entropy/Crosstalk`. |

## Output And Interpretation

`Throughput` is reported in `GB/s`. For stress analysis, compare throughput with max memory temperature, board power, memory utilization, and throttle reason. A slightly slower configuration can still be more stressful if it raises memory temperature or power more consistently.

## Failure Signals

| Symptom | Possible meaning |
| :--- | :--- |
| Verification fault | Write corruption, retention error, or injected payload damage. |
| Low GB/s | Write-combining failure, throttling, ECC pressure, or poor occupancy. |
| High memory temperature | Sustained GDDR/HBM write stress. |

## Source

The implementation lives in [`memory_write.cpp`](./memory_write.cpp).
