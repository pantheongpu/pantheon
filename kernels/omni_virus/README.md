# Omni Virus

`omni_virus` is a mixed-workload furnace. It launches concurrent memory, FP16, FP32, and SFU streams to stress multiple execution and memory paths at the same time.

![Omni Virus execution flow](./omni_virus_flow.svg)

## What It Stresses

| Area | Stress mechanism |
| :--- | :--- |
| Memory path | Alternating non-temporal stores. |
| FP16 path | Half2 FMA stream. |
| FP32 path | FMA stream. |
| SFU path | Transcendental stream. |
| Concurrency | Multiple streams run together to create combined pressure. |

## How It Works

1. The test allocates a memory buffer and compute sink buffers.
2. Four independent streams launch memory, FP16, FP32, and SFU work.
3. `kernel_loops` controls the dominant compute loop count.
4. With `--verify`, memory and compute streams are compared with golden outputs.

## Command Examples

```bash
pantheon --test omni_virus --gpu 0 --duration 30 --mem 90
```

## Runtime Parameters

| Parameter | Default | Effect |
| :--- | ---: | :--- |
| `block_size` | `256` | Threads per block. |
| `grid_size` | `0` | Number of blocks. `0` means auto-calculate. |
| `kernel_loops` | `5000` | Master compute loop count. |
| `warmup_iters` | `5` | Warmup launches before telemetry. |
| `sync_mode` | `2` | `0=Spin`, `1=Yield`, `2=Block`. |
| `init_pattern` | `0` | Memory initialization: `0=Zeroes`, `1=Ones`. |

## Source

The implementation lives in [`omni_virus.cpp`](./omni_virus.cpp).
