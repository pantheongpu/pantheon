# Memory Read Aggressive

`memory_read` is a VRAM read-bandwidth stress test. It initializes the requested memory budget with a selected data background, then repeatedly reads the allocation through coalesced `uint4` loads.

`memory_read_agg` is the same workload with a different data background, not a
separate kernel. It was previously a duplicate source file identical apart from
one constant, and is now a registry alias for
`memory_read --init_pattern 3`.

| `--init_pattern` | Background | Selected by |
| :--- | :--- | :--- |
| 0 | Solid zeros | `--init_pattern 0` |
| 1 | Solid ones | `--init_pattern 1` |
| 2 | Crosstalk `0xAAAAAAAA`/`0x55555555` + entropy | **default**, `memory_read` |
| 3 | Rail-to-rail `0x00000000`/`0xFFFFFFFF` + entropy | `memory_read_agg` |

Patterns 2 and 3 are XORed with per-address entropy so the payload stays
uncompressible; they differ in how adjacent words toggle the bus.

## What It Stresses

| Area | Stress mechanism |
| :--- | :--- |
| VRAM read bandwidth | Repeated coalesced reads over the allocated memory region. |
| Memory controller | Large grid-stride access pattern and unrolled `uint4` loads. |
| Compression avoidance | Entropy and alternating patterns reduce compressed zero-token behavior. |
| Retention/read correctness | Optional verification checks the initialized payload. |

## How It Works

1. The test allocates `--mem` percent of free VRAM.
2. An initialization kernel fills memory with the selected pattern.
3. The active kernel reads `uint4` values in an unrolled grid-stride loop.
4. An accumulator prevents the compiler from removing the reads.
5. Throughput is reported in `GB/s`.

## Command Examples

```bash
pantheon --test memory_read --gpu 0 --duration 30 --mem 90
pantheon --test memory_read_agg --gpu 0 --duration 30 --verify   # rail-to-rail background
```

## Runtime Parameters

| Parameter | Default | Effect |
| :--- | ---: | :--- |
| `block_size` | `256` | Threads per block. |
| `grid_size` | `0` | Number of blocks. `0` means auto-calculate. |
| `kernel_loops` | `10` | Full-buffer read passes per launch. |
| `warmup_iters` | `5` | Warmup launches before telemetry. |
| `sync_mode` | `2` | `0=Spin`, `1=Yield`, `2=Block`. |
| `init_pattern` | `2` | `0=Zeroes`, `1=Ones`, `2=Entropy/Rail-to-rail`. |

## Output And Interpretation

`Score` is reported in `GB/s`. Compare it against expected memory bandwidth and watch memory temperature, power, and throttle signals. Low bandwidth can point to memory controller pressure, ECC/error correction, thermal limits, or a weak memory clock state.

## Failure Signals

| Symptom | Possible meaning |
| :--- | :--- |
| Verification fault | Payload mismatch after initialization/read cycle. |
| Low GB/s | Memory throttling, poor link state, ECC pressure, or bandwidth bottleneck. |
| High memory temperature | GDDR/HBM thermal stress from sustained reads. |

## Source

The implementation lives in [`memory_read.cpp`](./memory_read.cpp).
