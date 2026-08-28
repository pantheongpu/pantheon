# Memory PC Ping-Pong

`memory_pc_pingpong` is a pseudo-channel crossbar stress test. It splits a VRAM allocation into two halves and repeatedly reads from one half while writing transformed data to the other, then reverses direction.

![Memory PC Ping-Pong execution flow](./memory_pc_pingpong_flow.svg)

## What It Stresses

| Area | Stress mechanism |
| :--- | :--- |
| Pseudo-channel/crossbar routing | Alternates traffic between two allocation halves. |
| VRAM read/write path | Reads one side, bit-inverts selected lanes, writes the other side. |
| Physical data movement | Non-temporal access encourages real memory traffic. |
| Data correctness | Optional verification checks the expected bounced payload. |

## How It Works

1. The test allocates `--mem` percent of VRAM and splits it in half.
2. Each launch chooses a direction: first half to second half, then back.
3. The kernel reads eight vectors, flips selected lanes, and writes to the opposite half.
4. Warmups are rounded to an even count so the state is restored before telemetry.

## Command Examples

```bash
pantheon --test memory_pc_pingpong --gpu 0 --duration 30 --mem 90
```

## Runtime Parameters

| Parameter | Default | Effect |
| :--- | ---: | :--- |
| `block_size` | `256` | Threads per block. |
| `grid_size` | `0` | Number of blocks. `0` means auto-calculate. |
| `kernel_loops` | `1` | Logged only; launch counter drives ping-pong direction. |
| `warmup_iters` | `5` | Warmup launches before telemetry. |
| `sync_mode` | `2` | `0=Spin`, `1=Yield`, `2=Block`. |
| `init_pattern` | `0` | `0=Zeroes`, `1=Ones`. |

## Source

The implementation lives in [`memory_pc_pingpong.cpp`](./memory_pc_pingpong.cpp).
