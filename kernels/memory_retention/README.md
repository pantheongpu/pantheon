# Memory Retention

`memory_retention` writes a known payload into VRAM, leaves it completely
untouched for a chosen interval, then verifies it. It is meant to expose cells
that lose their charge with **time**.

## How It Differs From `memory_retention_bake`

The two tests share a name but look for different defects.

| | `memory_retention_bake` | `memory_retention` |
| :--- | :--- | :--- |
| During the hold | Runs an ALU burn to heat the die | Nothing touches the payload |
| Finds | Cells that fail when **hot** | Cells that fail after **time** |
| Variable | Temperature | `--retention_delay` |

A cell can pass one and fail the other. Marginal cells often only fail past a
particular retention interval, which is why the delay is a knob to sweep rather
than a fixed value.

## What It Stresses

| Area | Stress mechanism |
| :--- | :--- |
| Charge retention | The payload sits unrefreshed by any access for the full delay. |
| DRAM, not cache | Payload is written and read back with non-temporal access, so a cached copy cannot mask a leaked cell. |
| Pattern sensitivity | `--init_pattern` selects zeros, ones, or entropy, so retention can be compared across data backgrounds. |

## Why The Wait Happens Inside The Binary

Device memory is released when the process exits. An external tool cannot write
a payload, wait, and read it back across separate invocations, because the
allocation does not survive the first one. The idle period therefore has to
live inside this workload, with the delay exposed as a knob.

## Usage

```bash
pantheon --test memory_retention --duration 60 --gpu 0 --mem 50
```

Run the workload binary directly to sweep the interval, and write every failing
address to a fault map:

```bash
./build/memory_retention 0 300 50 --verify --retention_delay 300 \
  --fault_map retention_300s.csv
```

`--retention_delay` defaults to `--duration`. Console fault output is capped at
a few lines, so use `--fault_map` when the complete set of failing addresses
matters.

## Interpreting Results

* `Verification: PASS` means every cell held its value for the interval.
* A failure reports the element index and an `XOR` mask naming the flipped
  bits. One bit position failing across many unrelated addresses points at a
  data lane rather than a cell.
* Sweeping the delay narrows the retention margin: the shortest delay that
  reproduces a failure is the cell's effective retention limit.

## Result

`Throughput` reports `retained-MiB`, the size of the payload that was held and
verified. Nothing is transferred during the hold, so a bandwidth figure would
be meaningless here.
