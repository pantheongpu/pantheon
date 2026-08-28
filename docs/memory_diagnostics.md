# Memory Diagnostics

The bandwidth workloads answer *does this GPU move data correctly at speed*.
The `diagnostics` suite answers a different question — **which cell is bad, and
why** — using structured patterns from memory-test literature rather than
whichever access order happens to run fastest.

```bash
pantheon --suite diagnostics --duration 300 --gpu 0 --mem 90
```

| Workload | Fault class | Cost |
| --- | --- | --- |
| `march_test` | Stuck-at, transition, and coupling faults, via an ordered March C- sequence run in both address directions. | Linear |
| `memory_hammer` | Disturbance — a victim cell that changes because of reads to its neighbours. | Linear |
| `galpat` | Address-decoder faults and coupling between arbitrary cell pairs, exhaustively within a region. | Quadratic, region-bounded |
| `memory_retention` | Charge loss over time, with the payload left untouched for the interval. | Linear |
| `ras_validator` | Whether platform ECC/RAS counters agree with observed corruption. | Linear |

## Using Them As A Funnel

These compose. Run the linear tests across all of memory to find suspect
addresses cheaply, then aim the quadratic one at what they implicate:

```bash
# 1. Cheap, full-coverage sweep. Record every failing address.
pantheon --test march_test --duration 600 --gpu 0 --mem 90 --verify
./build/memory_hammer 0 600 90 --verify --fault_map hammer.csv

# 2. Exhaustive check of one implicated region.
./build/galpat 0 300 90 --verify --region_offset <addr> --region_size 65536 \
  --fault_map galpat_region.csv
```

Fault maps are self-describing CSV with an `xor` column naming the flipped
bits, which is what separates a single defective cell from a stuck data lane:
one bit position failing across many unrelated addresses indicates shared
structure, not a cell.

## Data Backgrounds

Memory workloads take `--init_pattern` to select the data background written
before verification. A cell can pass under one background and fail under
another, so this is a variable to sweep rather than a setting to pick once.

| Name | Number | Targets |
| --- | --- | --- |
| `zeros` | 0 | Stuck-at-1 cells |
| `ones` | 1 | Stuck-at-0 cells |
| `crosstalk` | 2 | Coupling between adjacent lines |
| `rail_to_rail` | 3 | Maximum switching activity, worst-case droop |
| `checkerboard` | 4 | Adjacent-cell coupling |
| `walking_ones` | 5 | One bit set per word, rotating by address |
| `walking_zeros` | 6 | One bit clear per word, rotating by address |

Names are the preferred form; the numbers still parse so existing scripts keep
working, and fault maps record both. Matching ignores case and `-`/`_`, so
`rail_to_rail`, `rail-to-rail`, and `railtorail` are one background. An
unrecognised name is refused with the valid list rather than falling back to a
default. All backgrounds come from one shared selector, so a pattern means the
same thing in every workload that writes and verifies it.

## Reading The Results Honestly

Two limits are worth knowing before trusting a `PASS`.

**`memory_hammer` can be absorbed by cache.** A hammer only disturbs anything
if its reads actually reach DRAM. If the aggressor working set fits in L2, the
reads are served on-chip and nothing is being hammered — picking a
cache-bypassing load instruction is not sufficient on its own. The workload
spreads several aggressor pairs per thread across the allocation
(`--hammer_pairs`) so the footprint exceeds L2, and prints the comparison at
startup:

```text
  -> Aggressor Set: ~14 MiB vs L2 2 MiB (exceeds L2)
```

If that line reads `FITS IN L2`, the run proves nothing. The metric is named
`aggressor-reads/s` rather than "activations/s" for the same reason: without
the physical DRAM address mapping, Pantheon cannot prove a read opened a row.

**`galpat` covers only its region.** Coverage is quadratic, so a full GALPAT
over a GPU allocation cannot finish. The region actually tested is printed at
startup as `Region: [offset, end)` — a `PASS` says nothing about the rest of
memory.

Per-workload detail lives alongside each kernel under
`kernels/<test>/README.md`.
