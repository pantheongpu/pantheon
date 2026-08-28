# Pantheon v1.0.19 Release Notes

Pantheon v1.0.19 is a diagnostics and metric-honesty release. It adds four
memory-diagnostic workloads that answer "which cell is bad, and why" rather
than "how fast does this move data", gives `--init_pattern` readable names,
and corrects metrics that reported numbers no hardware could produce. It
retains the verification-coverage guarantees from v1.0.18.

## New: Memory Diagnostics

A new `diagnostics` suite groups workloads that look for defective cells using
structured patterns from memory-test literature rather than whichever access
order runs fastest.

```bash
pantheon --suite diagnostics --duration 300 --gpu 0 --mem 90
```

- `march_test` runs March C- in both address directions with per-thread
  private chunks, so the ordered sequence survives a parallel launch. It finds
  stuck-at, transition and coupling faults.
- `memory_hammer` reads aggressor pairs bracketing an untouched victim, looking
  for cells that change because of activity on their neighbours.
- `galpat` gallops one flipped cell against every other cell in a bounded
  region, exposing address-decoder faults. Coverage is quadratic, so it is
  region-bounded by design and intended as a second stage aimed at an address
  the linear tests implicated.
- `memory_retention` writes a payload, leaves it untouched for a chosen
  interval, and verifies it, finding cells that lose charge with time.

These compose as a funnel: sweep memory with the linear tests, record failing
addresses with `--fault_map`, then aim `galpat` at what they implicate.

## Fault Maps

Memory workloads accept `--fault_map <file>` and record every failing address
as self-describing CSV, including an `xor` column naming the flipped bits --
one bit position failing across many unrelated addresses indicates a stuck data
lane rather than a defective cell.

## `--init_pattern` Takes Names

`--init_pattern 2` had to be looked up in a table, and the flag was getting
worse with every pattern added. It now takes names, with numbers still accepted
so existing scripts and recorded fault maps keep working:

```
zeros (0)   crosstalk (2)      checkerboard (4)   walking_ones (5)
ones (1)    rail_to_rail (3)   walking_zeros (6)
```

Matching ignores case and `-`/`_`. An unrecognised name is now refused with the
valid list rather than passing through `atoi` and silently becoming zeros --
a typo'd background quietly testing the wrong thing is the failure mode this
area exists to prevent. Fault maps record the name alongside the number.

`memory_read_agg` and `memory_write_agg` are now registry aliases of
`memory_read` and `memory_write` with a different `--init_pattern`, rather than
separate binaries. Nothing a user types changes.

## Metric Corrections

**AI workloads no longer report fictional units.** Ten workloads share one
harness that measures thread-iterations per second, but each declared its own
semantic unit -- `requests/s`, `train-steps/s`, `image-tiles/s`. The numbers
those units labelled were the same number: on one RTX 3060 all ten land within
0.84% of each other, because the metric is a property of the launch geometry
rather than of the operation each workload is named after. `serving_mix`
reported 2.19e8 requests/s, which read as an inference rate is absurd by about
nine orders of magnitude.

All ten now report `ai-ops/s`. Results recorded under the old units remain
valid measurements of the same quantity.

**`memory_hammer` reports `aggressor-reads/s`, not activations.** Without the
physical DRAM address mapping Pantheon cannot prove a read opened a row, so the
metric is named for what it measures. Startup prints the aggressor footprint
against the L2 size, because a hammer whose working set fits in L2 is answered
by cache and disturbs nothing.

## Verification Fixes

- `memory_pc_pingpong --verify --inject_error` exited 0 and reported no fault.
  The injection is an XOR firing on launch index 0, which both the warmup loop
  and the active loop have, so the same bit was flipped twice and the two XORs
  cancelled. Its self-test could not fail, which means its verification path
  was never exercised. Warmup no longer injects.
- A kernel that skips itself at runtime -- `p2p_thrasher` between GPUs with no
  link between them -- still printed `Throughput: 0.0`, which was recorded as a
  real measurement and was indistinguishable from catastrophically slow
  hardware. Such runs are now recorded as `SKIP`.

## Robustness

Three launch parameters had no upper bound. A large value did not fail; the
launch simply stopped returning, which presents as a hang and on a display GPU
can trip the driver's timeout detection:

- `--grid_size` is clamped to the 262144 blocks the internal fill grids were
  already capped at.
- `--hammer_pairs` is capped at 256; past a handful there is nothing to gain.
- `--kernel_loops` is bounded both absolutely and by work, since a loop is a
  full pass over VRAM for `memory_read` and a few ALU operations elsewhere.
  Real defaults are untouched at every memory setting.

## Process Handling

Under `--profile` the process Pantheon launches is the profiler, and the GPU
workload runs as its child. Every termination path killed only the launched
process, so a watchdog timeout, a telemetry timeout or Ctrl-C reaped the
profiler and left the workload saturating the GPU with nothing watching it.
Launches now start their own session and termination signals the whole process
group.

## Packaging

The OptiX headers are NVIDIA-proprietary and are no longer required to build.
`rt_virus` probes for them and builds its dummy kernel when they are absent,
exactly as it already did for HIP-RT on AMD. Point `OPTIX_PATH` at an OptiX SDK
to build the real RT kernel. A `NOTICE` file records bundled third-party terms.

## Compatibility

`--init_pattern` numbers, all existing test and suite names, and recorded
results remain valid. The AI workload metric label changes from a per-workload
unit to `ai-ops/s`; the underlying quantity is unchanged.
