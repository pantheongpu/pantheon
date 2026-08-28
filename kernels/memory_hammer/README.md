# Memory Hammer

`memory_hammer` repeatedly reads pairs of **aggressor** addresses that bracket
an untouched **victim**, then verifies the entire buffer. It looks for cells
that change value because of activity on *neighbouring* rows rather than
because anything wrote to them. The workload never writes during the hammer
phase, so any difference found at verification is a disturbance.

## The Cache Problem

This is the detail that decides whether the test does anything at all.

A hammer only disturbs a victim if the reads actually reach DRAM and open a
row. GPU caches will happily answer a tight loop over two addresses forever, and
a cached read disturbs nothing. Choosing a cache-bypassing load instruction is
**not sufficient** — if the aggressor working set fits in L2, the reads are
still served on-chip.

The workload therefore gives every thread several aggressor pairs spread across
the whole allocation (`--hammer_pairs`, default 8) and walks all of them on each
round, so the aggregate footprint exceeds L2 and a re-read has to go back to
memory. Startup prints the comparison directly:

```
  -> Aggressor Set: ~14 MiB vs L2 2 MiB (exceeds L2)
```

If that line says `FITS IN L2 - reads will be cache hits`, the run is not
hammering anything; raise `--hammer_pairs`. The difference is visible in the
throughput: on a 12 GB Ampere card, one pair per thread sustains ~5.7e10
reads/s (cache speed) while eight pairs drop to ~2.2e10 reads/s, which is the
card's DRAM bandwidth.

## Geometry

Two aggressors sit `2 × --hammer_stride` elements apart with the victim
between them. Alternating between the two is what forces a row to close and
reopen; hammering one address repeatedly would sit on an open row buffer.

`--hammer_stride` defaults to 32768 elements (512 KiB). The stride that puts two
aggressors in the same bank on different rows depends on the DRAM address
mapping, which is not public for most parts, so the stride is a knob to
**sweep** rather than a value that is correct out of the box.

## Honest Limits

Without the physical address mapping we cannot prove that a given read opened a
given row, or that two addresses share a bank. The metric is therefore named
`aggressor-reads/s` — what is actually measured — and not "activations/s",
which would imply a DRAM-level guarantee this test cannot make. A `PASS` means
no disturbance was observed at this stride, not that the part is immune.

## Usage

```bash
pantheon --test memory_hammer --duration 60 --gpu 0 --mem 50
```

Sweeping the stride, with a fault map:

```bash
for s in 8192 16384 32768 65536; do
  ./build/memory_hammer 0 60 50 --verify --hammer_stride $s \
    --fault_map hammer_$s.csv
done
```

## Interpreting Results

* `Verification: PASS (0 disturbed cells)` means nothing moved at this stride.
* A failure names the victim index and an `XOR` mask of the flipped bits. A
  victim that flips only at one particular stride is the interesting case — it
  suggests a genuine neighbour relationship rather than a weak cell, which
  would fail regardless of stride.
* Confirm any hit by re-running at the same stride and by running
  `memory_retention` on the same region; a cell that fails both is weak, not
  disturbed.

## Result

`Throughput` reports `aggressor-reads/s`, and startup reports the aggressor
footprint against L2. Read the two together — a high read rate with a footprint
that fits in L2 means the cache absorbed the run.
