# March Test

`march_test` runs a **March C-** sequence over VRAM. Unlike the bandwidth
workloads, which read and write in whatever order goes fastest, a march applies
a fixed, ordered read-then-write sequence to every cell so that the *order* of
operations is itself part of the test.

## Why Ordering Matters

Bandwidth tests find cells that cannot hold a value. A march finds cells whose
value depends on **what happened to a neighbouring cell just before**. Those
coupling faults are invisible to any test that touches memory in parallel or in
an arbitrary order, because the disturbing access and the disturbed read have to
happen in a known sequence for the fault to be observable.

## The Sequence

March C- is six elements. `w0` writes zeros, `r0` reads and expects zeros, and
`⇑`/`⇓` are ascending and descending address order:

| # | Element | Catches |
| :--- | :--- | :--- |
| 0 | `⇑ (w0)` | — (initialisation) |
| 1 | `⇑ (r0, w1)` | Stuck-at-1, transition faults |
| 2 | `⇑ (r1, w0)` | Stuck-at-0, transition faults |
| 3 | `⇓ (r0, w1)` | Coupling faults, descending |
| 4 | `⇓ (r1, w0)` | Coupling faults, descending |
| 5 | `⇓ (r0)` | Final verification |

Running both directions is what separates March C- from a simple write/read
pass: a coupling fault between two cells is only exposed from one direction.

## Per-Thread Private Chunks

A march is inherently sequential, but a GPU is not. Each thread is given a
**private contiguous chunk** and marches it in order, so the ordered sequence is
preserved within a chunk while chunks run in parallel. The host caps the thread
count so no chunk falls below `MIN_CHUNK_ELEMENTS` (4096) — a march over a
handful of elements per thread would test almost nothing, since coupling faults
need a run of addresses to appear.

Console output reports `Chunk/thread` so the ordered run length is visible.
Coupling faults *between* chunks are not covered; sweeping `--grid_size` moves
the chunk boundaries.

## Usage

```bash
pantheon --test march_test --duration 60 --gpu 0 --mem 50
```

Direct invocation with a fault map:

```bash
./build/march_test 0 60 50 --verify --fault_map march_faults.csv
```

## Interpreting Results

* `Verification: PASS (0 march errors over N passes)` means every element
  satisfied every read in the sequence.
* A failure names the element index, the expected and actual values, and an
  `XOR` mask of the flipped bits.
* The march element that failed tells you the fault class — a failure at
  element 1 or 2 is a stuck-at or transition fault, while a failure only in the
  descending elements (3, 4) points at a coupling fault.

## Result

`Throughput` reports `march-ops/s`: read and write operations across all march
elements. It is a progress metric, not a bandwidth figure — the ordered access
pattern is deliberately not the fastest way to move data.
