# GALPAT (Galloping Pattern)

`galpat` writes a uniform background across a region, flips **one** cell to the
opposite value, then reads that cell alternately against every other cell in the
region. The single flipped cell "gallops" against its neighbours, which is what
exposes a cell whose value can be pulled by a specific *other* cell.

## What It Finds That A March Does Not

A march tests each cell against the operations applied to it in address order.
GALPAT tests each cell against **every other cell in the region**, which is why
it catches address-decoder faults and coupling between cells that are far apart
in address space but adjacent in the physical array.

The cost is quadratic: a full GALPAT over N cells is O(N²) reads. Running it
over an entire GPU allocation is not finishable, which is why this workload is
region-bounded by design.

## Region Bounding

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--region_offset` | 0 | First element of the region under test |
| `--region_size` | 1048576 | Number of elements in the region |
| `--region_chunk` | 512 | Gallop window each thread covers |

The region is **clamped** into the allocation rather than rejected, so a region
that runs past the end of memory shrinks instead of failing the run.

The intended use is to sweep `--region_offset` across memory, or to point it at
an address that another workload has already implicated. Its natural pairing is
as a **second stage**: `march_test` or `memory_hammer` finds a suspect address
cheaply, then `galpat` is aimed at that region for the expensive exhaustive
check.

## Usage

```bash
pantheon --test galpat --duration 60 --gpu 0 --mem 50
```

Aiming it at a region a fault map has already implicated:

```bash
./build/galpat 0 60 50 --verify --region_offset 4194304 --region_size 65536 \
  --fault_map galpat_region.csv
```

## Interpreting Results

* `Verification: PASS (0 gallop errors over N passes)` means no cell in the
  region was disturbed by any other cell in the region.
* A failure names the element index and an `XOR` mask of the flipped bits.
* Because the region is bounded, a `PASS` covers only the region that was
  tested. Startup prints the exact `Region: [offset, end)` that was covered —
  read it before concluding anything about the rest of memory.

## Result

`Throughput` reports `gallop-reads/s`. Because coverage is quadratic in region
size, this number is only comparable between runs that used the same
`--region_size`.
