# P2P Matrix

`p2p_matrix` measures how fast each GPU can send to every other GPU, one peer at a time, and checks that the data arrives intact. Where `p2p_thrasher` loads one pair of cards, this covers all of them, so a GPU with one slow or lost link shows up even when its neighbour is fine.

## How It Works

Pantheon starts one copy per selected GPU. Each copy measures its own row of the matrix: the transfers that start at its GPU. Together the copies cover every ordered pair of GPUs exactly once.

1. The copy finds the peers it has bidirectional peer access to. Peers it cannot reach are listed as not measured, and a GPU that can reach none skips cleanly.
2. It allocates a 256 MiB buffer on itself and one on each peer, then waits for its turn.
3. The copies of one run take turns through a lock file named for the runner's process, so that no two GPUs are timed while sharing a link. On an 8x V100 NVLink mesh a control run with all eight GPUs sending at once read the same as taking turns, so the lock cost time and changed nothing there; a switched fabric has not been tried.
4. For each peer it copies the buffer with `cudaMemcpyPeerAsync` / `hipMemcpyPeerAsync` in batches. The bandwidth of a peer is the median of its batches, so one stalled batch does not stand for it.
5. With `--verify`, the receiving GPU checks every word against a pattern that depends on the sending GPU.

The time is spread so the whole matrix takes about eight tenths of `--duration`, never less than 0.1 s per direction.

## Output And Interpretation

From one GPU of an 8x V100 (p3dn.24xlarge) NVLink node:

```
  -> GPU 3 to GPU 0: 48.49 GB/s  (data ok)
  -> GPU 3 to GPU 1: 24.2553 GB/s  (data ok)
  -> GPU 3 to GPU 2: 48.4902 GB/s  (data ok)
  -> GPU 3 to GPU 7: 24.2551 GB/s  (data ok)
  -> GPU 3 to GPU 4: no direct peer access, not measured
  ...
[PANTHEON] GPU 3: 4 peers measured, slowest GPU 3 to GPU 7 at 24.2551 GB/s, median 36.3727 GB/s, fastest 48.4902 GB/s
Throughput: 24.2551 GB/s
```

`Throughput` is the slowest direction from that GPU. The numbers above are two NVLink2 links (about 48.5 GB/s) to some peers and one link (about 24.3 GB/s) to others, about 97% of the 25 GB/s a link carries; the GPUs it has no NVLink to are not reachable directly on this instance and are listed, not measured.

On a node where every GPU is joined alike (NVSwitch, or a symmetric mesh) all GPUs read alike. A GPU that reads under 85% of the median GPU's score (with at least three GPUs measured) makes the verdict WATCH and is listed under `interconnect`.

Two limits to know. The score is the slowest peer, so on a node whose peers differ by design (this V100 mesh has single-link and double-link pairs) a pair that loses one of two links reads like its single-link neighbours and does not move the score; the NVLink link counts in the same report see that case. And nodes whose GPUs are not joined alike by design (some on one PCIe switch, some on another) can read as slow for that reason, so read the per-peer lines before concluding anything.

## Command Examples

```bash
pantheon --test p2p_matrix --gpu all --duration 60
pantheon --test p2p_matrix --gpu 0,1,2,3 --duration 30 --verify
```

## Runtime Parameters

| Parameter | Default | Effect |
| :--- | ---: | :--- |
| `block_size` | `256` | Threads per block for the initialization and verification kernels. |
| `grid_size` | `0` | Number of blocks. `0` means auto-calculate from occupancy. |
| `batch_copies` | `8` | Copies between two timings. |
| `buffer_mb` | `256` | Size of each transferred buffer. |

A run on one GPU, or on GPUs that cannot reach each other directly, prints `Skipping: <reason>` and no throughput line.
