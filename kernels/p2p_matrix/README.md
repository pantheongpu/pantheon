# P2P Matrix

`p2p_matrix` measures how fast each GPU can send to every other GPU, one peer at a time, and checks that the data arrives intact. Where `p2p_thrasher` loads one pair of cards, this covers all of them, so a GPU with one slow or lost link shows up even when its neighbour is fine.

## How It Works

Pantheon starts one copy per selected GPU. Each copy measures its own row of the matrix: the transfers that start at its GPU. Together the copies cover every ordered pair of GPUs exactly once.

1. The copy finds the peers it has bidirectional peer access to. Peers it cannot reach are listed as not measured, and a GPU that can reach none skips cleanly.
2. It allocates a 256 MiB buffer on itself and one on each peer, then waits for its turn.
3. The copies of one run take turns through a lock file named for the runner's process, because two GPUs sending at once share links and would read slow for it.
4. For each peer it copies the buffer with `cudaMemcpyPeerAsync` / `hipMemcpyPeerAsync` in batches. The bandwidth of a peer is the median of its batches, so one stalled batch does not stand for it.
5. With `--verify`, the receiving GPU checks every word against a pattern that depends on the sending GPU.

The time is spread so the whole matrix takes about eight tenths of `--duration`, never less than 0.1 s per direction.

## Output And Interpretation

```
  -> GPU 3 to GPU 0: 233.1 GB/s  (data ok)
  ...
[PANTHEON] GPU 3: 7 peers measured, slowest GPU 3 to GPU 5 at 148.2 GB/s, median 233.0 GB/s, fastest 234.4 GB/s
Throughput: 148.2 GB/s
```

`Throughput` is the slowest direction from that GPU. On a node where every GPU is joined alike (NVSwitch, or an NVLink mesh) all GPUs read alike. A GPU that reads under 75% of the median GPU's score (with at least three GPUs measured) makes the verdict WATCH and is listed under `interconnect`.

Nodes whose GPUs are not joined alike by design (some on one PCIe switch, some on another, or only some pairs bridged) can read as slow for that reason. Read the per-peer lines before concluding anything.

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
