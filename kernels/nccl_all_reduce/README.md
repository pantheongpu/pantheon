# NCCL All-Reduce

`nccl_all_reduce` runs one all-reduce across every selected GPU at once, through the vendor's collective library: NCCL on NVIDIA, RCCL on AMD. It is the call a multi-GPU training job makes, with the library choosing the algorithm (ring or tree), the transport (NVLink, NVSwitch, PCIe, shared memory) and the number of channels. Where `all_reduce` moves data between two GPUs with plain copies and `p2p_matrix` times each pair on its own, this loads the whole group the way a real job does, so a GPU or link that is fine alone but slows the group shows here.

## How It Works

Pantheon starts one copy per selected GPU and tells each which devices take part (`PANTHEON_PEER_GPUS`) and what the run is called (`PANTHEON_RUN_TOKEN`).

1. The copy loads the library at run time (`libnccl.so.2`, or `librccl.so.1` on AMD) and skips with the reason if it is not there. Run time rather than link time, so a machine without NCCL still runs every other workload. A pip install of NCCL (`nvidia-nccl-cu12`, which PyTorch brings) keeps the library out of the loader's path; Pantheon finds it there, and `PANTHEON_NCCL_LIB=/path/to/libnccl.so.2` names one by hand.
2. It allocates a 256 MiB send buffer, filled with its rank number plus one, and a receive buffer.
3. Rank 0 makes the library's unique id and writes it to a file named for the run; the others wait for it. All copies join one communicator.
4. Each copy calls the all-reduce in batches of ten and times each batch. The score is the median batch.
5. After every batch the copies all-reduce a one-element flag and stop together as soon as any of them has run out of time. Every member of a collective has to make the same number of calls, or the ones still calling wait for ever; a clock in each copy could not promise that.
6. With `--verify`, one more all-reduce goes into a cleared buffer and every element is compared with n(n+1)/2. The inputs are whole numbers a float holds exactly, so the answer does not depend on the order the library adds in, and one wrong bit shows.

A copy gives up with an error if rank 0 does not publish the id within two minutes, and the process is stopped by an alarm if the group never forms.

## Output And Interpretation

From one of two RTX 3060s in a workstation (PCIe, no NVLink, under WSL), NCCL 2.20.5 from pip:

```
[PANTHEON] GPU 0: NCCL_ALL_REDUCE rank 0 of 2 (devices 0,1, NCCL 2.20.5)
  -> Duration (s):   15
  -> Buffer (MiB):   256
  -> Batch size:     10
  -> Library:        /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2
  -> Verify Mode:    ON
  -> Collectives:    120
  -> Algorithm bandwidth: 2.06282 GB/s
Throughput: 2.06282 GB/s
Verification: PASS (0 errors)
```

`Throughput` is the bus bandwidth that nccl-tests reports: the plain size over time, times 2(n-1)/n, because an all-reduce over n ranks makes each rank's links carry that fraction of the buffer twice over. It can be compared with the speed of a link whatever the size of the group. The copies move together, so every GPU reads about the same.

There is no threshold for it yet: what a healthy node reads has to come from real runs of each kind of node, and none has been taken. The score is recorded so that the spread of real nodes can set one. A run that gets the wrong sum, or aborts, makes the verdict WATCH and is listed under `interconnect`. It is not a FAULT: the test cannot say which GPU, link or switch is at fault.

## Checked On

| Node | Library | What was seen |
| :--- | :--- | :--- |
| 2x RTX 3060 (PCIe, no NVLink, WSL), 2026-10-27 | NCCL 2.20.5 (pip) | Both copies join, make the same number of calls (100 to 120 in 12 to 15 s) and stop together, about 2.07 GB/s, sum correct. With `--inject_error` both report `Verification: FAIL` and exit 1. Without the library, with one GPU, or started by hand, each prints `Skipping: <reason>` and exits 0. |

Not yet run on a node with NVLink or NVSwitch, or with RCCL on AMD (it compiles for ROCm in CI); the figure above says nothing about those machines.

## Command Examples

```bash
pantheon --test nccl_all_reduce --gpu all --duration 60
pantheon --test nccl_all_reduce --gpu 0,1,2,3 --duration 30 --verify
pantheon --test nccl_all_reduce --gpu all --duration 10 --verify --inject_error   # must report FAIL
```

## Runtime Parameters

| Parameter | Default | Effect |
| :--- | ---: | :--- |
| `buffer_mb` | `256` | Size of the buffer that is reduced. |
| `batch_iters` | `10` | Collectives between two timings. |

A run on one GPU, or without the library, prints `Skipping: <reason>` and no throughput line.
