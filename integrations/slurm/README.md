# Slurm

Two files, both built on [`pantheon_node_check.py`](../pantheon_node_check.py).

| File | When it runs | What it does |
|---|---|---|
| [`epilog.sh`](epilog.sh) | After each job that had a GPU | Tests the cards of that job for 10 seconds and drains the node when a card has a fault |
| [`burnin.sbatch`](burnin.sbatch) | When you submit it | Tests every card of one node with four workloads and keeps the reports |

## The epilog

```bash
install -m 755 ../pantheon_node_check.py /usr/local/sbin/pantheon_node_check.py
install -m 755 epilog.sh /etc/slurm/epilog.d/50-pantheon.sh
```

and in `slurm.conf`:

```
Epilog=/etc/slurm/epilog.d/*
```

`slurmd` runs the epilog as root with an environment of its own, so the
script does not read yours. Settings go in `/etc/pantheon/epilog.conf`, which
the epilog reads as a shell file:

```bash
# where nvcc or hipcc and the pantheon command are
PANTHEON_PATH=/usr/local/cuda/bin:/opt/pantheon/bin
# defaults shown
PANTHEON_WORKLOADS="march_test"
PANTHEON_DURATION=10
PANTHEON_BUILD_CACHE_DIR=/var/cache/pantheon
PANTHEON_LOG=/var/log/pantheon-epilog.log
```

What you get in the log, from our test cluster:

```
2026-09-29T16:18:20+00:00 job=1 gpus=0 exit=0 PANTHEON HEALTHY: 1 GPU HEALTHY
2026-09-29T16:18:37+00:00 job=3 gpus=0 exit=2 PANTHEON FAULT: GPU 0 FAULT (march_test failed: memory errors detected or the workload aborted)
    GPU 0 (Mock GPU): FAULT, march_test failed: memory errors detected or the workload aborted
```

and after the second job:

```console
$ scontrol show node node1 | grep -o -E 'State=[A-Z+]+|Reason=.*'
State=IDLE+DRAIN
Reason=PANTHEON FAULT: GPU 0 FAULT (march_test failed: memory errors detected or the workload aborted) [root@2026-09-29T16:18:37]
```

Things to know:

- Only a fault drains the node. A warning, or a check that could not run,
  goes to the log and the node stays in service.
- The node is in the state `completing` while the epilog runs. With the
  defaults that is about 20 seconds for each job that had a GPU.
- Run the node check once by hand as root on each node, so that the
  workloads are compiled before the first job ends:
  `PANTHEON_BUILD_CACHE_DIR=/var/cache/pantheon pantheon_node_check.py --test march_test --duration 2`
- A job without a GPU leaves no entry and costs no time.

## The acceptance job

```bash
sbatch --nodelist=node042 --gres=gpu:8 burnin.sbatch
```

The job fails when the verdict is not `HEALTHY`, and its exit code is the
one of the node check. The reports of every card are in
`pantheon-<node>-<job>/database/`. `galpat` and `march_test` run first
because they load a card the least.

```console
$ cat pantheon-node1-5.out
PANTHEON HEALTHY: 1 GPU HEALTHY
GPU 0 (Mock GPU): HEALTHY, march_test 231835000.0 march-ops/s, memory_read 2.33954 GB/s
note: CPU backend: these results describe no GPU
reports: pantheon-node1-5
```

`PANTHEON_WORKLOADS`, `PANTHEON_DURATION` and `PANTHEON_EXTRA_ARGS` in the
environment of `sbatch` change what runs. The output above is from a test
cluster without GPUs, hence the CPU backend.
