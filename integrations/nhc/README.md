# NHC

[`pantheon.nhc`](pantheon.nhc) adds the check `check_pantheon` to
[NHC](https://github.com/mej/nhc), the node health check that Slurm and PBS
run on a node. It is built on [`pantheon_node_check.py`](../pantheon_node_check.py).

NHC's own GPU check calls `nvidia-healthmon`, which NVIDIA no longer ships.
`check_pantheon` tests the memory of each card and is the same on NVIDIA and
on AMD cards.

```bash
install -m 755 ../pantheon_node_check.py /usr/local/sbin/pantheon_node_check.py
install -m 644 pantheon.nhc /etc/nhc/scripts/pantheon.nhc
```

and in `/etc/nhc/nhc.conf`:

```
* || export TIMEOUT=90
* || check_pantheon
```

| Result of the check | What NHC does |
|---|---|
| `HEALTHY` | Nothing |
| `WATCH` | Writes the result to its log, the node stays in service |
| `FAULT` | Fails the node, with the first line of the result as the reason |
| Nothing was tested | Writes to its log; fails the node when `PANTHEON_FAIL_ON_NO_RESULT=1` |

From our test cluster, with a fault injected:

```console
$ nhc
ERROR:  nhc:  Health check failed:  check_pantheon:  PANTHEON FAULT: GPU 0 FAULT (march_test failed: memory errors detected or the workload aborted)
$ scontrol show node node1 | grep -o -E 'State=[A-Z+]+|Reason=.*'
State=IDLE+DRAIN
Reason=NHC: Health check failed:  check_pantheon:  PANTHEON FAULT: GPU 0 FAULT (march_test failed: memory errors detected or the workload aborted) [root@2026-09-29T16:21:19]
```

Things to know:

- **The check loads the GPUs.** Let NHC run on idle nodes only, with
  `HealthCheckNodeState=IDLE` in `slurm.conf`.
- NHC stops a run after `TIMEOUT` seconds, 30 by default. The check takes
  about 20 seconds once the workloads are compiled, and the first run on a
  node compiles them. Set the timeout as above, or run the node check once by
  hand on each node first.
- Arguments after `check_pantheon` go to the node check, for example
  `check_pantheon --test galpat --duration 20`.
- NHC runs as root with a short `PATH`. Add a line such as
  `* || export PATH="/usr/local/cuda/bin:/opt/pantheon/bin:$PATH"` when
  `nvcc` or `pantheon` are somewhere else.
