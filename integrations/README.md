# Integrations

Files that put Pantheon where a cluster already looks for the health of its
nodes. Each one is a plain file that you copy; none of them is installed with
the `pantheon-gpu` package. They work with Pantheon 1.2.2 and later.

| What | For | Files |
|---|---|---|
| Node check | Anything that wants a verdict and an exit code | [`pantheon_node_check.py`](pantheon_node_check.py) |
| Slurm | Test the cards of a job when it ends; accept a node before it goes into service | [`slurm/`](slurm/) |
| NHC | The health check that Slurm or PBS runs on a node | [`nhc/`](nhc/) |
| ReFrame | A regression test with one performance value for each card | [`reframe/`](reframe/) |
| Prometheus | The verdict and the numbers as metrics for node_exporter, and a Grafana dashboard that shows them | [`prometheus/`](prometheus/) |
| Kubernetes | A Job for one node and a Helm chart for many | [`kubernetes/`](kubernetes/) |
| Apptainer | A definition file for sites that run containers through Apptainer | [`apptainer/`](apptainer/) |

Elsewhere:

- GitHub Actions: [`pantheongpu/gpu-health-check`](https://github.com/pantheongpu/gpu-health-check)
  checks the cards of a self-hosted runner before a job uses them.
- Packages: `pipx install pantheon-gpu`, `conda install -c conda-forge pantheon-gpu`,
  and the container image `ghcr.io/pantheongpu/pantheon`.

## The node check

A scheduler wants a yes or a no. `pantheon_node_check.py` runs the workloads
that you name, reads the reports that Pantheon writes, and prints a verdict
for the node and for each card. It needs Python 3.9 or later and nothing else.

```console
$ pantheon_node_check.py --gpu 0 --duration 8
PANTHEON HEALTHY: 1 GPU HEALTHY
GPU 0 (NVIDIA GeForce RTX 3060): HEALTHY, memory_read 336.269 GB/s, march_test 2901230000.0 march-ops/s
  note: PCIe link recovery during memory_read, march_test: link power-state cycling, not a fault
$ echo $?
0
```

A card that returns wrong data (here with Pantheon's own fault injection):

```console
$ pantheon_node_check.py --gpu 0 --test march_test --duration 5
PANTHEON FAULT: GPU 0 FAULT (march_test failed: memory errors detected or the workload aborted)
GPU 0 (NVIDIA GeForce RTX 3060): FAULT, march_test failed: memory errors detected or the workload aborted
$ echo $?
2
```

The first line can stand alone. It is the line that ends up as the reason a
node was drained.

| Exit code | Verdict | Meaning |
|---|---|---|
| 0 | `HEALTHY` | Every workload completed and nothing was found |
| 0 | `SKIPPED` | The job has no GPU, so nothing was tested |
| 1 | `WATCH` | The card works, and something deserves a look: it throttled on temperature, it reached 90 C, its memory reached 95 C, it counted correctable errors, or a workload that is not a memory test did not complete |
| 2 | `FAULT` | A memory test failed, or the card counted uncorrectable errors during the run |
| 3 | `INCOMPLETE`, `NO GPU TESTED`, `NOT RUN` | Nothing was tested or the run did not finish. It is never a pass |

Three things to know:

- **The verdict comes from the reports, not from Pantheon's exit code.**
  Pantheon exits with an error when a workload fails. Errors that a card
  counted during a run that completed are in the report only.
- **No compiler, no test.** Pantheon compiles its workloads with `nvcc` or
  `hipcc` for the cards it finds. Without a compiler it runs on a CPU backend
  that tests no hardware, and the check answers `NO GPU TESTED`, exit code 3.
  `--platform mock` asks for that backend, to try the check on a machine
  without a GPU.
- **The first run on a node is slower.** Pantheon compiles the workloads once,
  which took 80 seconds on our machine, and keeps them in
  `~/.cache/pantheongpu/builds` or in the directory that
  `PANTHEON_BUILD_CACHE_DIR` names. Later runs take the duration plus a few
  seconds.

### Which cards are tested

| `--context` | Cards | Used by |
|---|---|---|
| `node` (default) | Every card of the node | A health check on an idle node, a cron job |
| `prolog`, `epilog` | The cards in `SLURM_JOB_GPUS`, numbered as the node numbers them | The Slurm epilog |
| `job` | The cards in `CUDA_VISIBLE_DEVICES`, numbered as the job sees them | A batch job |

`--gpu 0,1` names the cards and wins over the context. The check runs
Pantheon with `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES` and
`ROCR_VISIBLE_DEVICES` unset, because these renumber the cards. Cards that
are named by UUID, and MIG devices, are not guessed: the check stops and asks
for `--gpu`.

`--report-dir DIR` keeps the reports and the log of each workload. `--json`
prints the result for a program to read, and `--textfile PATH` writes it as
Prometheus metrics, see [`prometheus/`](prometheus/).

### How much load

The workload sets the load, so a check can be as gentle as it needs to be.
Average power during the run, as the median over the cards in our
[public database](https://pantheongpu.com/benchmarks/):

| Workload | A100 SXM4 40GB (limit 400 W) | L40S (limit 350 W) | H100 PCIe (limit 350 W) |
|---|---|---|---|
| `galpat`, memory pattern test | 101 W | 87 W | 91 W |
| `march_test`, memory pattern test | 176 W | 185 W | 176 W |
| `memory_read`, memory bandwidth | 283 W | 218 W | 346 W |
| `incinerator`, compute | 307 W | 265 W | 289 W |

## How these files were tested

- The node check has unit tests in [`tests/test_node_check.py`](../tests/test_node_check.py),
  which run in CI, and CI runs it against Pantheon's CPU backend.
- On two RTX 3060 cards: a healthy run, the card of a job by its node number,
  both cards at once, a card that does not exist, and a fault injected with
  `--inject_error`.
- The Slurm epilog, the acceptance job and the NHC check ran under Slurm
  23.11.4 and NHC (its development branch of August 2026) on a one-node
  test cluster in a container, with
  Pantheon on its CPU backend: a healthy run, a job without a GPU, and an
  injected fault, which drained the node both ways.
- The ReFrame test ran with ReFrame 4.10.4 on an RTX 3060 and on the CPU
  backend, and failed as it should on an injected fault.
- The Prometheus output has unit tests, and a test keeps the Grafana dashboard
  and the exporter on the same set of metrics. The Helm chart passes `helm lint`
  and its rendered manifests validate against the Kubernetes API schema; it
  has not run on a cluster with GPUs. The Apptainer image was built from the
  definition and ran a workload on the CPU backend; it has not run with
  `--nv` on a GPU node.

Not tested: a cluster with real GPUs under Slurm, AMD cards, and MIG devices.
If you run one of these files there, we would like to hear how it went in the
[discussions](https://github.com/pantheongpu/pantheon/discussions).
