# Contributing to Pantheon

## Building and running

Pantheon compiles a separate binary per workload. Pick a backend explicitly —
there is no auto-detection at build time:

```bash
make PLATFORM=MOCK -j$(nproc)     # CPU backend, no GPU required
make PLATFORM=CUDA -j$(nproc)     # NVIDIA
make PLATFORM=HIP  -j$(nproc)     # AMD
```

The mock backend exists so the suite can be built, tested and debugged on a
machine with no GPU. Run the suite through the checked-out runner:

```bash
python3 pantheon.py --test baseline_metrics --duration 10
python3 -m pytest
```

## What CI checks, and what it cannot

CI runs on GitHub-hosted runners with no GPU. It builds the mock backend,
runs the unit tests, smoke tests the runner, compiles every kernel with `nvcc`,
and verifies that each workload detects an injected fault.

Two things it cannot check, both of which need real hardware:

- **Verification against real silicon.** The mock backend computes correct
  results by construction; it cannot tell you whether a GPU does.
- **Concurrency nondeterminism.** The mock executes a single logical thread, so
  atomic ordering there is trivially deterministic. A kernel whose result
  depends on execution order verifies clean under mock and fails on a GPU. This
  has happened: a workload that fed `atomicAdd`'s return value into its result
  passed CI and failed every run on hardware.

If your change touches kernel code, run it on a real GPU before opening a pull
request, with `--verify` and with `--inject_error`.

## Adding a workload

1. Create `kernels/<name>/<name>.cpp`.
2. Add `kernels/<name>/README.md` describing what it stresses, how to read a
   failure, and what its metric does *not* mean. A test enforces that every
   registered workload has one.
3. Register it in `TEST_REGISTRY` in `pantheon.py`, and in a suite if it belongs
   to one.

A workload must honour two flags:

- `--verify` prints `Verification: PASS` or `FAIL` with a count. Reporting
  nothing on success is not acceptable: silence is indistinguishable from a
  verification that never ran, which is how a broken self-test went unnoticed
  for several releases.
- `--inject_error` corrupts one value so verification is expected to fail. CI
  asserts this, because the clean case alone proves nothing.

## Metrics

Report the quantity you actually measure. A metric named for something it does
not measure is worse than no metric: workloads here once reported `requests/s`
for what was a thread-iteration counter, and ten of them printed the same number
under ten different unit names.

If a result depends on a condition the user must satisfy for it to mean
anything, print that condition. `memory_hammer` prints its aggressor footprint
against the L2 size, because a hammer whose working set fits in cache disturbs
nothing and would otherwise report a confident, meaningless pass.

## Third-party code

Do not add a file whose licence forbids redistribution. A test scans the tree
for such terms and fails the build. NVIDIA OptiX headers are the standing
example: they are not carried here, and `rt_virus` probes for them and builds a
dummy kernel when absent. See `NOTICE`.
