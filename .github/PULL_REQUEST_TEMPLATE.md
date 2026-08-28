**What this changes, and why**

**How it was verified**

- [ ] `python3 -m pytest` passes
- [ ] `make PLATFORM=MOCK` builds
- [ ] Ran on a real GPU (state which), if this touches kernel code

CI cannot verify results against real silicon, and cannot see concurrency
nondeterminism — the mock backend runs a single logical thread, so a kernel
whose result depends on execution order passes CI and fails on hardware. If
this changes a kernel, please run it with `--verify` and `--inject_error` on a
GPU and say what you saw.

**If this adds or changes a workload**

- [ ] `kernels/<name>/README.md` says what it stresses and what its metric does not mean
- [ ] `--verify` prints a verdict on success as well as failure
- [ ] `--inject_error` is detected
- [ ] The reported metric is named for the quantity actually measured
