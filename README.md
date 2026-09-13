# Pantheon: Universal GPU Stress & Diagnostics Suite

Pantheon is a cross-platform (CUDA/ROCm) stress testing tool designed to isolate and hammer specific GPU subsystems. Unlike generic benchmarks (Furmark, 3DMark), Pantheon allows you to test specific silicon limits.

## Requirements

### Running From Source

This repository is the source tree. It does not build installable packages or
a standalone executable; run the checked-out runner directly:

```bash
python3 pantheon.py --test baseline_metrics --duration 10
python3 pantheon.py --test tensor_virus --duration 60 --gpu 0
```

Pantheon auto-detects CUDA, ROCm/HIP, or mock mode at runtime. Do not pass
`--platform cuda` for normal use.

Kernels are compiled on first use, or explicitly:

```bash
make PLATFORM=CUDA      # NVIDIA
make PLATFORM=HIP       # AMD
make PLATFORM=MOCK      # CPU backend, no GPU required
```

### Runtime Dependencies
- **Hardware Drivers:** Latest NVIDIA or AMD Linux drivers for the target GPU.
- **Compiler:** CUDA `nvcc` for NVIDIA GPUs, or ROCm/HIP `hipcc` for AMD GPUs. Pantheon uses the local compiler to build GPU workload binaries for the current machine.
- **System tools:** `make` and a C++ compiler such as `g++`.
- **Linux compatibility:** Official x86-64 binaries are built in a `manylinux_2_28` environment and require GLIBC 2.28 or newer.

### Compiler Requirements
To run stress tests natively (and avoid **Mock Mode**), you must have the backend compiler for your specific GPU architecture installed:

#### For NVIDIA GPUs (CUDA)
**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install nvidia-cuda-toolkit
```

#### For AMD GPUs (ROCm/HIP)
**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install hipcc
```

## Quick Start
```bash
python3 pantheon.py --version

# Run a specific "Power Virus" test
python3 pantheon.py --test tensor_virus --duration 60

# Run a specific "VRM Cracker" test
python3 pantheon.py --test pulse_virus --duration 60

# Run the inference-path stress suite on GPU 0
 python3 pantheon.py --test inference --duration 60 --gpu 0 --mem 50

# Ten-minute health check with a verdict at the end
python3 pantheon.py --test quick
```

## Ray tracing workload (rt_virus)

NVIDIA's OptiX headers are proprietary and not redistributable, so this tree
does not ship them and `rt_virus` builds a dummy kernel that reports
`SKIP` by default. To run the real workload on NVIDIA hardware, download the
[OptiX SDK](https://developer.nvidia.com/designworks/optix/download) (free
NVIDIA developer account required) and point `OPTIX_PATH` at its `include`
directory:

```bash
OPTIX_PATH=/opt/NVIDIA-OptiX-SDK-8.0.0/include python3 pantheon.py --test rt_virus --duration 60
```

The build cache keys on `OPTIX_PATH`, so setting or changing it triggers a
rebuild automatically. On AMD hardware, build with `ENABLE_HIPRT=1` instead,
which clones and builds HIP-RT itself.

## AI and inference workloads

The `inference`, `training` and `ai_auxiliary` suites exercise GPU execution
and memory patterns relevant to model serving. They are diagnostic workloads,
not model benchmarks: they load no weights, measure no model quality, and do
not replace an end-to-end benchmark such as vLLM or TensorRT-LLM.

```bash
python3 pantheon.py --test inference --duration 60 --gpu 0 --mem 50
```

See [`docs/ai_workloads.md`](docs/ai_workloads.md) for what each workload
stresses and what its reported rate does and does not mean.

## Hardware Counter Profiling

Add `--profile` to wrap each workload in the native vendor profiler:

```bash
python3 pantheon.py --test tensor_virus --duration 30 --gpu 0 --profile
```

`--profile` is an exhaustive diagnostic mode. Pantheon profiles every selected
workload and GPU, running GPUs serially to avoid profiler contention and to
keep the artifacts isolated. On NVIDIA it runs Nsight Compute (`ncu`) for
hardware counters and Nsight Systems (`nsys`) for a CUDA/NVTX timeline. On AMD
it uses `rocprofv3` to collect performance counters together with the HIP
runtime, kernel-dispatch, and memory-operation trace.

The default counter sets cover SM/CU occupancy and active blocks, instruction
pipeline and memory-operation mix, branch and memory-access efficiency,
scheduler stalls, L1/L2/cache behavior, DRAM traffic, and host/peer transfer
fabric indicators where the installed profiler exposes them. Each profile also
includes the RAS and PCIe reliability snapshots collected before and after the
workload.
Supported devices report ECC correctable and uncorrectable counts, memory
retirement state, and PCIe replay, CRC, TLP/DLLP, and fatal/non-fatal counters.
AMD profiles also collect exposed per-block RAS data and CPER firmware records.
Linux PCIe AER counters and new AER/poison events are included when the host
exposes them. Unsupported hardware is labeled unavailable rather than reported
as zero. One workload/GPU bundle is written below the timestamped run directory:

```text
results/<run-id>/profiles/<workload>/gpu<id>/
  profile_manifest.json
  time_series.csv
  hardware_counters.csv
  ras.json
  summary.json
  trace.*
```

`summary.xlsx` links each result to its bundle and includes the flattened
counter values. Pantheon refuses an explicit profile request when the required
vendor tools are unavailable, instead of silently running without profiler
artifacts.

### What a counter value describes

`--profile` does not limit the profiler's launch count, so one capture holds
the workload's stress kernel beside every fill, checksum and teardown kernel
around it. A counter reported for the whole workload would be meaningless, so
each value is the median across the launches of the **kernel that owned the
profiled runtime**, and the row records which kernel that was:

```text
Counter Kernel                tensor_virus_kernel(int, unsigned int *, int, int)
Counter Kernel Launches       50
Counter Kernel Runtime Share  98.7 %
```

Read a counter as a property of that kernel, not of the run. A low runtime
share means the profiler could not find a dominant kernel and the counters
describe only part of the work.

Attribution follows runtime rather than launch count, so a cheap helper
launched thousands of times cannot outvote the kernel doing the work.

A metric the profiler reports under two different units is split rather than
averaged together -- Nsight Compute emits `Memory Throughput` once as a
percentage of peak and once in byte/s, so both appear:

```text
Counter Memory Throughput [%]        68.38 %
Counter Memory Throughput [byte/s]   11218196555.16 byte/s
```

### Cost

Profiling replays every kernel, so it is far slower than a normal stress run.
Measured across 134 profiled and 62 unprofiled workloads on the same cards at
`--duration 300`: a median of **1212s profiled against 333s unprofiled, about
3.6x**. Budget roughly half a day for a full 46-workload suite on one GPU.

### Host memory

Nsight Compute stages device memory in **host** RAM while it replays kernels,
so a profiled run needs host memory comparable to the GPU memory in use. At
`--mem 99` that means the host should have appreciably more RAM than the card
has VRAM.

Rigs where VRAM approaches host RAM become unreachable partway through a
profiled suite -- SSH stops answering while the hypervisor still reports the
instance healthy. Measured across eleven cards, every rig whose VRAM at 99%
was below about 0.7x host RAM completed; the three above it did not, including
one with 96GB of VRAM on a 62GB host. If you cannot give the host that
headroom, run the suite without `--profile` rather than lowering `--mem`, which
would make the scores incomparable with everything else.

### Requirements and troubleshooting

Both vendor tools must be present: on NVIDIA that is `ncu` **and** `nsys`, and
Pantheon refuses the run if either is missing.

**Profiling requires administrative rights.** The NVIDIA driver ships with
`RmProfilingAdminOnly=1`, so an ordinary user gets almost no counters:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0
```

Run the profiled workload as root, or clear the restriction in the driver
module and reload it.

**Nsight Compute must match the driver.** A build newer than the installed
driver installs and reports its version happily, then fails every capture:

```text
==ERROR== Nsight Compute failed to connect to the CUDA driver
          (stub libcuda.so[.1] on path?)
```

Install the package matching the CUDA version `nvidia-smi` reports -- for a
driver advertising CUDA 12.8, `cuda-nsight-compute-12-8`.

**`InterprocessLockFailed`** means `/tmp/nsight-compute-lock` is owned by
another user; a profiler run as an ordinary user leaves a lock a later root run
cannot take. Remove the file.

## Fault Maps

Console fault reporting is capped at a few lines per run so a widescale
corruption cannot flood the log. That keeps the console readable but means a
run reports an accurate error *count* and only a handful of error *locations*.

Memory workloads accept `--fault_map <file>` to record every failing address:

```bash
./build/memory_read 0 60 90 --verify --fault_map faults.csv
```

The file is self-describing CSV:

```text
# workload=memory_read gpu=0 init_pattern=2 init_pattern_name=crosstalk total_faults=3 recorded=3 capacity=4096
index,expected,actual,xor
1337,0xe76e5272,0xecc3ec9d,0x0badbeef
```

The `xor` column names the flipped bits, which is what distinguishes a single
defective cell from a stuck data lane: one bit position failing across many
unrelated addresses indicates shared structure, not a cell. Supported by
`memory_read`, `memory_write`, `atomic_virus`, `ras_validator`,
`memory_retention`, `march_test`, `memory_hammer`, and `galpat`
(`memory_read_agg` and `memory_write_agg` are aliases of the first two with a
different `--init_pattern`).

## Memory Diagnostics

The `diagnostics` suite looks for defective cells rather than measuring
bandwidth: `march_test`, `memory_hammer`, `galpat`, `memory_retention` and
`ras_validator`.

```bash
python3 pantheon.py --suite diagnostics --duration 300 --gpu 0 --mem 90
```

See [`docs/memory_diagnostics.md`](docs/memory_diagnostics.md) for what each
one finds, how they compose as a funnel, the full `--init_pattern` table, and
the caveats that decide whether a result means anything.

## Reliability checks on every run

Pantheon collects lightweight before-and-after reliability snapshots for every
workload, even without `--profile`. Each result records a `RAS Status`, a
human-readable `RAS Error Delta`, and the path to its JSON report under
`results/<run-id>/reliability/<workload>/gpu<id>/ras.json`.

The snapshot covers NVIDIA ECC and PCIe error counters, AMD RAS and CPER
records where their management tools expose them, and Linux PCIe AER events.
`CLEAN` means supported counters did not change during the workload;
`WARNING` records new correctable or recoverable events; `ERROR` records new
uncorrectable, fatal, or poison events; and `UNAVAILABLE` means the platform
does not expose the relevant telemetry. Hardware counters and traces still
require `--profile`.

Profiler counter names can vary by GPU generation and driver/toolkit version.
Override or extend the default sets without editing the source:

```bash
# Replace the NVIDIA metric list entirely
PANTHEON_CUDA_METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum" \
  python3 pantheon.py --test memory_read --duration 30 --profile

# Add one AMD counter to Pantheon's default list
PANTHEON_HIP_METRICS_APPEND="CUSTOM_COUNTER_NAME" \
  python3 pantheon.py --test memory_write --duration 30 --profile
```

## Verdict

Every run ends with one word per GPU and the evidence for it:

```
VERDICT GPU 0, NVIDIA H100 80GB HBM3: WATCH  (thermally throttled on 2; correctable errors on 8)
  ! thermal: memory_read thermally throttled, GPU at 95 C, memory_read_agg thermally throttled, GPU at 94 C
  ! correctable errors on 8 workload(s): pcie.bad_tlp +9, pcie.lcrc +6
  note: PCIe link recovery events on 8 workload(s): link power-state cycling, not a fault
```

* **HEALTHY**: every workload completed, no error counter that matters moved,
  and nothing ran hot.
* **WATCH**: something deserves a second look: a thermal limit, a GPU at 90 C or
  memory at 95 C, correctable ECC or PCIe errors, or a workload that did not
  complete. One line per kind, worst first.
* **FAULT**: a memory diagnostic (`march_test`, `galpat`, `memory_hammer`,
  `memory_retention`, `memory_retention_bake`, `ras_validator`) failed, or an
  uncorrectable error was counted. Read the workload's log in `results/`.
* **INCOMPLETE**: nothing beyond the idle baseline completed.

PCIe link recovery counts are reported as a note, never held against the
card: they tick on every run on some healthy platforms. The verdict is also
written into the report under `verdicts`, with a structured `findings` block.

`--test quick` is the ten-minute version of this: idle baseline, memory read,
a march test, a retention check and one power-limited compute load.

## Interpretation of Results

The summary report (`results/<timestamp>/summary.xlsx`) contains detailed "Pro" metrics. Here is how to interpret them:

* **Efficiency (MB/J):** Calculated as `Throughput / Watts`.
    * **Healthy:** Stays relatively constant throughout the run.
    * **Degraded:** If this drops significantly during a 1-hour burn-in, your silicon is "leaking" current (thermal runaway) or the VRMs are becoming inefficient due to heat.
* **PCIe Link:** Verifies the physical connection speed (e.g., `Gen4 x16`).
    * **Red Flag:** If it drops to `x8` or `Gen3` under load, check your riser cable, motherboard slot, or GPU mounting pressure.
* **Throttle Reason:** Tells you *why* performance is limited.
    * `[POWER]`: **Normal.** The card hit its TDP limit (expected for viruses).
    * `[THERMAL]`: **Critical.** The core is overheating (typically >83°C Edge or >110°C Junction). Check thermal paste.
    * `[VOLTAGE]`: The VRMs cannot supply enough stable voltage to maintain the clock.
* **Junction (Hotspot) vs. Edge:** * On newer AMD drivers, the primary temperature reported is the **Junction**. This is the absolute hottest single point on the silicon die. 
    * It is normal for Junction to be 15-25°C hotter than the traditional "Edge" average.
* **Max Mem Temp:** The hottest point on your VRAM (Memory/GDDR6X). Keep this under 100°C to avoid permanent damage.
* **Throughput (GB/s):**
    * For `memory_read` / `memory_write`, this should be within 90% of your card's theoretical max bandwidth.
    * Low throughput = Memory Controller instability or aggressive error correction (ECC) kicking in.

## Incremental Reports

For long `--test all` runs, Pantheon writes one atomic
`database/pantheon_report_<run>_<sequence>_<workload>_gpu<id>.json` file when
each workload finishes. These are complete workload records, not snapshots of
an active workload, so a results publisher can safely copy them while the
remaining queue is still running. The aggregate report is still written at the
end of the run. Reports marked `partial`, `failed`, or `incomplete` are
excluded by downstream importers.

## Memory type and vendor

Each GPU entry in a report carries `memory_type` (GDDR6, GDDR6X, GDDR7, HBM3,
...) and `memory_vendor` (Samsung, SK hynix, Micron, ...). On NVIDIA they are
read from the driver's own memory information table through its control
interface on `/dev/nvidiactl`, without root, the same data GPU-Z shows on
Windows; on AMD from `rocm-smi --showmemvendor`. `memory_vendor_source` says
which path answered, or why neither could.

The vendor is the one the board was configured for, one value per card. A
repaired card fitted with another vendor's chips still reports the original,
so treat it as the declared vendor and read the memory diagnostics for what the
chips actually do.

## Per-workload documentation

Each workload has its own page under `kernels/<test>/README.md` describing what
it stresses, how to read a failure, and what its reported metric does not mean.

## Community

Questions, results and ideas go to GitHub Discussions:
https://github.com/pantheongpu/pantheon/discussions. Post the card, the driver,
the command you ran and either the report file or the lines from the final
summary you found interesting. Reproducible bugs go to Issues. Results that
come with a report file can be submitted to the public database through the
[website repository](https://github.com/pantheongpu/pantheongpu_website/issues/new?template=benchmark-submission.yml).

## Disclaimer

Pantheon is an independent project. It is not generated, verified, or endorsed
by **AMD** or **NVIDIA**, and its results are not produced by official vendor
performance labs. Measurements reflect the driver versions, power limits and
thermal conditions of whatever machine they were taken on, so figures from
different environments are not directly comparable.
