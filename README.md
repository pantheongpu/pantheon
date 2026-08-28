[![Daily container workload smoke tests](https://github.com/pantheongpu/pantheongpu/actions/workflows/daily-container-smoke.yml/badge.svg?branch=main)](https://github.com/pantheongpu/pantheongpu/actions/workflows/daily-container-smoke.yml)

<!-- DAILY_SMOKE_STATUS:START -->
### Daily container smoke status

Last completed: [2026-08-27 15:57 UTC](https://github.com/pantheongpu/pantheongpu/actions/runs/33086925129)<br>
Result: **14/14 passed**. These are short mock-backend checks of every workload inside disposable containers.

| Operating system | Latest result |
| --- | --- |
| AlmaLinux 8 | ✅ Passed |
| AlmaLinux 9 | ✅ Passed |
| CentOS Stream 9 | ✅ Passed |
| Debian 11 | ✅ Passed |
| Debian 12 | ✅ Passed |
| Fedora 43 | ✅ Passed |
| Fedora 44 | ✅ Passed |
| RHEL UBI 8 | ✅ Passed |
| RHEL UBI 9 | ✅ Passed |
| Rocky Linux 8 | ✅ Passed |
| Rocky Linux 9 | ✅ Passed |
| Ubuntu 20.04 | ✅ Passed |
| Ubuntu 22.04 | ✅ Passed |
| Ubuntu 24.04 | ✅ Passed |
<!-- DAILY_SMOKE_STATUS:END -->

<!-- TOOLKIT_MATRIX_STATUS:START -->
### Toolkit compile matrix

Last completed: [2026-08-25 21:34 UTC](https://github.com/pantheongpu/pantheongpu/actions/runs/32898044184)<br>
Result: **8/8 passed**. Weekly compile-only builds of every kernel against real CUDA and ROCm toolchains in disposable containers.

| Toolkit | Latest result |
| --- | --- |
| CUDA 11.8 | ✅ Passed |
| CUDA 12.4 | ✅ Passed |
| CUDA 12.6 | ✅ Passed |
| CUDA 12.8 | ✅ Passed |
| CUDA 13.0 | ✅ Passed |
| CUDA 13.2 | ✅ Passed |
| ROCm 6.4 (gfx942) | ✅ Passed |
| ROCm 7.1 (gfx942) | ✅ Passed |
<!-- TOOLKIT_MATRIX_STATUS:END -->

## Disclaimer: Data Sources & Affiliation

> **Important:** The performance metrics and benchmark data displayed on the **Pantheon Leadership Board** are generated from independent testing conducted on physical hardware instances sourced from third-party cloud compute providers, including but not limited to **Vast.ai** and **Runpod.io**.

* **Independent Methodology:** These results represent the physical limits and performance characteristics of hardware as deployed in public data center environments.
* **Non-Official Data:** This data is **not** generated, verified, or endorsed by internal **AMD** or **NVIDIA** engineering teams. These benchmarks are conducted independently of official vendor performance labs.
* **Environment Variance:** Metrics reflect the specific driver versions, power limits, and thermal conditions provided by the respective cloud platforms at the time of execution.

# Pantheon: Universal GPU Stress & Diagnostics Suite

Pantheon is a cross-platform (CUDA/ROCm) stress testing tool designed to isolate and hammer specific GPU subsystems. Unlike generic benchmarks (Furmark, 3DMark), Pantheon allows you to test specific silicon limits.

## Requirements

### Binary Package Install
Official releases are distributed as binary-only Debian packages. The installed command is:

```bash
pantheon --test baseline_metrics --duration 10
pantheon --test tensor_virus --duration 60 --gpu 0
```

Pantheon auto-detects CUDA, ROCm/HIP, or mock mode at runtime. Do not pass `--platform cuda` for normal installed use.

To install a downloaded package locally:

```bash
sudo apt install ./dist/pantheongpu_1.0.19_amd64.deb
```

The public website repository can publish the same `.deb` through an APT repository so users can install with `sudo apt-get install pantheongpu`.

### Uninstall

If Pantheon was installed from the Debian package:

```bash
sudo apt-get remove pantheongpu
```

Use `sudo apt-get purge pantheongpu` instead to request complete package
removal. If Pantheon was installed with a portable bundle's `install.sh`, run:

```bash
sudo ./uninstall.sh
```

Pass the same custom `PREFIX` or `BINDIR` values used during installation.
The portable uninstaller removes the invoking user's compiled workload cache.

### Runtime Dependencies
- **Hardware Drivers:** Latest NVIDIA or AMD Linux drivers for the target GPU.
- **Compiler:** CUDA `nvcc` for NVIDIA GPUs, or ROCm/HIP `hipcc` for AMD GPUs. Pantheon uses the local compiler to build GPU workload binaries for the current machine.
- **System tools:** `make` and a C++ compiler such as `g++`.
- **Linux compatibility:** Official x86-64 binaries are built in a `manylinux_2_28` environment and require GLIBC 2.28 or newer.

### Supported Linux Versions

The release workflow installs and runs every release bundle in containers for:

| Distribution | Minimum tested version |
| :--- | :--- |
| Ubuntu | 20.04 LTS |
| Debian | 11 |
| Red Hat Enterprise Linux | RHEL UBI 8 |
| CentOS | CentOS Stream 9 |
| RHEL-compatible rebuilds | Rocky Linux 8 and AlmaLinux 8 |
| Fedora | Current supported release |

CI runs these checks as a parallel release matrix and also tests newer Ubuntu, Debian, RHEL UBI, CentOS Stream, Rocky Linux, and AlmaLinux versions. Red Hat UBI is the publicly available RHEL userspace used for RHEL compatibility validation. Other x86-64 distributions should work when they provide GLIBC 2.28 or newer, `make`, and a C++ compiler. Container checks run the mock backend; native CUDA or ROCm use still requires a compatible vendor driver and toolkit on the target host.

Building against the oldest supported GLIBC baseline is intentional: binaries compiled directly on Ubuntu 24.04 can import symbols unavailable on older systems. CI scans the launcher and every embedded ELF library and rejects releases requiring GLIBC newer than 2.28.

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
pantheon --version

# Run a specific "Power Virus" test
pantheon --test tensor_virus --duration 60

# Run a specific "VRM Cracker" test
pantheon --test pulse_virus --duration 60

# Run the inference-path stress suite on GPU 0
 pantheon --test inference --duration 60 --gpu 0 --mem 50
```

## Inference-path stress tests

Pantheon's `inference` suite targets the GPU execution and memory patterns that
are important to LLM serving. These are diagnostic workloads, not model
benchmarks: they do not load model weights, measure model quality, or replace
an end-to-end server benchmark such as vLLM or TensorRT-LLM.

| Test | Focus | Primary result |
| --- | --- | --- |
| `llm_decode` | Autoregressive token generation with dependent KV-cache gathers and small projection-like math | `tokens/s` |
| `llm_prefill` | Long-prompt processing with causal attention-style scans and projection-like math | `prompt-tokens/s` |
| `kv_cache_churn` | Paged/ragged KV-cache reads and updates | `cache-updates/s` |

Additional AI suites are available for attention, RoPE, quantized projection,
mixed serving pressure, speculative decode, MoE routing, training-step pressure,
runtime allocation/graph behavior, RAG embedding, and vision encoding. See
[`docs/ai_workloads.md`](docs/ai_workloads.md) for commands and limitations.

Run the suite or an individual test:

```bash
 pantheon --test inference --duration 60 --gpu 0 --mem 50
pantheon --test llm_decode --duration 60 --gpu 0 --mem 50 --profile
```

`--mem` chooses the percentage of currently free VRAM reserved for the
workload's synthetic context or KV-cache store. Begin with `--mem 25` on a GPU
that is also serving workloads. Use `--profile` to correlate the result with
SM activity, cache behavior, DRAM traffic, power, clocks, and throttling.

## Hardware Counter Profiling

Add `--profile` to wrap each workload in the native vendor profiler:

```bash
pantheon --test tensor_virus --duration 30 --gpu 0 --profile
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
counter values. Profiling can require multiple vendor passes and is therefore
substantially slower than a normal stress run. Pantheon refuses an explicit
profile request when the required vendor tools are unavailable, instead of
silently running without profiler artifacts.

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

The bandwidth workloads answer "does this GPU move data correctly at speed".
The `diagnostics` suite answers a different question — "**which cell is bad, and
why**" — using structured patterns from memory-test literature rather than
whichever access order runs fastest.

```bash
pantheon --suite diagnostics --duration 300 --gpu 0 --mem 90
```

| Workload | Fault class | Cost |
| :--- | :--- | :--- |
| `march_test` | Stuck-at, transition, and coupling faults, via an ordered March C- sequence in both address directions | Linear |
| `memory_hammer` | Disturbance — a victim cell that changes because of reads to its neighbours | Linear |
| `galpat` | Address-decoder faults and coupling between arbitrary cell pairs, exhaustively within a region | Quadratic, so region-bounded |
| `memory_retention` | Charge loss over time, with the payload left untouched | Linear |
| `ras_validator` | Platform ECC/RAS counters agreeing with observed corruption | Linear |

These compose as a funnel: run `march_test` and `memory_hammer` across all of
memory to find suspect addresses cheaply, then aim `galpat` at the implicated
region with `--region_offset` for the exhaustive check. Use `--fault_map` at
every stage so the addresses carry from one to the next.

Two caveats worth reading before trusting a result:

* `memory_hammer` prints its aggressor footprint against the L2 size at
  startup. If the footprint fits in L2 the reads never reach DRAM and the run
  proves nothing — see `kernels/memory_hammer/README.md`.
* `galpat` covers only the region it prints at startup, not all of memory.

## Data Backgrounds (`--init_pattern`)

Memory workloads take `--init_pattern` to select the data background written
before verification. A cell can pass under one background and fail under
another, so the pattern is a variable to sweep rather than a setting to pick
once.

| Name | Number | Background | Targets |
| :--- | :--- | :--- | :--- |
| `zeros` | 0 | All zeros | Stuck-at-1 cells |
| `ones` | 1 | All ones | Stuck-at-0 cells |
| `crosstalk` | 2 | `0xAAAA…`/`0x5555…` + entropy | Coupling between adjacent lines |
| `rail_to_rail` | 3 | `0x0000…`/`0xFFFF…` + entropy | Maximum switching activity, worst-case droop |
| `checkerboard` | 4 | `0xAAAA…`/`0x5555…`, no entropy | Adjacent-cell coupling |
| `walking_ones` | 5 | One bit set per word | A single bit walking by address |
| `walking_zeros` | 6 | One bit clear per word | A single bit walking by address |

Names are the preferred form — `--init_pattern crosstalk` says what it does
where `--init_pattern 2` has to be looked up. Matching ignores case and
`-`/`_`, so `rail_to_rail`, `rail-to-rail`, and `railtorail` are the same
background. The numbers still parse, so existing scripts and recorded fault
maps keep working, and fault maps record both.

An unrecognised name is refused with the list of valid ones rather than
falling back to a default, because a typo silently testing zeros is worse than
a run that will not start.

Patterns 4-6 are deliberately not mixed with entropy: a checkerboard only
exposes coupling if neighbouring values are actually complementary, and a
walking pattern only walks if exactly one bit moves per address. `crosstalk`
and `rail_to_rail` are reachable as the `memory_read_agg` and
`memory_write_agg` aliases. All backgrounds come from one shared selector in
`kernels/common/common.h`, so a pattern means the same thing in every workload
that writes and verifies it.

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
  pantheon --test memory_read --duration 30 --profile

# Add one AMD counter to Pantheon's default list
PANTHEON_HIP_METRICS_APPEND="CUSTOM_COUNTER_NAME" \
  pantheon --test memory_write --duration 30 --profile
```

## Tuning

Automated tuning has been removed from this repository to keep Pantheon focused on the core GPU stress runner. A separate tuning repository may be created later.

Detailed test documentation is maintained alongside each workload under `kernels/<test>/README.md`.

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

## Website Dashboard

For long `--test all` runs, Pantheon also writes one atomic
`database/pantheon_report_<run>_<sequence>_<workload>_gpu<id>.json` file when
each workload finishes. These are complete workload records, not snapshots of
an active workload, so a results publisher can safely copy them while the
remaining queue is still running. The aggregate report is still written at
the end of the run. Reports marked `partial`, `failed`, or `incomplete` remain
excluded by the website importer.

Pantheon includes a built-in web dashboard to visualize your benchmark results and compare different GPUs.

### 1. Install Dependencies
The dashboard is built with MkDocs. Install the Python dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Local Server
The website reads the checked-in `docs/assets/web_data.json` dataset. Start the live preview server:
```bash
mkdocs serve
```
Open http://127.0.0.1:8000 in your browser to view the performance leaderboard.

## Creating a Release
When cutting a new version of Pantheon, use the release workflow instead of manually zipping the repository. Official downloads are binary-only release artifacts; do not publish source archives from this private repository.

### Local Release Build
Build all binary release artifacts:

```bash
make release VERSION=1.0.19
```

This creates:

```text
dist/pantheongpu_1.0.19_amd64.deb
dist/pantheongpu_1.0.19_amd64.tar.gz
dist/pantheongpu_1.0.19_amd64.zip
```

The `.deb` installs binary wrappers only:

```text
/usr/bin/pantheon
/opt/pantheongpu/bin/pantheon
```

It does not ship Python entrypoint files, kernel `.cpp` sources, or the private repository tree.

Release build environment setup is documented in [docs/release_process.md](docs/release_process.md).

The `.tar.gz` and `.zip` files are binary distribution bundles. They include the `.deb`, standalone binaries, `INSTALL.md`, release notes, license, and install instructions for Ubuntu/Debian, RHEL-family systems, and generic Linux. They do not include the private source tree.

`build_pantheon.sh` builds the standalone binary directory without creating old-style archives by default. Official releases use `make release` / `packaging/build_release_bundle.sh` to create the curated binary bundles.

### Local Install Smoke Test

```bash
sudo apt install ./dist/pantheongpu_1.0.19_amd64.deb
pantheon --test baseline_metrics --duration 10
pantheon --test fp64_virus --duration 30 --gpu 0
```

Remove the smoke-test installation when finished:

```bash
sudo apt remove pantheongpu
```

Do not pass `--platform cuda` for normal use. Pantheon auto-detects CUDA, ROCm/HIP, or mock mode at runtime and builds the local GPU workload binaries for the current machine.

Installed Pantheon builds cache compiled workload binaries by version, platform, and GPU target under `${XDG_CACHE_HOME:-$HOME/.cache}/pantheongpu/builds/`, for example `hip-gfx942`, `hip-gfx950`, or `cuda-86`. To use a different writable cache:

```bash
PANTHEON_BUILD_CACHE_DIR="$HOME/.cache/pantheongpu/builds" pantheon --test baseline_metrics --duration 10
```

### Official GitHub Release
Update `VERSION`, commit the change, and push the branch:
```bash
printf "1.0.19\n" > VERSION
git add VERSION
git commit -m "Bump version to 1.0.19"
git push
```
The release workflow runs tests, builds docs, compiles mock kernels, builds portable GLIBC 2.28 release artifacts, and tests the exact candidate across the supported Linux matrix. Only after every compatibility job passes does it create the matching `v<version>` tag and publish the `.deb`, `.tar.gz`, `.zip`, and `SHA256SUMS` files.

After publishing the Pantheon GitHub Release, the workflow also dispatches a website release event to `pantheongpu/pantheongpu_website`. Configure the `PANTHEON_WEBSITE_RELEASE_TOKEN` repository secret with permission to create repository dispatch events in the website repo. The website workflow should listen for:
```yaml
on:
  repository_dispatch:
    types: [pantheongpu_released]
```

To verify downloaded release files:
```bash
sha256sum -c SHA256SUMS
```
