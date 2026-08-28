import argparse
import io
import signal
import subprocess
import time
import os
import sys
import glob
import sqlite3
import threading
import datetime
import json
import platform
import shutil
import hashlib
import shlex
import re
import pandas as pd
import numpy as np
import atexit
import builtins
import html
from monitor import HardwareMonitor

try:
    import pynvml
except ImportError:
    pynvml = None

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Try to import psutil, warn if missing
try:
    import psutil
except ImportError:
    print("[PANTHEON] Warning: 'psutil' module not found. System info will be limited.")
    print("           Install it via: pip3 install psutil")
    psutil = None

# --- GLOBAL PROCESS TRACKER (For Cleanup) ---
ACTIVE_PROCS = []

def terminate_process_tree(proc):
    """Kill a launched process and everything it spawned.

    Under --profile the launched process is the profiler (ncu, nsys, rocprofv3)
    and the GPU workload is its child. Killing only the profiler orphans the
    workload, which keeps saturating the GPU with nothing left watching it.
    Each process is started in its own session, so the whole group can go at
    once.
    """
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        # No process group (or already gone) -- fall back to the direct child.
        try:
            proc.kill()
        except OSError:
            pass


def cleanup_zombies():
    """Kill any hanging subprocesses on exit."""
    global ACTIVE_PROCS
    for p in ACTIVE_PROCS:
        if p.poll() is None:  # If still running
            try:
                terminate_process_tree(p)
                tprint(f"[CLEANUP] Killed zombie process PID {p.pid}")
            except:
                pass

# Register immediately so it catches early crashes
atexit.register(cleanup_zombies)

def is_frozen_app():
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')


def is_packaged_app():
    return is_frozen_app() or "__compiled__" in globals()


# --- Configuration ---
if is_frozen_app():
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_BUILD_DIR = os.path.join(BASE_DIR, "build")
BUILD_DIR = SOURCE_BUILD_DIR
KERNEL_DIR = os.path.join(BASE_DIR, "kernels")
RESULTS_BASE_DIR = "results"
DATABASE_DIR = "database"
BUILD_CACHE_FILE = os.path.join(BUILD_DIR, ".pantheon_build_cache.json")
INSTALL_PREFIX = os.environ.get("PANTHEON_INSTALL_PREFIX", "/opt/pantheongpu")
DEFAULT_CUDA_PROFILE_METRICS = [
    # Whole-SM pressure and occupancy
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.per_cycle_active",
    "sm__ctas_active.avg.pct_of_peak_sustained_active",
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__cycles_elapsed.avg",
    "gpu__time_duration.sum",
    # Pipeline mix
    "sm__inst_executed_pipe_fma.sum",
    "sm__inst_executed_pipe_fp64.sum",
    "sm__inst_executed_pipe_tensor.sum",
    "sm__inst_executed_pipe_fp16.sum",
    "sm__inst_executed_pipe_adu.sum",
    "sm__inst_executed_pipe_alu.sum",
    "sm__inst_executed_pipe_cbu.sum",
    "sm__inst_executed_pipe_lsu.sum",
    "sm__inst_executed_pipe_tex.sum",
    "sm__inst_executed_pipe_uniform.sum",
    "sm__inst_executed_pipe_xu.sum",
    "sm__inst_executed_pipe_ipa.sum",
    # Instruction efficiency and memory-operation mix
    "smsp__thread_inst_executed_per_inst_executed.ratio",
    "sm__sass_average_branch_targets_threads_uniform.pct",
    "smsp__inst_executed_op_global_ld.sum",
    "smsp__inst_executed_op_global_st.sum",
    "smsp__inst_executed_op_shared_ld.sum",
    "smsp__inst_executed_op_shared_st.sum",
    "smsp__inst_executed_op_local_ld.sum",
    "smsp__inst_executed_op_local_st.sum",
    # Memory/cache fabric pressure
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__bytes_read.sum.per_second",
    "dram__bytes_write.sum.per_second",
    "dram__bytes.sum.per_second",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum",
    "dram__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_write.sum",
    "lts__t_sectors_op_atom.sum",
    "lts__t_sector_hit_rate.pct",
    "lts__t_requests_op_read.sum",
    "lts__t_requests_op_write.sum",
    "lts__t_requests_op_atom.sum",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct",
    "smsp__sass_average_data_bytes_per_sector_mem_local_op_ld.pct",
    "smsp__sass_average_data_bytes_per_sector_mem_local_op_st.pct",
    "smsp__sass_average_data_bytes_per_wavefront_mem_shared_op_ld.pct",
    "smsp__sass_average_data_bytes_per_wavefront_mem_shared_op_st.pct",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    # Scheduler and stall signals
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warps_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warps_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warps_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__warps_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    "smsp__warps_issue_stalled_not_selected_per_warp_active.pct",
    "smsp__warps_issue_stalled_no_instruction_per_warp_active.pct",
    "smsp__warps_issue_stalled_wait_per_warp_active.pct",
    "smsp__warps_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warps_issue_stalled_branch_resolving.avg.per_cycle_active",
    "smsp__warps_issue_stalled_dispatch_stall.avg.per_cycle_active",
    "smsp__warps_issue_stalled_drain.avg.per_cycle_active",
    "smsp__warps_issue_stalled_imc_miss.avg.per_cycle_active",
    "smsp__warps_issue_stalled_lg_throttle.avg.per_cycle_active",
    "smsp__warps_issue_stalled_membar.avg.per_cycle_active",
    "smsp__warps_issue_stalled_misc.avg.per_cycle_active",
    "smsp__warps_issue_stalled_sleeping.avg.per_cycle_active",
    "smsp__warps_issue_stalled_tex_throttle.avg.per_cycle_active",
    # Copy/PCIe/NVLink-facing traffic when workloads exercise host or peer paths
    "pcie__read_bytes.sum",
    "pcie__write_bytes.sum",
    "pcie__read_bytes.sum.per_second",
    "pcie__write_bytes.sum.per_second",
    "nvlink__rx_data.sum",
    "nvlink__tx_data.sum",
]
DEFAULT_HIP_PROFILE_METRICS = [
    # Whole-GPU/SIMD activity
    "GRBM_COUNT",
    "GRBM_GUI_ACTIVE",
    "SQ_WAVES",
    "SQ_BUSY_CYCLES",
    "SQ_ACTIVE_INST_ANY",
    # Instruction mix
    "SQ_INSTS_VALU",
    "SQ_INSTS_SALU",
    "SQ_INSTS_VMEM",
    "SQ_INSTS_SMEM",
    "SQ_INSTS_FLAT",
    "SQ_INSTS_LDS",
    "SQ_INSTS_MFMA",
    "SQ_INSTS_BRANCH",
    "SQ_INSTS_GDS",
    "SQ_INSTS_EXP_GDS",
    "SQ_INSTS_VOPD",
    "SQ_INSTS_VOPC",
    "SQ_INSTS_VOP1",
    "SQ_INSTS_VOP2",
    "SQ_INSTS_VOP3",
    "SQ_INSTS_VINTRP",
    "SQ_INSTS_BARRIER",
    "SQ_INSTS_SENDMSG",
    "SQ_WAIT_INST_LDS",
    "SQ_WAIT_INST_VMEM",
    "SQ_WAIT_INST_FLAT",
    "SQ_WAIT_INST_LGKM",
    "SQ_WAIT_INST_EXPORT",
    "SQ_LDS_IDX_ACTIVE",
    # Cache/memory pressure
    "TCP_TOTAL_CACHE_ACCESSES",
    "TCP_TOTAL_CACHE_MISSES",
    "TCP_TCC_READ_REQ_LATENCY",
    "TCP_TCC_WRITE_REQ_LATENCY",
    "TCP_TCC_READ_REQ",
    "TCP_TCC_WRITE_REQ",
    "TCP_READ_TAGCONFLICT_STALL_CYCLES",
    "TCP_WRITE_TAGCONFLICT_STALL_CYCLES",
    "TCP_TCC_ATOMIC_WITH_RET_REQ",
    "TCP_TCC_ATOMIC_WITHOUT_RET_REQ",
    "TCP_TCC_READREQ_STALL",
    "TCP_TCC_WRITEREQ_STALL",
    "TCC_BUSY",
    "TCC_EA_RDREQ",
    "TCC_EA_WRREQ",
    "TCC_EA_RDREQ_32B",
    "TCC_EA_RDREQ_64B",
    "TCC_EA_WRREQ_32B",
    "TCC_EA_WRREQ_64B",
    "TCC_HIT",
    "TCC_MISS",
    "TCC_EA0_RDREQ",
    "TCC_EA0_WRREQ",
    "TCC_EA_ATOMIC",
    "TCC_EA_ATOMIC_LEVEL",
    "TCC_EA_WRREQ_LEVEL",
    "TCC_EA_RDREQ_LEVEL",
    "TA_BUSY_avr",
    "TD_BUSY_avr",
    "TA_FLAT_READ_WAVEFRONTS",
    "TA_FLAT_WRITE_WAVEFRONTS",
    # DMA / fabric indicators for peer and host-transfer workloads
    "SDMA0_ACTIVE",
    "SDMA1_ACTIVE",
]
CUDA_PROFILE_ANALYSIS_SET = "full"


class ProfileUnavailableError(RuntimeError):
    """Raised when an explicit --profile request cannot be fulfilled completely."""


def profile_artifact_dir(run_dir, test_name, gpu_id):
    """Return the isolated artifact directory for one workload/GPU profile."""
    return os.path.join(run_dir, "profiles", path_component(test_name), f"gpu{gpu_id}")


def reliability_artifact_dir(run_dir, test_name, gpu_id):
    """Return the lightweight RAS/AER artifact directory for one workload/GPU."""
    return os.path.join(run_dir, "reliability", path_component(test_name), f"gpu{gpu_id}")


def tool_version(tool):
    """Return a compact profiler version string without making profiling fragile."""
    if not tool:
        return "Unavailable"
    try:
        result = subprocess.run([tool, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False)
        output = (result.stdout or result.stderr or "").strip().splitlines()
        return output[0].strip() if output else "Unknown"
    except OSError:
        return "Unknown"


def discover_profile_tools(platform_name):
    """Resolve the tools required for exhaustive counters plus a timeline trace."""
    if platform_name == "CUDA":
        return {"counter": shutil.which("ncu"), "trace": shutil.which("nsys")}
    if platform_name == "HIP":
        return {"counter": shutil.which("rocprofv3"), "trace": shutil.which("rocprofv3")}
    return {"counter": None, "trace": None}


def validate_profile_tools(platform_name, tools):
    missing = [name for name, path in tools.items() if not path]
    if missing:
        names = ", ".join(missing)
        raise ProfileUnavailableError(
            f"--profile requires {names} profiler support for {platform_name}; "
            "refusing to run an incomplete profile."
        )


def filter_supported_profile_metrics(platform_name, tools, gpu_id, requested_metrics):
    """Keep supported default counters and record unavailable architecture-specific ones."""
    if platform_name == "CUDA":
        command = [tools["counter"], "--query-metrics", "--query-metrics-mode", "all", "--devices", str(gpu_id)]
    elif platform_name == "HIP":
        availability_tool = shutil.which("rocprofv3-avail")
        if not availability_tool:
            return list(requested_metrics), [], "rocprofv3-avail unavailable; metrics not pre-filtered"
        command = [availability_tool, "list", "--pmc"]
    else:
        return [], list(requested_metrics), "unsupported platform"

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False)
    except OSError as exc:
        return list(requested_metrics), [], f"capability query failed: {exc}"
    if result.returncode:
        return list(requested_metrics), [], "capability query failed; metrics not pre-filtered"

    available = set(re.findall(r"^\s*([A-Za-z][A-Za-z0-9_.]*)\s+", result.stdout, flags=re.MULTILINE))
    supported = [metric for metric in requested_metrics if metric in available]
    skipped = [metric for metric in requested_metrics if metric not in available]
    if not supported:
        raise ProfileUnavailableError(
            f"No requested {platform_name} performance counters are supported on GPU {gpu_id}."
        )
    return supported, skipped, "validated"


def write_profile_manifest(path, payload):
    payload = dict(payload)
    payload["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, cls=NumpyEncoder, sort_keys=True)


def create_profile_manifest(run_dir, test_name, gpu_id, platform_name, tools, metrics, workload_cmd, supported_metrics=None, skipped_metrics=None, metric_validation="", ras_before=None):
    artifact_dir = profile_artifact_dir(run_dir, test_name, gpu_id)
    ensure_dir(artifact_dir)
    manifest_path = os.path.join(artifact_dir, "profile_manifest.json")
    manifest = {
        "schema_version": 1,
        "status": "planned",
        "test": test_name,
        "gpu_id": gpu_id,
        "platform": platform_name,
        "workload_command": shell_join(workload_cmd),
        "requested_metrics": metrics,
        "supported_metrics": supported_metrics or [],
        "skipped_metrics": skipped_metrics or [],
        "metric_validation": metric_validation,
        "analysis_set": CUDA_PROFILE_ANALYSIS_SET if platform_name == "CUDA" else "",
        "profilers": {
            name: {"path": path or "", "version": tool_version(path)}
            for name, path in tools.items()
        },
        "artifacts": {"counter_files": [], "trace_files": [], "logs": []},
    }
    if ras_before is not None:
        manifest["ras_before"] = ras_before
    write_profile_manifest(manifest_path, manifest)
    return artifact_dir, manifest_path, manifest


def update_profile_manifest(manifest_path, **updates):
    if not manifest_path:
        return
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        manifest = {}
    manifest.update(updates)
    write_profile_manifest(manifest_path, manifest)


def ras_value(value, unit="count"):
    """Normalize a vendor RAS value without treating unavailable telemetry as zero."""
    text = str(value).strip()
    normalized = text.strip("[]").strip().lower()
    if normalized in ("", "n/a", "not supported", "unknown"):
        return {"status": "unsupported", "value": None, "unit": unit}
    if text.lower() in ("enabled", "disabled", "yes", "no", "pending", "none"):
        return {"status": "supported", "value": text, "unit": "state"}
    try:
        return {"status": "supported", "value": int(text, 0), "unit": unit}
    except ValueError:
        try:
            return {"status": "supported", "value": float(text), "unit": unit}
        except ValueError:
            return {"status": "supported", "value": text, "unit": unit}


def ras_source(status, metrics=None, detail="", events=None):
    return {
        "status": status,
        "detail": detail,
        "metrics": metrics or {},
        "events": events or [],
    }


def flatten_ras_json(value, prefix=""):
    """Flatten vendor JSON while retaining only RAS and PCIe reliability signals."""
    flattened = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}/{key}" if prefix else str(key)
            flattened.update(flatten_ras_json(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            flattened.update(flatten_ras_json(child, f"{prefix}/{index}"))
    else:
        name = prefix.lower()
        if any(token in name for token in ("ecc", "correctable", "uncorrectable", "deferred", "poison", "replay", "ras")):
            flattened[prefix] = ras_value(value)
    return flattened


def nvidia_ras_snapshot(gpu_id):
    fields = [
        "ecc.mode.current",
        *[
            f"ecc.errors.{severity}.{window}.{location}"
            for severity in ("corrected", "uncorrected")
            for window in ("volatile", "aggregate")
            for location in (
                "device_memory", "dram", "register_file", "l1_cache", "l2_cache",
                "texture_memory", "cbu", "sram", "total",
            )
        ],
        "ecc.errors.uncorrected.volatile.sram.parity",
        "ecc.errors.uncorrected.volatile.sram.secded",
        "ecc.errors.uncorrected.aggregate.sram.parity",
        "ecc.errors.uncorrected.aggregate.sram.secded",
        "ecc.errors.uncorrected.aggregate.sram.thresholdExceeded",
        "ecc.errors.uncorrected.aggregate.sram.l2",
        "ecc.errors.uncorrected.aggregate.sram.sm",
        "ecc.errors.uncorrected.aggregate.sram.mcu",
        "ecc.errors.uncorrected.aggregate.sram.pcie",
        "ecc.errors.uncorrected.aggregate.sram.other",
        "retired_pages.sbe",
        "retired_pages.dbe",
        "retired_pages.pending",
    ]
    metrics = {}
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_id), f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False,
        )
    except OSError as exc:
        return ras_source("unavailable", detail=str(exc))
    if result.returncode:
        return ras_source("unavailable", detail=(result.stderr or result.stdout).strip())
    values = next((line.split(",") for line in result.stdout.splitlines() if line.strip()), [])
    for field, value in zip(fields, values):
        metrics[field] = ras_value(value)

    # The modern NVML field API exposes transport error counters that are not
    # consistently available through nvidia-smi query fields.
    pcie_fields = {
        "pcie.replay": "NVML_FI_DEV_PCIE_REPLAY_COUNTER",
        "pcie.correctable": "NVML_FI_DEV_PCIE_COUNT_CORRECTABLE_ERRORS",
        "pcie.bad_tlp": "NVML_FI_DEV_PCIE_COUNT_BAD_TLP",
        "pcie.bad_dllp": "NVML_FI_DEV_PCIE_COUNT_BAD_DLLP",
        "pcie.non_fatal": "NVML_FI_DEV_PCIE_COUNT_NON_FATAL_ERROR",
        "pcie.fatal": "NVML_FI_DEV_PCIE_COUNT_FATAL_ERROR",
        "pcie.lcrc": "NVML_FI_DEV_PCIE_COUNT_LCRC_ERROR",
        "pcie.lane": "NVML_FI_DEV_PCIE_COUNT_LANE_ERROR",
        "pcie.naks_received": "NVML_FI_DEV_PCIE_COUNT_NAKS_RECEIVED",
        "pcie.naks_sent": "NVML_FI_DEV_PCIE_COUNT_NAKS_SENT",
        "pcie.l0_to_recovery": "NVML_FI_DEV_PCIE_L0_TO_RECOVERY_COUNTER",
        "pcie.replay_rollover": "NVML_FI_DEV_PCIE_REPLAY_ROLLOVER_COUNTER",
    }
    if not pynvml:
        for name in pcie_fields:
            metrics[name] = {"status": "unavailable", "value": None, "unit": "count"}
    else:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            names = [(name, getattr(pynvml, constant, None)) for name, constant in pcie_fields.items()]
            supported = [(name, field_id) for name, field_id in names if field_id is not None]
            values = pynvml.nvmlDeviceGetFieldValues(handle, [field_id for _, field_id in supported])
            for (name, _), field in zip(supported, values):
                if field.nvmlReturn == pynvml.NVML_SUCCESS:
                    metrics[name] = ras_value(field.value.ullVal)
                else:
                    metrics[name] = {"status": "unsupported", "value": None, "unit": "count"}
            for name, field_id in names:
                if field_id is None:
                    metrics[name] = {"status": "unsupported", "value": None, "unit": "count"}
        except Exception as exc:
            for name in pcie_fields:
                metrics.setdefault(name, {"status": "unavailable", "value": None, "unit": "count"})
            return ras_source("partial", metrics, f"NVML PCIe fields unavailable: {exc}")
    supported_values = [item for item in metrics.values() if item["status"] == "supported"]
    return ras_source("supported" if supported_values else "unsupported", metrics)


def amd_ras_snapshot(gpu_id):
    commands = []
    if find_tool("amd-smi"):
        commands.append(["amd-smi", "metric", "-e", "-k", "--gpu", str(gpu_id), "--json"])
    if find_tool("rocm-smi"):
        commands.append(["rocm-smi", "--showrasinfo", "--json"])
    if not commands:
        return ras_source("unavailable", detail="AMD SMI utility was not found")
    errors = []
    for command in commands:
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False)
            if result.returncode:
                errors.append((result.stderr or result.stdout).strip())
                continue
            metrics = flatten_ras_json(json.loads(result.stdout))
            if metrics:
                return ras_source("supported", metrics)
            return ras_source("unsupported", detail="The driver reported no RAS metrics")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
    return ras_source("unavailable", detail="; ".join(filter(None, errors)))


def amd_cper_snapshot(gpu_id):
    """Collect AMD firmware CPER records when the installed AMD SMI supports it."""
    if not find_tool("amd-smi"):
        return ras_source("unavailable", detail="amd-smi is required for CPER collection")
    command = ["amd-smi", "ras", "--cper", "--gpu", str(gpu_id), "--json"]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False)
    except OSError as exc:
        return ras_source("unavailable", detail=str(exc))
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        status = "unsupported" if "not supported" in detail.lower() else "unavailable"
        return ras_source(status, detail=detail)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return ras_source("unavailable", detail="amd-smi returned non-JSON CPER output")
    metrics = flatten_ras_json({"CPER": payload})
    return ras_source("supported" if metrics else "unsupported", metrics)


def gpu_pci_bdf(platform_name, gpu_id):
    if platform_name == "CUDA":
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=pci.bus_id", "--format=csv,noheader"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip().splitlines()[0].lower()
        except (OSError, IndexError):
            return ""
    return ""


def linux_aer_snapshot(platform_name, gpu_id):
    if platform.system().lower() != "linux":
        return ras_source("unavailable", detail="PCIe AER is collected only on Linux")
    bdf = gpu_pci_bdf(platform_name, gpu_id)
    if not bdf:
        return ras_source("unavailable", detail="GPU PCI bus identifier was unavailable")
    device_path = os.path.realpath(os.path.join("/sys/bus/pci/devices", bdf))
    metrics = {}
    current = device_path
    while current.startswith("/sys/"):
        for path in glob.glob(os.path.join(current, "aer_*")):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    metrics[f"{os.path.basename(current)}/{os.path.basename(path)}"] = ras_value(handle.read().strip())
            except OSError:
                continue
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    if not metrics:
        return ras_source("unsupported", detail=f"AER counters are not exposed for {bdf}")
    return ras_source("supported", metrics, f"PCI BDF {bdf}")


def linux_aer_events():
    """Capture readable PCIe AER/poison events without modifying kernel state."""
    try:
        result = subprocess.run(["dmesg", "--time-format", "iso"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False)
    except OSError as exc:
        return [], str(exc)
    if result.returncode:
        return [], (result.stderr or result.stdout).strip()
    events = [
        line.strip() for line in result.stdout.splitlines()
        if re.search(r"\b(AER|PCIe Bus Error|poison(?:ed)? TLP|NVRM: Xid)\b", line, flags=re.IGNORECASE)
    ]
    return events[-100:], ""


def collect_ras_snapshot(platform_name, gpu_id):
    """Return a non-fatal, capability-aware RAS snapshot for one GPU."""
    if platform_name == "CUDA":
        vendor = nvidia_ras_snapshot(gpu_id)
    elif platform_name == "HIP":
        vendor = amd_ras_snapshot(gpu_id)
    else:
        vendor = ras_source("unsupported", detail=f"RAS collection is not available for {platform_name}")
    aer = linux_aer_snapshot(platform_name, gpu_id)
    events, event_error = linux_aer_events() if platform_name in ("CUDA", "HIP") else ([], "")
    if event_error:
        aer["event_status"] = "unavailable"
        aer["event_detail"] = event_error
    else:
        aer["event_status"] = "supported"
        aer["events"] = events
    sources = {"vendor_ras": vendor, "linux_pcie_aer": aer}
    if platform_name == "HIP":
        sources["amd_cper"] = amd_cper_snapshot(gpu_id)
    return {
        "schema_version": 1,
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform_name,
        "gpu_id": gpu_id,
        "sources": sources,
    }


def diff_ras_snapshots(before, after):
    """Compute per-workload counter deltas; lifetime counters remain in both snapshots."""
    rows = []
    sources = sorted(set(before.get("sources", {})) | set(after.get("sources", {})))
    for source_name in sources:
        prior = before.get("sources", {}).get(source_name, {})
        final = after.get("sources", {}).get(source_name, {})
        metric_names = sorted(set(prior.get("metrics", {})) | set(final.get("metrics", {})))
        for metric_name in metric_names:
            old = prior.get("metrics", {}).get(metric_name, {"status": "unavailable", "value": None, "unit": "count"})
            new = final.get("metrics", {}).get(metric_name, {"status": "unavailable", "value": None, "unit": old.get("unit", "count")})
            delta = None
            if isinstance(old.get("value"), (int, float)) and isinstance(new.get("value"), (int, float)):
                delta = new["value"] - old["value"]
            rows.append({
                "source": source_name,
                "metric": metric_name,
                "before": old.get("value"),
                "after": new.get("value"),
                "delta": delta,
                "unit": new.get("unit", old.get("unit", "count")),
                "status": new.get("status", old.get("status", "unavailable")),
            })
    before_events = set(before.get("sources", {}).get("linux_pcie_aer", {}).get("events", []))
    after_events = after.get("sources", {}).get("linux_pcie_aer", {}).get("events", [])
    return {"metrics": rows, "new_aer_events": [event for event in after_events if event not in before_events]}


def summarize_ras_delta(delta, before=None, after=None):
    """Classify reliability changes without treating unsupported telemetry as clean."""
    warnings = []
    errors = []
    for item in delta.get("metrics", []):
        if item.get("status") != "supported":
            continue
        value = item.get("delta")
        if not isinstance(value, (int, float)) or value <= 0:
            continue
        label = f"{item.get('source')}.{item.get('metric')} +{value:g}"
        name = str(item.get("metric", "")).lower()
        if any(token in name for token in ("uncorrect", "fatal", "poison", "unrecoverable")):
            errors.append(label)
        else:
            warnings.append(label)

    for event in delta.get("new_aer_events", []):
        label = f"PCIe AER: {event}"
        if any(token in str(event).lower() for token in ("fatal", "poison", "uncorrect")):
            errors.append(label)
        else:
            warnings.append(label)

    if errors:
        return {"status": "ERROR", "details": errors + warnings}
    if warnings:
        return {"status": "WARNING", "details": warnings}
    snapshots = (before or {}, after or {})
    available = any(
        source.get("status") in ("supported", "partial")
        for snapshot in snapshots
        for source in snapshot.get("sources", {}).values()
    )
    if not available:
        return {"status": "UNAVAILABLE", "details": ["RAS/AER telemetry is not exposed by this system"]}
    return {"status": "CLEAN", "details": []}

TEST_REGISTRY = {
    # Existing
    "baseline_metrics": {"bin": "idle",          "args": [], "desc": "Baseline Idle Metrics (No Load)"},
    "memory_thermal_asym":  {"bin": "memory_thermal_asym",   "args": [], "desc": "Memory Asymmetric Thermal (Stack Isolator)"},
    "memory_cache_fracture":{"bin": "memory_cache_fracture", "args": [], "desc": "Memory Cache Fracturing (Queue Overload)"},
    "memory_retention_bake":{"bin": "memory_retention_bake", "args": [], "desc": "Memory Retention Bake (Charge Leak Test)"},
    "memory_retention":     {"bin": "memory_retention",      "args": [], "desc": "Memory Retention (Charge Leak Over Time)"},
    "march_test":           {"bin": "march_test",           "args": [], "desc": "March C- (Linear Memory Test)"},
    "memory_hammer":        {"bin": "memory_hammer",        "args": [], "desc": "Memory Hammer (Row Disturb)"},
    "galpat":               {"bin": "galpat",               "args": [], "desc": "GALPAT (Galloping Pattern, Region-Bounded)"},
    "memory_pc_pingpong":   {"bin": "memory_pc_pingpong",    "args": [], "desc": "Memory Pseudo-Channel Ping-Pong (Crossbar)"},
    "memory_bank_thrash":{"bin": "memory_bank_thrash", "args": [], "desc": "Memory Bank Thrasher (Row Misses)"},
    "memory_tsv_thrasher": {"bin": "memory_tsv_thrasher", "args": [], "desc": "Memory TSV Thrasher (PHY Toggle-Rate Stress)"},
    "tlb_avalanche": {"bin": "tlb_avalanche", "args": [], "desc": "TLB Avalanche (MMU Page Walk Stress)"},
    "ras_validator": {"bin": "ras_validator", "args": [], "desc": "RAS Validator (ECC Scrub Stress)"},
    "int_virus": {"bin": "int_virus", "args": [], "desc": "Integer Virus (INT32/Logic Pipeline)"},
    "scheduler": {"bin": "scheduler_virus", "args": [], "desc": "Scheduler Virus (Context Switch / KIPS)"},
    "mma_virus": {"bin": "mma_virus", "args": [], "desc": "MMA Virus (Physical Matrix/Tensor Cores)"},
    "fp64_virus": {"bin": "fp64_virus", "args": [], "desc": "FP64 Chokehold (Double Precision)"},
    "memory_write":      {"bin": "memory_write",       "args": [], "desc": "Memory Write (Standard)"},
    "memory_write_agg": {"bin": "memory_write", "args": ["--init_pattern", "crosstalk"], "desc": "Memory Write (Crosstalk Background)"},
    "memory_read":       {"bin": "memory_read",        "args": [], "desc": "Memory Read (Standard)"},
    "memory_read_agg": {"bin": "memory_read", "args": ["--init_pattern", "rail_to_rail"], "desc": "Memory Read (Rail-to-Rail Background)"},
    "voltage":        {"bin": "compute_virus",   "args": [], "desc": "Voltage Virus (ALU Hammer)"},
    "incinerator":    {"bin": "compute_virus_agg","args": [],"desc": "Incinerator (LDS Stress)"},
    "cache_lat":      {"bin": "cache_latency",   "args": [], "desc": "Cache Latency"},
    "sfu_stress":     {"bin": "sfu_stress",      "args": [], "desc": "SFU Virus (Transcendental Math)"},
    "pcie_bandwidth": {"bin": "pcie_bandwidth",  "args": [], "desc": "PCIe Thrasher (Host <-> Device)"},
    "pulse_virus":    {"bin": "pulse_virus",     "args": [], "desc": "Transient Pulse (VRM Attack 10Hz)"},
    "tensor_virus":   {"bin": "tensor_virus",    "args": [], "desc": "Tensor Virus (FP16 Matrix Power)"},
    "atomic_virus":   {"bin": "atomic_virus",    "args": [], "desc": "Atomic Virus (L2 Cache Thrash)"},
    "omni_virus":      {"bin": "omni_virus",       "args": [], "desc": "Omni Virus (Mem + FP16 + FP32 Async)"},

    # --- NEW: Specialized Hardware Blocks ---
    "p2p_thrasher":     {"bin": "p2p_thrasher",     "args": [], "desc": "P2P Thrasher (Multi-GPU Interconnect)"},
    "all_reduce":       {"bin": "all_reduce",       "args": [], "desc": "All-Reduce (Two-Rank Peer DMA Collective)"},
    "transformer_virus":{"bin": "transformer_virus","args": [], "desc": "Transformer Virus (FP8/FP4 Engine)"},
    "llm_decode":       {"bin": "llm_decode",       "args": [], "desc": "LLM Decode (KV-Cache Gather and Projection)"},
    "llm_prefill":      {"bin": "llm_prefill",      "args": [], "desc": "LLM Prefill (Causal Attention and Projections)"},
    "kv_cache_churn":   {"bin": "kv_cache_churn",   "args": [], "desc": "KV Cache Churn (Paged/Ragged Cache Updates)"},
    "fused_attention":  {"bin": "fused_attention",  "args": [], "desc": "Fused Attention (Synthetic Path Proxy)"},
    "rope_stress":      {"bin": "rope_stress",      "args": [], "desc": "RoPE Stress (Synthetic Rotary-Math Proxy)"},
    "quantized_gemm":   {"bin": "quantized_gemm",   "args": [], "desc": "Quantized GEMM (Synthetic Projection Proxy)"},
    "serving_mix":      {"bin": "serving_mix",      "args": [], "desc": "Serving Mix (Synthetic Request-Pressure Proxy)"},
    "speculative_decode":{"bin": "speculative_decode","args": [], "desc": "Speculative Decode (Synthetic Draft/Verify Proxy)"},
    "moe_router":       {"bin": "moe_router",       "args": [], "desc": "MoE Router (Synthetic Top-K Routing Proxy)"},
    "transformer_train_step": {"bin": "transformer_train_step", "args": [], "desc": "Transformer Train Step (Forward/Backward Proxy)"},
    "allocation_fragmentation": {"bin": "allocation_fragmentation", "args": [], "desc": "Allocation Fragmentation (Synthetic Runtime Proxy)"},
    "graph_replay":      {"bin": "graph_replay",      "args": [], "desc": "Graph Replay (CUDA Graphs / HIP Graphs)"},
    "rag_embedding":    {"bin": "rag_embedding",    "args": [], "desc": "RAG Embedding (Synthetic Projection Proxy)"},
    "vision_encoder":   {"bin": "vision_encoder",   "args": [], "desc": "Vision Encoder (Synthetic Projection Proxy)"},
    "rt_virus":         {"bin": "rt_virus",         "args": [], "desc": "RT Virus (Ray Tracing BVH Stress)"},
    "media_enc_virus":  {"bin": "media_enc_virus",  "args": [], "desc": "Media Enc Virus (Fixed-Function Encoder)"}
}

SUITES = {
    "baseline": [
        "baseline_metrics"
    ],
    "core": [
        "voltage", "incinerator", "fp64_virus", "int_virus", 
        "mma_virus", "sfu_stress", "pulse_virus", "tensor_virus", 
        "transformer_virus", "omni_virus"
    ],
    "memory": [
        "memory_thermal_asym", "memory_cache_fracture", "memory_retention_bake",
        "memory_retention", "memory_bank_thrash", 
        "memory_write", "memory_write_agg", "memory_read", "memory_read_agg", 
        "cache_lat", "atomic_virus"
    ],
    "interconnect": [
        "tlb_avalanche", "scheduler", 
        "pcie_bandwidth", "p2p_thrasher", "all_reduce", "memory_pc_pingpong",
        "memory_tsv_thrasher"
    ],
    "diagnostics": [
        "march_test", "galpat", "memory_hammer", "memory_retention",
        "ras_validator"
    ],
    "fixedFunction": [
        "rt_virus", "media_enc_virus"
    ],
    "inference": [
        "llm_decode", "llm_prefill", "kv_cache_churn", "fused_attention",
        "rope_stress", "quantized_gemm", "serving_mix", "speculative_decode",
        "moe_router"
    ],
    "training": [
        "transformer_train_step"
    ],
    "runtime": [
        "allocation_fragmentation", "graph_replay"
    ],
    "ai_auxiliary": [
        "rag_embedding", "vision_encoder"
    ]
}

# These workloads require peer access or a multi-rank collective. Running
# them with one visible GPU produces a misleading zero result.
MIN_GPUS_FOR_TEST = {
    "p2p_thrasher": 2,
    "all_reduce": 2,
}


def filter_gpu_compatible_tests(queue, available_gpu_count):
    """Return runnable tests and tests skipped for lack of visible GPUs."""
    runnable = []
    skipped = []
    for test_name in queue:
        required = MIN_GPUS_FOR_TEST.get(test_name, 1)
        if available_gpu_count < required:
            skipped.append((test_name, required))
        else:
            runnable.append(test_name)
    return runnable, skipped

def get_app_version():
    """Reads the version string from the VERSION file."""
    if is_frozen_app():
        # PyInstaller bundled location
        base_dir = sys._MEIPASS
    else:
        # Script location (pantheon.py is already in the base dir)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
    version_path = os.path.join(base_dir, "VERSION")
    try:
        with open(version_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "vUnknown"

PANTHEON_VERSION = get_app_version()

def tprint(*args, **kwargs):
    """Custom print function that automatically prepends a timestamp."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    msg = " ".join(str(a) for a in args)
    
    # Smart handling so structural newlines stay outside the timestamp
    if msg.startswith("\n"):
        msg = f"\n[{ts}] {msg[1:]}"
    else:
        msg = f"[{ts}] {msg}"
        
    builtins.print(msg, **kwargs)

def log(msg):
    tprint(f"[PANTHEON] {msg}")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def rocm_bin_directories():
    """Return likely ROCm binary directories in preference order."""
    candidates = []
    for variable in ("ROCM_PATH", "HIP_PATH"):
        root = os.environ.get(variable, "").strip()
        if root:
            candidates.append(os.path.join(os.path.expanduser(root), "bin"))
    candidates.extend(("/opt/rocm/bin", "/opt/rocm/llvm/bin"))

    directories = []
    for candidate in candidates:
        normalized = os.path.abspath(candidate)
        if normalized not in directories and os.path.isdir(normalized):
            directories.append(normalized)
    return directories


def cuda_bin_directories():
    """Return standard CUDA toolkit bin directories, newest toolkit first."""
    candidates = []
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(variable, "").strip()
        if root:
            candidates.append(os.path.join(os.path.expanduser(root), "bin"))
    candidates.append("/usr/local/cuda/bin")
    candidates.extend(sorted(glob.glob("/usr/local/cuda-*/bin"), reverse=True))

    directories = []
    for candidate in candidates:
        normalized = os.path.abspath(candidate)
        if normalized not in directories and os.path.isdir(normalized):
            directories.append(normalized)
    return directories


def augment_rocm_path():
    """Expose standard CUDA and ROCm tool locations to the runner and monitor."""
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    candidates = cuda_bin_directories() + rocm_bin_directories()
    additions = [directory for directory in candidates if directory not in path_entries]
    if additions:
        os.environ["PATH"] = os.pathsep.join(additions + path_entries)


def find_tool(name):
    augment_rocm_path()
    return shutil.which(name)


def path_component(value):
    safe = []
    for char in str(value):
        if char.isalnum() or char in ("-", "_", "."):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "cache"

# --- System & GPU Info Gathering ---

def get_size(byte_count, suffix="B"):
    """Scale bytes to proper format"""
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if byte_count < factor:
            return f"{byte_count:.2f}{unit}{suffix}"
        byte_count /= factor

def get_gpu_static_info(platform_name=None):
    """Detects static GPU details (Name, VRAM, Driver, TDP) via CLI tools."""
    if platform_name == "MOCK":
        return [{
            "id": 0,
            "type": "MOCK",
            "manufacturer": "Pantheon",
            "name": "Mock GPU",
            "memory_total": "2048 MB",
            "driver_version": "Mock",
            "power_limit": "N/A",
            "uuid": "MOCK-GPU-0",
            "serial": "Mock"
        }]

    gpu_list = []
    
    # 1. Try NVIDIA
    if platform_name in (None, "CUDA") and find_tool("nvidia-smi"):
        try:
            cmd = ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,power.limit,uuid,serial", "--format=csv,noheader,nounits"]
            out = subprocess.check_output(cmd, encoding="utf-8").strip()

            vendor_out = subprocess.check_output(["nvidia-smi", "-q"], encoding="utf-8")
            vendors = [line.split(":")[-1].strip() for line in vendor_out.split('\n') if "Subsystem Vendor" in line]

            for i, line in enumerate(out.split('\n')):
                parts = line.split(", ")
                if len(parts) >= 7:
                    gpu_list.append({
                        "id": int(parts[0]),
                        "type": "NVIDIA",
                        "manufacturer": vendors[i] if i < len(vendors) else "NVIDIA",
                        "name": parts[1],
                        "memory_total": f"{parts[2]} MB",
                        "driver_version": parts[3],
                        "power_limit": float(parts[4]) if parts[4] != "[Not Supported]" else "N/A",
                        "uuid": parts[5],
                        "serial": parts[6] if parts[6] not in ["[Not Supported]", "N/A"] else "Unknown"
                    })
        except Exception as e: 
            print(f"[PANTHEON DEBUG] NVIDIA parsing failed: {e}")

    # 2. Try AMD
    if platform_name in (None, "HIP") and find_tool("rocm-smi") and not gpu_list:
        try:
            out = subprocess.check_output([
                "rocm-smi",
                "--showproductname",
                "--showmeminfo", "vram",
                "--showmaxpower",
                "--showserial",
                "--showuniqueid",
                "--json"
            ], encoding="utf-8")
            data = json.loads(out)
            for key, val in data.items():
                if not key.startswith("card"): continue
                idx = int(key.replace("card", ""))
                name = val.get("Card Series", "Unknown AMD GPU")
                vram = val.get("VRAM Total Memory (B)", "0")
                if vram != "0":
                    vram = f"{int(vram) // (1024*1024)} MB"

                # Extract AMD Power Limit
                pwr_limit = "N/A"
                for k, v in val.items():
                    if "Max Graphics Package Power" in k or "Max Power Limit" in k or "Power Cap" in k:
                        try: pwr_limit = float(v); break
                        except: pass
               
                # Extract Serial and Unique ID
                serial = val.get("Serial Number", "Unknown")
                uuid = val.get("Unique ID", "Unknown")

                # Extract AMD OEM
                manufacturer = "AMD"
                if "Oem id" in val:
                    manufacturer = val["Oem id"]

                gpu_list.append({
                    "id": idx,
                    "type": "AMD",
                    "manufacturer": manufacturer,
                    "name": name,
                    "memory_total": vram,
                    "driver_version": "ROCm Driver",
                    "power_limit": pwr_limit,
                    "uuid": uuid,
                    "serial": serial
                })
        except: pass

    return gpu_list

def get_toolkit_version(platform_name):
    """Get CUDA or ROCm/HIP version."""
    version = "Unknown"
    if platform_name == "CUDA":
        try:
            out = subprocess.check_output(["nvcc", "--version"], encoding="utf-8")
            for line in out.split('\n'):
                if "release" in line:
                    version = line.split("release ")[1].split(",")[0]
        except: pass
    elif platform_name == "HIP":
        try:
            # Try to read local version file common on ROCm installs
            if os.path.exists("/opt/rocm/.info/version"):
                with open("/opt/rocm/.info/version") as f:
                    version = f.read().strip()
            else:
                out = subprocess.check_output([find_tool("hipcc") or "hipcc", "--version"], encoding="utf-8")
                version = "HIPCC Detected"
        except: pass
    return version

def get_system_snapshot(platform_name):
    """Aggregates all system info into a dictionary.

    Reports built from this snapshot are committed to the public website
    repository, so it must never include host identifiers such as the
    hostname or IP address.
    """
    snapshot = {
        "pantheon_version": PANTHEON_VERSION,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "os_info": {
            "system": platform.system(),
            "release": platform.release(), # Kernel Version
            "version": platform.version(),
            "arch": platform.machine(),
        },
        "cpu_info": "psutil_missing",
        "ram_info": "psutil_missing",
        "gpu_static_info": get_gpu_static_info(platform_name),
        "toolkit_version": get_toolkit_version(platform_name)
    }

    if psutil:
        vm = psutil.virtual_memory()
        snapshot["cpu_info"] = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "freq": f"{psutil.cpu_freq().current:.2f}Mhz" if psutil.cpu_freq() else "N/A"
        }
        snapshot["ram_info"] = {
            "total": get_size(vm.total),
            "available": get_size(vm.available)
        }

    return snapshot


def write_incremental_workload_reports(snapshot, rows, run_id, sequence):
    """Publish atomically completed workload rows for incremental consumers.

    Each file is a complete report for one workload/GPU pair.  This lets the
    website importer publish finished work while the overall ``--test all``
    run is still active.  A report is never written with a partial row, and
    the final aggregate report remains the source of record at the end.
    """
    ensure_dir(DATABASE_DIR)
    for row in rows:
        test_name = path_component(str(row.get("Test Name", "unknown")))
        gpu_id = row.get("GPU ID", 0)
        filename = (
            f"pantheon_report_{run_id}_{sequence:04d}_{test_name}_gpu{gpu_id}.json"
        )
        payload = dict(snapshot)
        payload["run_id"] = run_id
        payload["record_kind"] = "completed_workload"
        payload["run_status"] = "complete"
        payload["completed_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload["test_results"] = [row]

        target = os.path.join(DATABASE_DIR, filename)
        temporary = f"{target}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4, cls=NumpyEncoder, allow_nan=False)
        os.replace(temporary, target)
        log(f"Incremental report saved: {target}")

# --- Core Logic ---

def detect_platform(preferred="auto"):
    augment_rocm_path()
    if preferred != "auto":
        return preferred.upper()

    # 1. Force Mock Mode via Environment Variable (for CI)
    if os.environ.get("PANTHEON_MOCK") == "1":
        return "MOCK"
    
    # 2. Try Auto-Detect. Prefer the compiler matching the detected GPU vendor
    # when both CUDA and HIP toolchains are installed.
    has_nvcc = find_tool("nvcc") is not None
    has_hipcc = find_tool("hipcc") is not None
    has_nvidia_gpu = find_tool("nvidia-smi") is not None
    has_amd_gpu = find_tool("rocm-smi") is not None or find_tool("amd-smi") is not None

    if has_nvidia_gpu and has_nvcc:
        return "CUDA"
    if has_amd_gpu and has_hipcc:
        return "HIP"
    if has_nvcc:
        return "CUDA"
    if has_hipcc:
        return "HIP"
    
    # 3. Fallback to Mock if nothing else found (Optional, good for local dev without GPU)
    if find_tool("g++"):
        print("[PANTHEON] Warning: No GPU compiler found. Defaulting to CPU Mock mode.")
        return "MOCK"
        
    return "UNKNOWN"


def detect_build_target(platform_name):
    """Mirror the Makefile's architecture target selection for cache safety."""
    if platform_name == "CUDA":
        detected_arch = ""
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                encoding="utf-8",
                stderr=subprocess.DEVNULL,
            )
            detected_arch = out.splitlines()[0].strip().replace(".", "")
        except:
            detected_arch = ""
        if detected_arch == "90":
            detected_arch = "90a"
        return detected_arch or "86"

    if platform_name == "HIP":
        target_gfx = os.environ.get("TARGET_GFX", "").strip()
        if target_gfx:
            match = re.search(r"\bgfx[0-9a-z]+", target_gfx, flags=re.IGNORECASE)
            return match.group(0).lower() if match else target_gfx
        for cmd in (["amdgpu-arch"], ["rocm_agent_enumerator"]):
            try:
                executable = find_tool(cmd[0])
                if not executable:
                    continue
                out = subprocess.check_output([executable], encoding="utf-8", stderr=subprocess.DEVNULL)
                for line in out.splitlines():
                    line = line.strip()
                    match = re.search(r"\bgfx[0-9a-z]+", line, flags=re.IGNORECASE)
                    if match and match.group(0).lower() != "gfx000":
                        return match.group(0).lower()
            except:
                pass
        return "gfx942"

    return "mock"


def build_cache_root():
    """Return the root for persistent compiled workload binaries."""
    override = os.environ.get("PANTHEON_BUILD_CACHE_DIR", "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    if is_packaged_app():
        cache_home = os.environ.get("XDG_CACHE_HOME", "").strip()
        if not cache_home:
            cache_home = os.path.join(os.path.expanduser("~"), ".cache")
        return os.path.abspath(os.path.join(cache_home, "pantheongpu", "builds"))
    return SOURCE_BUILD_DIR


def configure_build_directory(platform_name):
    """Select the build directory for this platform/target before cache checks."""
    global BUILD_DIR, BUILD_CACHE_FILE

    if is_packaged_app() or os.environ.get("PANTHEON_BUILD_CACHE_DIR", "").strip():
        target = path_component(detect_build_target(platform_name))
        platform_part = path_component(platform_name.lower())
        version_part = path_component(PANTHEON_VERSION)
        BUILD_DIR = os.path.join(build_cache_root(), version_part, f"{platform_part}-{target}")
    else:
        BUILD_DIR = SOURCE_BUILD_DIR

    BUILD_CACHE_FILE = os.path.join(BUILD_DIR, ".pantheon_build_cache.json")
    return BUILD_DIR


def iter_build_inputs():
    """Files that should invalidate cached kernel binaries when changed."""
    paths = [os.path.join(BASE_DIR, "Makefile")]
    patterns = ("*.cpp", "*.h", "*.hpp", "*.cuh")
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(KERNEL_DIR, "**", pattern), recursive=True))
    return sorted(path for path in paths if os.path.isfile(path))


def build_inputs_digest():
    digest = hashlib.sha256()
    for path in iter_build_inputs():
        rel_path = os.path.relpath(path, BASE_DIR).replace(os.sep, "/")
        digest.update(rel_path.encode("utf-8"))
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def expected_build_binaries():
    return sorted({
        os.path.join(BUILD_DIR, config["bin"])
        for config in TEST_REGISTRY.values()
    })


def build_cache_key(platform_name):
    return {
        "platform": platform_name,
        "target": detect_build_target(platform_name),
        "target_gfx": os.environ.get("TARGET_GFX", ""),
        "enable_hiprt": os.environ.get("ENABLE_HIPRT", ""),
        "inputs_digest": build_inputs_digest(),
        "binaries": [os.path.basename(path) for path in expected_build_binaries()],
    }


def is_build_cache_current(platform_name):
    try:
        with open(BUILD_CACHE_FILE, "r", encoding="utf-8") as f:
            cached = json.load(f)
    except:
        return False

    if cached != build_cache_key(platform_name):
        return False

    return all(os.path.isfile(path) and os.access(path, os.X_OK) for path in expected_build_binaries())


def write_build_cache(platform_name):
    ensure_dir(BUILD_DIR)
    with open(BUILD_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(build_cache_key(platform_name), f, indent=2, sort_keys=True)


def build_kernels(platform):
    configure_build_directory(platform)
    try:
        ensure_dir(BUILD_DIR)
    except PermissionError:
        print(f"[ERROR] Cannot write Pantheon build cache: {BUILD_DIR}")
        print("        Re-run with sudo, or set PANTHEON_BUILD_CACHE_DIR to a writable directory.")
        sys.exit(1)

    if is_build_cache_current(platform):
        log(f"Detected Platform: {platform}. Kernel build cache is current at {BUILD_DIR}; skipping compilation.")
        return {}

    log(f"Detected Platform: {platform}. Compiling kernels into {BUILD_DIR}...")
    # Keep building independent targets after a compiler error. A single
    # vendor-specific or optional workload must not prevent the rest of the
    # selected suite from being available.
    cmd = ["make", "-k", f"PLATFORM={platform}", f"BUILD_DIR={BUILD_DIR}"]
    if platform == "HIP":
        hipcc = find_tool("hipcc")
        if not hipcc:
            print("[ERROR] AMD GPU detected, but hipcc was not found.")
            print("        Install the ROCm HIP compiler or add its bin directory to PATH.")
            print("        Pantheon also searches $ROCM_PATH/bin, $HIP_PATH/bin, and /opt/rocm/bin.")
            sys.exit(1)
        target_gfx = detect_build_target(platform)
        cmd.extend([f"HIPCC={hipcc}", f"TARGET_GFX={target_gfx}"])
        rocm_path = os.environ.get("ROCM_PATH", "").strip()
        if not rocm_path:
            rocm_path = os.path.dirname(os.path.dirname(os.path.realpath(hipcc)))
        cmd.append(f"ROCM_PATH={rocm_path}")
    # Add cwd=BASE_DIR to force execution in the correct folder
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, cwd=BASE_DIR)
    if result.returncode != 0:
        build_log = os.path.join(BUILD_DIR, "build.log")
        with open(build_log, "w", encoding="utf-8") as handle:
            handle.write(result.stdout or "")
            if result.stderr:
                handle.write("\n--- stderr ---\n")
                handle.write(result.stderr)

        unavailable = {
            test_name: (
                f"Compilation did not produce {config['bin']}. "
                f"See {build_log} for compiler output."
            )
            for test_name, config in TEST_REGISTRY.items()
            if not (os.path.isfile(os.path.join(BUILD_DIR, config["bin"]))
                    and os.access(os.path.join(BUILD_DIR, config["bin"]), os.X_OK))
        }
        if not unavailable:
            # `make -k` can return non-zero after an optional target fails
            # even though every registered workload binary is usable.
            write_build_cache(platform)
            log("Build completed with non-workload target warnings.")
            return {}

        print("[PANTHEON] Some workload builds failed; continuing with available workloads.")
        print(f"[PANTHEON] Full compiler output: {build_log}")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        for test_name in sorted(unavailable):
            print(f"[PANTHEON] Will skip {test_name}: {unavailable[test_name]}")
        return unavailable
    write_build_cache(platform)
    log("Build Complete.")
    return {}


def extract_profiler_data(test_name, gpu_id, artifact_dir):
    """Extract rocprof counter data into this workload/GPU's artifact bundle."""
    prof_dir = os.path.join(artifact_dir, "rocprof")

    csv_candidates = sorted(
        path for path in glob.glob(os.path.join(prof_dir, "**", "*.csv"), recursive=True)
        if "pmc" in os.path.basename(path).lower() or "counter" in os.path.basename(path).lower()
    )
    if csv_candidates:
        csv_path = os.path.join(artifact_dir, "hardware_counters.csv")
        shutil.copyfile(csv_candidates[0], csv_path)
        print(f"[PANTHEON] Extracted hardware counters to: {csv_path}")
        return csv_path
    
    # Hunt for the SQLite file generated by rocprofv3
    db_files = sorted(glob.glob(os.path.join(prof_dir, "*.db")))
    if not db_files:
        return
        
    db_path = db_files[0]
    
    try:
        conn = sqlite3.connect(db_path)
        tables_df = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        
        # Find the specific performance counter table
        pmc_table = next((t for t in tables_df['name'] if 'pmc' in t.lower() or 'counter' in t.lower()), None)
        
        if pmc_table:
            quoted_table = pmc_table.replace('"', '""')
            df = pd.read_sql_query(f'SELECT * FROM "{quoted_table}"', conn)
            
            csv_path = os.path.join(artifact_dir, "hardware_counters.csv")
            df.to_csv(csv_path, index=False)
            print(f"[PANTHEON] Extracted hardware counters to: {csv_path}")
            return csv_path
            
    except Exception as e:
        print(f"[ERROR] Failed to extract profiler DB: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
    return None



def parse_profile_metric_list(raw_value):
    """Parse a comma/newline-separated profiler metric list."""
    metrics = []
    for item in str(raw_value or "").replace("\n", ",").split(","):
        metric = item.strip()
        if metric and metric not in metrics:
            metrics.append(metric)
    return metrics


def get_profile_metrics(platform_name):
    if platform_name == "CUDA":
        override = os.environ.get("PANTHEON_CUDA_METRICS", "").strip()
        if override:
            return parse_profile_metric_list(override)
        metrics = list(DEFAULT_CUDA_PROFILE_METRICS)
        metrics.extend(parse_profile_metric_list(os.environ.get("PANTHEON_CUDA_METRICS_APPEND", "")))
        return list(dict.fromkeys(metrics))
    if platform_name == "HIP":
        override = os.environ.get("PANTHEON_HIP_METRICS", "").strip()
        if override:
            return parse_profile_metric_list(override)
        metrics = list(DEFAULT_HIP_PROFILE_METRICS)
        metrics.extend(parse_profile_metric_list(os.environ.get("PANTHEON_HIP_METRICS_APPEND", "")))
        return list(dict.fromkeys(metrics))
    return []


def build_counter_command(platform_name, tools, metrics, artifact_dir, workload_cmd):
    """Build the exhaustive counter collection pass for one workload/GPU."""
    if platform_name == "CUDA":
        # Do not limit launch count: --profile is deliberately exhaustive.
        return [
            tools["counter"], "--csv", "--set", CUDA_PROFILE_ANALYSIS_SET,
            "--metrics", ",".join(metrics),
        ] + workload_cmd
    if platform_name == "HIP":
        counter_dir = os.path.join(artifact_dir, "rocprof")
        ensure_dir(counter_dir)
        return [
            tools["counter"], "--pmc", ",".join(metrics), "--runtime-trace",
            "--output-format", "csv", "--output-config", "-d", counter_dir, "--",
        ] + workload_cmd
    raise ProfileUnavailableError(f"No profiler command is available for {platform_name}.")


def build_trace_command(platform_name, tools, artifact_dir, workload_cmd):
    """Build a separate timeline trace pass; vendor counter passes are not nested."""
    if platform_name == "CUDA":
        trace_prefix = os.path.join(artifact_dir, "trace")
        return [
            # Nsight Systems exposes these options in the short form across
            # the installed 2024.x builds.  The long forms can be parsed as
            # ambiguous abbreviations because related options such as
            # --trace-fork-before-exec and --samples-per-backtrace exist.
            tools["trace"], "profile", "-t", "cuda,nvtx,osrt", "-s", "none",
            "--cpuctxsw=none", "--force-overwrite=true", "--output", trace_prefix,
        ] + workload_cmd
    if platform_name == "HIP":
        # The counter pass above includes the runtime trace, so a second pass is unnecessary.
        return []
    return []


def collect_trace_artifacts(artifact_dir):
    patterns = ("*.nsys-rep", "*.qdrep", "*.qdstrm", "*.sqlite", "*.pftrace", "*trace*.csv")
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(artifact_dir, "**", pattern), recursive=True))
    return sorted(dict.fromkeys(files))


def import_nsys_qdstrm(artifact_dir):
    """Import raw Nsight Systems streams when nsys did not create a report.

    Some ARM64 Nsight Systems packages collect a valid .qdstrm file but cannot
    launch their bundled importer automatically.  Preserve the raw stream and
    invoke the package's importer explicitly so --profile still delivers the
    portable .nsys-rep artifact promised by the CLI.
    """
    streams = sorted(glob.glob(os.path.join(artifact_dir, "*.qdstrm")))
    if not streams:
        return [], "no Nsight Systems stream found"

    importers = sorted(glob.glob("/usr/lib/nsight-systems/host-*/QdstrmImporter"))
    if not importers:
        return [], "Nsight Systems importer was not found"

    imported = []
    errors = []
    for stream in streams:
        report = os.path.splitext(stream)[0] + ".nsys-rep"
        if os.path.isfile(report):
            imported.append(report)
            continue
        result = subprocess.run(
            [importers[0], "--input-file", stream, "--output-file", report, "--force-overwrite"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False,
        )
        if result.returncode or not os.path.isfile(report):
            detail = (result.stderr or result.stdout or f"exit code {result.returncode}").strip()
            errors.append(f"{os.path.basename(stream)}: {detail}")
        else:
            imported.append(report)
    return imported, "; ".join(errors)


def shell_join(cmd):
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_test(test_name, gpu_ids, duration, mem_pct, platform, run_dir, profile=False, verify=False, inject_error=False, ras_before=None):
    global ACTIVE_PROCS
    config = TEST_REGISTRY[test_name]
    binary = os.path.join(BUILD_DIR, config["bin"])
    procs = []
    
    for gpu in gpu_ids:
        workload_cmd = [binary, str(gpu), str(duration), str(mem_pct)] + config["args"]
        cmd = list(workload_cmd)
       
        if verify:
            workload_cmd.append("--verify")
            cmd.append("--verify")
        if inject_error:
            workload_cmd.append("--inject_error")
            cmd.append("--inject_error")

        profiler = None
        profile_tools = {}
        profile_metrics = []
        artifact_dir = ""
        manifest_path = ""
        trace_cmd = []

        # 2. Inject Hardware Profilers
        if profile:
            profile_tools = discover_profile_tools(platform)
            validate_profile_tools(platform, profile_tools)
            requested_metrics = get_profile_metrics(platform)
            profile_metrics, skipped_metrics, metric_validation = filter_supported_profile_metrics(
                platform, profile_tools, gpu, requested_metrics,
            )
            artifact_dir, manifest_path, _ = create_profile_manifest(
                run_dir, test_name, gpu, platform, profile_tools, requested_metrics, workload_cmd,
                supported_metrics=profile_metrics,
                skipped_metrics=skipped_metrics, metric_validation=metric_validation,
                ras_before=(ras_before or {}).get(gpu),
            )
            profiler = platform
            log(f"Collecting exhaustive {platform} counters for {test_name} on GPU {gpu}...")
            cmd = build_counter_command(platform, profile_tools, profile_metrics, artifact_dir, workload_cmd)
            trace_cmd = build_trace_command(platform, profile_tools, artifact_dir, workload_cmd)

        log(f"Launching {test_name} on GPU {gpu} (Alloc: {mem_pct}% VRAM)...")
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 universal_newlines=True, start_new_session=True)
        except OSError as exc:
            message = f"Could not launch {test_name} on GPU {gpu}: {exc}"
            tprint(f"[ERROR] {message}")
            procs.append({
                "gpu": gpu,
                "process": None,
                "test_name": test_name,
                "workload_cmd": shell_join(workload_cmd),
                "workload_argv": list(workload_cmd),
                "profile_cmd": shell_join(cmd) if profiler else "",
                "profiler": profiler,
                "profile_files": [],
                "profile_metrics": profile_metrics,
                "profile_tools": profile_tools,
                "profile_artifact_dir": artifact_dir,
                "profile_manifest": manifest_path,
                "trace_cmd": trace_cmd,
                "trace_files": [],
                "stdout": "",
                "stderr": message,
                "launch_error": message,
                "output_thread": None,
            })
            continue
        proc_info = {
            "gpu": gpu,
            "process": p,
            "test_name": test_name,
            "workload_cmd": shell_join(workload_cmd),
            "workload_argv": list(workload_cmd),
            "profile_cmd": shell_join(cmd) if profiler else "",
            "profiler": profiler,
            "profile_files": [],
            "profile_metrics": profile_metrics,
            "profile_tools": profile_tools,
            "profile_artifact_dir": artifact_dir,
            "profile_manifest": manifest_path,
            "trace_cmd": trace_cmd,
            "trace_files": [],
            "stdout": "",
            "stderr": "",
        }

        def drain_output(info=proc_info):
            try:
                info["stdout"], info["stderr"] = info["process"].communicate()
            except Exception as exc:
                info["stderr"] = f"Failed to collect process output: {exc}"

        output_thread = threading.Thread(target=drain_output, daemon=True)
        proc_info["output_thread"] = output_thread
        procs.append(proc_info)
        ACTIVE_PROCS.append(p)
        output_thread.start()

    return procs


def resolve_test_queue(test_arg):
    """Resolve a CLI test selector into concrete registry test names."""
    if test_arg == "all":
        return list(TEST_REGISTRY.keys())
    if test_arg in SUITES:
        return list(SUITES[test_arg])
    if test_arg in TEST_REGISTRY:
        return [test_arg]
    raise ValueError(f"Unknown test or suite: {test_arg}")


def parse_gpu_selection(gpu_arg, gpu_count):
    """Resolve and validate a CLI GPU selector."""
    if gpu_arg == "all":
        return list(range(gpu_count))

    gpu_ids = []
    for raw in str(gpu_arg).split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            gpu_id = int(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid GPU ID: {raw}") from exc
        if gpu_id < 0 or gpu_id >= gpu_count:
            raise ValueError(f"GPU ID {gpu_id} is outside the detected range 0-{max(0, gpu_count - 1)}")
        if gpu_id not in gpu_ids:
            gpu_ids.append(gpu_id)

    if not gpu_ids:
        raise ValueError("No GPU IDs selected")
    return gpu_ids


def validate_run_parameters(duration, mem_pct):
    """Validate shared runtime limits before launching kernels."""
    if int(duration) < 1:
        raise ValueError("--duration must be at least 1 second")
    if not 1 <= int(mem_pct) <= 99:
        raise ValueError("--mem must be between 1 and 99")


def cooldown_after_workload(duration):
    """Allow the GPU to cool between workload runs without unbounded delay."""
    cooldown_seconds = min(float(duration) / 10.0, 60.0)
    if cooldown_seconds <= 0:
        return 0.0
    log(f"Cooling down GPU(s) for {cooldown_seconds:.1f}s before the next workload...")
    time.sleep(cooldown_seconds)
    return cooldown_seconds


def parse_kernel_output(out, err, returncode):
    throughput = "N/A"
    unit = ""
    status = "PASS"
    had_error = False
    pantheon_lines = []

    skipped_reason = None
    if returncode != 0:
        throughput = 0.0
        unit = "ERR"
        status = "FAIL"
        had_error = True

    if out:
        for line in out.split('\n'):
            if "Throughput:" in line:
                try:
                    raw_val = line.split("Throughput:")[1].strip()
                    parts = raw_val.split(' ')
                    throughput = float(parts[0])
                    if len(parts) > 1:
                        unit = parts[1]
                except:
                    pass
            elif "[SDC FAULT]" in line:
                throughput = 0.0
                unit = "ERR"
                status = "FAIL"
                had_error = True
                pantheon_lines.append(line.strip())
            elif line.strip().startswith("Verification:"):
                # A workload reporting FAIL must be recorded as failed even if it
                # exits 0. Previously only [SDC FAULT] and a non-zero exit marked
                # a run bad, so this line was decorative: a verdict of FAIL with a
                # clean exit was filed as PASS, throughput and all.
                verdict = line.split("Verification:", 1)[1].strip().upper()
                if verdict.startswith("FAIL"):
                    throughput = 0.0
                    unit = "ERR"
                    status = "FAIL"
                    had_error = True
                pantheon_lines.append(line.strip())
            elif "[PANTHEON]" in line or line.strip().startswith("->"):
                if "Skipping" in line:
                    # The kernel decided at runtime that it cannot run here --
                    # P2P between GPUs with no link between them, for example.
                    # It still prints Throughput: 0.0, which would otherwise be
                    # recorded as a real measurement and be indistinguishable
                    # from catastrophically slow hardware.
                    skipped_reason = line.strip()
                pantheon_lines.append(line.strip())

    if skipped_reason and not had_error:
        unit = "SKIP"
        status = "SKIP"

    if err and returncode != 0:
        pantheon_lines.append(err.strip())

    return throughput, unit, status, had_error, pantheon_lines


def throughput_variance_percent(out):
    """Return coefficient of variation for repeated workload throughput samples."""
    samples = []
    for line in (out or "").splitlines():
        if "Throughput:" not in line:
            continue
        try:
            samples.append(float(line.split("Throughput:", 1)[1].strip().split()[0]))
        except (ValueError, IndexError):
            continue
    if len(samples) < 2 or np.mean(samples) == 0:
        return "N/A"
    return round(float(np.std(samples) / abs(np.mean(samples)) * 100.0), 2)


def build_result_row(test_name, gpu, duration, mem_pct, throughput, unit, stats, status="PASS", commands=None, profile_commands=None, profile_files=None, counter_summary=None, throughput_variance="N/A"):
    eff = 0
    # avg_pwr is "N/A" when the card exposed no power sensor, so this cannot
    # assume a number. Comparing a string to 0 raises in Python 3.
    avg_pwr = stats.get("avg_pwr", 0)
    if not isinstance(avg_pwr, (int, float)):
        avg_pwr = 0
    if throughput != "N/A" and avg_pwr > 0:
        if unit == "GB/s":
            eff = round((throughput * 1024) / avg_pwr, 2)
        elif unit == "TFLOPS":
            eff = round((throughput * 1000) / avg_pwr, 2)
        else:
            eff = round(throughput / avg_pwr, 2)

    row = {
        "Test Name": test_name,
        "Version": PANTHEON_VERSION,
        "Description": TEST_REGISTRY[test_name]["desc"] if test_name in TEST_REGISTRY else "Tuning workload mix",
        "GPU ID": gpu,
        "Duration (s)": duration,
        "Mem Usage (%)": mem_pct,
        "Score": throughput,
        "Unit": unit,
        "Throughput Variance (%)": throughput_variance,
        "Avg Temp (C)": stats.get("avg_temp", 0),
        "Max Temp (C)": stats.get("max_temp", 0),
        "Avg Mem Temp (C)": stats.get("avg_mem_temp", 0),
        "Max Mem Temp (C)": stats.get("max_mem_temp", 0),
        "Avg Power (W)": stats.get("avg_pwr", 0),
        "Max Power (W)": stats.get("max_pwr", 0),
        "Avg Clock(MHz)": stats.get("avg_clk", 0),
        "Min Clock (MHz)": stats.get("min_clk", 0),
        "Max Clock (MHz)": stats.get("max_clk", 0),
        "Avg GPU Util (%)": stats.get("avg_gpu_util", 0),
        "Max GPU Util (%)": stats.get("max_gpu_util", 0),
        "Peak Memory (MiB)": stats.get("peak_mem_used", 0),
        "Memory Total (MiB)": stats.get("mem_total", 0),
        "Energy (Wh)": stats.get("energy_wh", 0),
        "Thermal Rise (C)": stats.get("thermal_rise", 0),
        "Throttle Time (s)": stats.get("throttle_time", 0),
        "Efficiency": eff,
        "PCIe Gen": stats.get("pcie_gen", 0),
        "PCIe Width": stats.get("pcie_width", 0),
        "Limit Reason": stats.get("throttle_reason", "N/A"),
        "Max Fan (%)": stats.get("max_fan", 0),
        "Volts Core (mV)": stats.get("max_volts_core", 0),
        "Volts SoC (mV)": stats.get("max_volts_soc", 0)
    }
    if status != "PASS":
        row["Status"] = status
    if commands:
        row["Command Lines"] = " || ".join(commands or [])
    if profile_commands:
        row["Profiler Command Lines"] = " || ".join(profile_commands or [])
    if profile_files:
        row["Profiler Counter Files"] = " || ".join(profile_files or [])
    row.update(counter_summary or {})
    return row


def build_failure_row(test_name, gpu, duration, mem_pct, reason, stage):
    """Create a report row for a workload that could not be built or launched."""
    row = build_result_row(
        test_name, gpu, duration, mem_pct, 0.0, "ERR", {}, status="FAIL",
    )
    row["Failure Stage"] = stage
    row["Failure Reason"] = str(reason)
    row["RAS Status"] = "NOT RUN"
    row["RAS Error Delta"] = "Workload did not start"
    return row


def read_hardware_counter_file(path):
    """Read counter CSV while tolerating profiler and workload preamble output."""
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        # Nsight Compute prepends profiler and workload output before its CSV
        # header.  Find the real header so those lines cannot turn into a
        # malformed DataFrame and silently drop every collected counter.
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
        header_index = next(
            (
                index for index, line in enumerate(lines)
                if "Metric Name" in line and "Metric Value" in line
            ),
            None,
        )
        source = io.StringIO("".join(lines[header_index:])) if header_index is not None else path
        return pd.read_csv(source, comment="#")
    except:
        try:
            return pd.read_csv(path)
        except:
            return pd.DataFrame()


def summarize_hardware_counter_file(path):
    """Flatten profiler CSV output into summary columns for the main report."""
    df = read_hardware_counter_file(path)
    if df.empty:
        return {}

    summary = {}
    columns = {str(col).strip().lower(): col for col in df.columns}
    metric_col = next((columns[name] for name in ("metric name", "metric") if name in columns), None)
    value_col = next((columns[name] for name in ("metric value", "value") if name in columns), None)
    unit_col = next((columns[name] for name in ("metric unit", "unit") if name in columns), None)

    if metric_col and value_col:
        for _, item in df.iterrows():
            raw_metric_name = item.get(metric_col, "")
            if pd.isna(raw_metric_name):
                continue
            metric_name = str(raw_metric_name).strip()
            if not metric_name or metric_name.lower() == "nan":
                continue
            value = item.get(value_col, "")
            if isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
                value = int(value)
            unit = str(item.get(unit_col, "")).strip() if unit_col else ""
            key = f"Counter {metric_name}"
            summary[key] = f"{value} {unit}".strip()
        return summary

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        summary[f"Counter {col} Avg"] = round(float(values.mean()), 4)
        summary[f"Counter {col} Max"] = round(float(values.max()), 4)
    return summary


def profile_counter_table(paths):
    """Return one readable aggregate row per counter across profiler replay passes."""
    frames = [read_hardware_counter_file(path) for path in paths or []]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    columns = {str(col).strip().lower(): col for col in df.columns}
    metric_col = next((columns[name] for name in ("metric name", "metric") if name in columns), None)
    value_col = next((columns[name] for name in ("metric value", "value") if name in columns), None)
    unit_col = next((columns[name] for name in ("metric unit", "unit") if name in columns), None)
    if not metric_col or not value_col:
        return []

    rows = []
    for metric_name, group in df.groupby(metric_col, dropna=True, sort=True):
        if not str(metric_name).strip() or str(metric_name).strip().lower() == "nan":
            continue
        numeric_values = pd.to_numeric(group[value_col], errors="coerce").dropna()
        if numeric_values.empty:
            continue
        unit = ""
        if unit_col:
            unit_values = group[unit_col].dropna().astype(str)
            unit = unit_values.iloc[0].strip() if not unit_values.empty else ""
        median = float(numeric_values.median())
        value = int(median) if median.is_integer() else round(median, 4)
        rows.append({
            "metric": str(metric_name), "value": value, "unit": unit,
            "samples": int(len(numeric_values)),
            "min": float(numeric_values.min()), "max": float(numeric_values.max()),
        })
    return rows


def format_profile_value(value, unit=""):
    if value is None:
        return "N/A"
    elif isinstance(value, (int, np.integer)):
        rendered = f"{value:,}"
    elif isinstance(value, (float, np.floating)):
        rendered = f"{value:,.4f}".rstrip("0").rstrip(".")
    else:
        rendered = str(value)
    return f"{rendered} {unit}".strip()


def generate_profile_html(artifact_dir, workload_summary, counter_paths):
    """Write a portable, human-readable dashboard beside every profile artifact."""
    result = workload_summary.get("result", {})
    counters = profile_counter_table(counter_paths)
    counter_map = {item["metric"]: item for item in counters}
    cards = [
        ("Workload score", format_profile_value(result.get("Score", "N/A"), result.get("Unit", ""))),
        ("Average temperature", format_profile_value(result.get("Avg Temp (C)", "N/A"), "°C")),
        ("Peak temperature", format_profile_value(result.get("Max Temp (C)", "N/A"), "°C")),
        ("Average power", format_profile_value(result.get("Avg Power (W)", "N/A"), "W")),
    ]
    for metric, label in (
        ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "DRAM throughput"),
        ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "SM throughput"),
        ("sm__warps_active.avg.pct_of_peak_sustained_active", "Active warps"),
        ("smsp__issue_active.avg.pct_of_peak_sustained_active", "Scheduler issue active"),
    ):
        if metric in counter_map:
            item = counter_map[metric]
            cards.append((label, format_profile_value(item["value"], item["unit"])))

    card_html = "".join(
        f'<div class="card"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div></div>'
        for label, value in cards
    )
    rows_html = "".join(
        "<tr>"
        f"<td><code>{html.escape(item['metric'])}</code></td>"
        f"<td>{html.escape(format_profile_value(item['value'], item['unit']))}</td>"
        f"<td>{item['samples']}</td>"
        f"<td>{html.escape(format_profile_value(item['min'], item['unit']))} – {html.escape(format_profile_value(item['max'], item['unit']))}</td>"
        "</tr>"
        for item in counters
    ) or '<tr><td colspan="4">No machine-readable profiler counters were found.</td></tr>'
    ras_report = workload_summary.get("ras", {})
    ras_rows = ras_report.get("delta", {}).get("metrics", [])
    ras_sources = ras_report.get("after", {}).get("sources", {})
    ras_source_html = "".join(
        f"<li><code>{html.escape(str(name))}</code>: {html.escape(str(source.get('status', 'unavailable')))}"
        f"{(': ' + html.escape(str(source.get('detail')))) if source.get('detail') else ''}</li>"
        for name, source in sorted(ras_sources.items())
    ) or "<li>No RAS source was available.</li>"
    ras_rows_html = "".join(
        "<tr>"
        f"<td>{html.escape(str(item['source']))}</td>"
        f"<td><code>{html.escape(str(item['metric']))}</code></td>"
        f"<td>{html.escape(format_profile_value(item.get('before', 'N/A'), item.get('unit', '')))}</td>"
        f"<td>{html.escape(format_profile_value(item.get('after', 'N/A'), item.get('unit', '')))}</td>"
        f"<td>{html.escape(format_profile_value(item.get('delta', 'N/A'), item.get('unit', '')))}</td>"
        f"<td>{html.escape(str(item.get('status', 'unavailable')))}</td>"
        "</tr>"
        for item in ras_rows
    ) or '<tr><td colspan="6">No supported RAS counters were exposed by this GPU and host.</td></tr>'
    aer_events = ras_report.get("delta", {}).get("new_aer_events", [])
    aer_event_html = "".join(f"<li><code>{html.escape(str(event))}</code></li>" for event in aer_events) or "<li>None</li>"
    status = html.escape(str(workload_summary.get("status", result.get("Status", "PASS"))))
    test_name = html.escape(str(workload_summary.get("test", result.get("Test Name", "unknown"))))
    gpu_id = html.escape(str(workload_summary.get("gpu_id", result.get("GPU ID", "?"))))
    supported_count = len(counters)
    artifact_files = []
    for pattern in ("*.csv", "*.json", "*.log", "*.nsys-rep", "*.qdrep", "*.sqlite", "*.pftrace"):
        artifact_files.extend(glob.glob(os.path.join(artifact_dir, pattern)))
    artifact_links = " · ".join(
        f'<a href="{html.escape(os.path.basename(path))}">{html.escape(os.path.basename(path))}</a>'
        for path in sorted(dict.fromkeys(artifact_files))
        if os.path.basename(path) != "profile_summary.html"
    ) or "No profile artifacts were found."
    report = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pantheon profile — {test_name} GPU {gpu_id}</title><style>
:root{{color-scheme:dark;--bg:#0b1020;--panel:#151c31;--text:#e8edf8;--muted:#9aa8c5;--line:#2a3657;--good:#67e8a8}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at top right,#1d335e 0,transparent 34%),var(--bg);color:var(--text);font:15px/1.5 Segoe UI,Arial,sans-serif}}main{{max-width:1180px;margin:auto;padding:40px 24px 64px}}h1{{margin:0 0 4px}}h2{{font-size:18px}}.sub,.note{{color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:24px 0}}.card,section{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-top:16px}}.label{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}}.value{{color:var(--good);font-size:23px;font-weight:700;margin-top:4px;overflow-wrap:anywhere}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{text-align:left;padding:10px 8px;border-bottom:1px solid var(--line);vertical-align:top}}th{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em}}code{{color:#65d5ff;overflow-wrap:anywhere}}a{{color:#65d5ff}}</style></head><body><main>
<h1>Pantheon profile</h1><p class="sub">Workload: <code>{test_name}</code> · GPU {gpu_id} · Status: {status} · Duration: {html.escape(str(result.get('Duration (s)', 'N/A')))} seconds · Memory target: {html.escape(str(result.get('Mem Usage (%)', 'N/A')))}%</p>
<div class="grid">{card_html}</div>
<section><h2>Collected counters</h2><p class="note">{supported_count} unique machine-readable counters. Values are medians across profiler replay samples; min–max shows replay variation.</p><table><thead><tr><th>Metric</th><th>Median</th><th>Samples</th><th>Replay range</th></tr></thead><tbody>{rows_html}</tbody></table></section>
<section><h2>RAS and PCIe reliability</h2><p class="note">Counters are snapshots before and after this workload. Delta is the workload interval change; unavailable hardware is never shown as zero.</p><h3>Source availability</h3><ul>{ras_source_html}</ul><table><thead><tr><th>Source</th><th>Metric</th><th>Before</th><th>After</th><th>Delta</th><th>Status</th></tr></thead><tbody>{ras_rows_html}</tbody></table><h3>New Linux AER events</h3><ul>{aer_event_html}</ul></section>
<section><h2>Raw artifacts</h2><p>{artifact_links}</p></section>
</main></body></html>'''
    output_path = os.path.join(artifact_dir, "profile_summary.html")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    return output_path


def merge_counter_summaries(paths):
    merged = {}
    counts = {}
    for path in paths or []:
        for key, value in summarize_hardware_counter_file(path).items():
            if key not in merged:
                merged[key] = value
            else:
                counts[key] = counts.get(key, 1) + 1
                merged[f"{key} #{counts[key]}"] = value
    return merged


def process_timeout_seconds(test_name, duration, profile):
    """Wall-clock budget for kernel subprocesses (init + warmup + duration)."""
    if profile:
        return None
    if test_name == "tlb_avalanche":
        return duration * 4 + 1800
    if test_name == "cache_lat":
        return duration * 12 + 3600
    if test_name in ("ras_validator", "memory_read", "memory_read_agg",
                     "memory_write", "memory_write_agg"):
        return duration * 12 + 3600
    return duration * 2 + 300


def wait_for_processes(procs, timeout):
    start_wait = time.time()
    while True:
        all_done = True
        for proc_info in procs:
            p = proc_info["process"]
            if p is None:
                continue
            if p.poll() is None:
                all_done = False

        if all_done:
            return False

        if timeout is not None and time.time() - start_wait > timeout:
            tprint(f"\n[WATCHDOG] Test exceeded {timeout}s limit. Terminating...")
            for proc_info in procs:
                p = proc_info["process"]
                if p is None:
                    continue
                if p.poll() is None:
                    terminate_process_tree(p)
            return True

        time.sleep(1)


def run_profile_telemetry_pass(test_name, proc_infos, gpu_ids, duration, monitor, telemetry_dir):
    """Run a clean workload pass for profile throughput and hardware telemetry.

    Nsight Compute replays kernels and spends substantial time in setup and
    report generation. Its process-level stdout and telemetry therefore are
    not suitable for the benchmark row. A direct pass gives the profile the
    same workload-only measurements as a normal run while preserving the
    profiler artifacts from the counter and trace passes.
    """
    monitor.start_collection(gpu_ids, telemetry_dir, test_name)
    outputs = {}
    errors = []
    try:
        for proc_info in proc_infos:
            gpu = proc_info["gpu"]
            try:
                process = subprocess.Popen(
                    proc_info["workload_argv"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    start_new_session=True,
                )
                try:
                    stdout, stderr = process.communicate(
                        timeout=process_timeout_seconds(test_name, duration, False),
                    )
                except subprocess.TimeoutExpired:
                    terminate_process_tree(process)
                    stdout, stderr = process.communicate()
                    errors.append(f"Telemetry pass timed out for GPU {gpu}.")
                outputs[gpu] = (stdout or "", stderr or "", process.returncode)
                if process.returncode:
                    errors.append(f"Telemetry pass failed for GPU {gpu} (code {process.returncode}).")
            except OSError as exc:
                errors.append(f"Telemetry pass could not launch on GPU {gpu}: {exc}")
    finally:
        stats = monitor.stop_collection()
    return stats, outputs, errors


def execute_test(test_name, gpu_ids, duration, mem_pct, platform, run_dir, monitor, profile=False, verify=False, inject_error=False):
    """Run one workload on the selected GPUs."""
    global ACTIVE_PROCS
    ACTIVE_PROCS = []
    procs = []
    rows = []
    run_had_errors = False
    ras_before_by_gpu = {}
    reliability_dirs = {}

    telemetry_dir = run_dir
    if profile and len(gpu_ids) == 1:
        telemetry_dir = profile_artifact_dir(run_dir, test_name, gpu_ids[0])
        ensure_dir(telemetry_dir)
    for gpu in gpu_ids:
        reliability_dir = (
            profile_artifact_dir(run_dir, test_name, gpu)
            if profile else reliability_artifact_dir(run_dir, test_name, gpu)
        )
        ensure_dir(reliability_dir)
        reliability_dirs[gpu] = reliability_dir
        ras_before_by_gpu[gpu] = collect_ras_snapshot(platform, gpu)
    log(f"Collecting RAS and PCIe reliability snapshots for {test_name}...")
    monitor.start_collection(gpu_ids, telemetry_dir, test_name)
    try:
        procs.extend(run_test(
            test_name, gpu_ids, duration, mem_pct, platform, run_dir,
            profile, verify, inject_error, ras_before_by_gpu,
        ))

        # Nsight Compute legitimately replays a workload once per counter group.
        # A normal stress-test timeout would abort a complete --profile capture.
        wait_for_processes(procs, process_timeout_seconds(test_name, duration, profile))
    except KeyboardInterrupt:
        print("\n[PANTHEON] Interrupted by user.")
        for proc_info in procs:
            p = proc_info["process"]
            if p.poll() is None:
                terminate_process_tree(p)
        raise
    finally:
        hw_stats = monitor.stop_collection()

    telemetry_outputs = {}
    if profile:
        telemetry_infos = [item for item in procs if item.get("process") is not None and item["process"].returncode == 0]
        if telemetry_infos:
            telemetry_stats, telemetry_outputs, telemetry_errors = run_profile_telemetry_pass(
                test_name, telemetry_infos, gpu_ids, duration, monitor, telemetry_dir,
            )
            hw_stats = telemetry_stats
            if telemetry_errors:
                run_had_errors = True
                for detail in telemetry_errors:
                    tprint(f"[ERROR] {detail}")

    trial_status_by_gpu = {gpu: "PASS" for gpu in gpu_ids}
    parsed = []
    for proc_info in procs:
        gpu = proc_info["gpu"]
        p = proc_info["process"]
        component_name = proc_info["test_name"]
        proc_info["output_thread"].join() if proc_info.get("output_thread") else None
        profiler_out = proc_info.get("stdout", "")
        telemetry_sample = telemetry_outputs.get(gpu)
        if telemetry_sample is not None:
            out, err, returncode = telemetry_sample
            throughput, unit, status, had_error, pantheon_lines = parse_kernel_output(out, err, returncode)
        elif proc_info.get("launch_error"):
            out = ""
            err = proc_info["launch_error"]
            throughput, unit, status, had_error = 0.0, "ERR", "FAIL", True
            pantheon_lines = [err]
        else:
            out = profiler_out
            err = proc_info["stderr"]
            throughput, unit, status, had_error, pantheon_lines = parse_kernel_output(out, err, p.returncode)
        if had_error:
            run_had_errors = True
            trial_status_by_gpu[gpu] = "FAIL"
            exit_code = p.returncode if p is not None else "launch"
            tprint(f"[ERROR] GPU {gpu} Test Failed ({test_name}, code {exit_code}):")
        for line in pantheon_lines:
            if line:
                tprint(line)

        if p is not None and p.returncode == 0 and proc_info["profiler"]:
            if proc_info["profiler"] == "HIP":
                csv_path = extract_profiler_data(component_name, gpu, proc_info["profile_artifact_dir"])
                if csv_path:
                    proc_info["profile_files"].append(csv_path)
            elif proc_info["profiler"] == "CUDA":
                csv_path = os.path.join(
                    proc_info["profile_artifact_dir"],
                    "hardware_counters.csv",
                )
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write(profiler_out)
                proc_info["profile_files"].append(csv_path)
                print(f"[PANTHEON] Extracted NVIDIA hardware counters to: {csv_path}")

            if proc_info["trace_cmd"]:
                trace_log = os.path.join(proc_info["profile_artifact_dir"], "trace.log")
                log(f"Collecting {platform} timeline trace for {component_name} on GPU {gpu}...")
                trace_result = subprocess.run(
                    proc_info["trace_cmd"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False,
                )
                with open(trace_log, "w", encoding="utf-8") as handle:
                    handle.write(trace_result.stdout or "")
                    if trace_result.stderr:
                        handle.write("\n--- stderr ---\n")
                        handle.write(trace_result.stderr)
                if trace_result.returncode:
                    run_had_errors = True
                    trial_status_by_gpu[gpu] = "FAIL"
                    tprint(f"[ERROR] Timeline trace failed for GPU {gpu} (code {trace_result.returncode}).")
                proc_info["trace_files"] = collect_trace_artifacts(proc_info["profile_artifact_dir"])
                if not trace_result.returncode and not any(path.endswith(".nsys-rep") for path in proc_info["trace_files"]):
                    imported_reports, import_error = import_nsys_qdstrm(proc_info["profile_artifact_dir"])
                    if imported_reports:
                        proc_info["trace_files"] = collect_trace_artifacts(proc_info["profile_artifact_dir"])
                        tprint(f"[PANTHEON] Imported Nsight Systems trace report for GPU {gpu}.")
                    else:
                        run_had_errors = True
                        trial_status_by_gpu[gpu] = "FAIL"
                        detail = f": {import_error}" if import_error else ""
                        tprint(f"[ERROR] Timeline trace did not produce an Nsight Systems report for GPU {gpu}{detail}")

            update_profile_manifest(
                proc_info["profile_manifest"],
                status="complete" if trial_status_by_gpu[gpu] == "PASS" else "failed",
                artifacts={
                    "counter_files": proc_info["profile_files"],
                    "trace_files": proc_info["trace_files"],
                    "logs": [trace_log] if proc_info["trace_cmd"] else [],
                },
                commands={
                    "counter": proc_info["profile_cmd"],
                    "trace": shell_join(proc_info["trace_cmd"]) if proc_info["trace_cmd"] else "",
                },
            )

        elif proc_info["profiler"]:
            counter_log = os.path.join(proc_info["profile_artifact_dir"], "counter.log")
            with open(counter_log, "w", encoding="utf-8") as handle:
                handle.write(out or "")
                if err:
                    handle.write("\n--- stderr ---\n")
                    handle.write(err)
            run_had_errors = True
            trial_status_by_gpu[gpu] = "FAIL"
            update_profile_manifest(
                proc_info["profile_manifest"],
                status="failed",
                artifacts={"counter_files": [], "trace_files": [], "logs": [counter_log]},
                commands={"counter": proc_info["profile_cmd"], "trace": ""},
            )

        parsed.append((gpu, throughput, unit, status, throughput_variance_percent(out)))

    for gpu in gpu_ids:
        stats = hw_stats.get(gpu, {})
        gpu_result = next((item for item in parsed if item[0] == gpu), None)
        throughput = gpu_result[1] if gpu_result else 0.0
        unit = gpu_result[2] if gpu_result else ""
        throughput_variance = gpu_result[4] if gpu_result else "N/A"
        if trial_status_by_gpu[gpu] == "FAIL":
            throughput = 0.0
            unit = "ERR"
        gpu_proc_infos = [item for item in procs if item["gpu"] == gpu]
        commands = [item["workload_cmd"] for item in gpu_proc_infos]
        profile_commands = [item["profile_cmd"] for item in gpu_proc_infos if item["profile_cmd"]]
        profile_files = [path for item in gpu_proc_infos for path in item["profile_files"]]
        counter_summary = merge_counter_summaries(profile_files)

        row = build_result_row(
            test_name,
            gpu,
            duration,
            mem_pct,
            throughput,
            unit,
            stats,
            status=trial_status_by_gpu[gpu],
            commands=commands,
            profile_commands=profile_commands,
            profile_files=profile_files,
            counter_summary=counter_summary,
            throughput_variance=throughput_variance,
        )
        launch_errors = [item["launch_error"] for item in gpu_proc_infos if item.get("launch_error")]
        if launch_errors:
            row["Failure Stage"] = "launch"
            row["Failure Reason"] = " || ".join(launch_errors)
        ras_after = collect_ras_snapshot(platform, gpu)
        ras_delta = diff_ras_snapshots(ras_before_by_gpu.get(gpu, {}), ras_after)
        ras_report = {
            "before": ras_before_by_gpu.get(gpu, {}),
            "after": ras_after,
            "delta": ras_delta,
        }
        ras_path = os.path.join(reliability_dirs[gpu], "ras.json")
        with open(ras_path, "w", encoding="utf-8") as handle:
            json.dump(ras_report, handle, indent=2, cls=NumpyEncoder, allow_nan=False)
        ras_summary = summarize_ras_delta(ras_delta, ras_before_by_gpu.get(gpu, {}), ras_after)
        row["RAS Status"] = ras_summary["status"]
        row["RAS Error Delta"] = " || ".join(ras_summary["details"]) or "None"
        row["RAS Report"] = ras_path
        if ras_summary["status"] != "CLEAN":
            tprint(f"[RAS] GPU {gpu} {ras_summary['status']}: {row['RAS Error Delta']}")

        if profile and gpu_proc_infos:
            proc_info = gpu_proc_infos[0]
            row["Profile Artifact Directory"] = proc_info["profile_artifact_dir"]
            row["Profile Manifest"] = proc_info["profile_manifest"]
            row["Profiler Trace Files"] = " || ".join(proc_info["trace_files"])
            workload_summary = {
                "test": test_name,
                "gpu_id": gpu,
                "status": trial_status_by_gpu[gpu],
                "result": row,
                "counter_summary": counter_summary,
                "ras": ras_report,
            }
            with open(
                os.path.join(proc_info["profile_artifact_dir"], "summary.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(workload_summary, handle, indent=2, cls=NumpyEncoder, allow_nan=False)
            html_summary = generate_profile_html(
                proc_info["profile_artifact_dir"], workload_summary, profile_files,
            )
            row["Profile HTML Summary"] = html_summary
            update_profile_manifest(
                proc_info["profile_manifest"],
                html_summary=html_summary,
                ras_before=ras_before_by_gpu.get(gpu, {}),
                ras_after=ras_after,
                ras_report=ras_path,
            )
        rows.append(row)
        tprint(f"[RESULT] GPU {gpu} | {row['Score']} {row['Unit']} | {row['Avg Temp (C)']}C Avg / {row['Max Temp (C)']}C Max | {row['Avg Power (W)']}W Avg / {row['Max Power (W)']}W Max")

    return rows, run_had_errors


def main():
    run_had_errors = False
    parser = argparse.ArgumentParser(description="PANTHEON: Universal GPU Stress Suite")
    parser.add_argument("--version", action="version", version=f"%(prog)s {PANTHEON_VERSION}")
    parser.add_argument("--test", type=str, default="all", help="Target: specific test name, suite name (core, memory, interconnect, inference, baseline), or 'all'")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds per test")
    parser.add_argument("--gpu", type=str, default="all", help="Comma separated list of GPU IDs (e.g. 0,1) or 'all'")
    parser.add_argument("--mem", type=int, default=99, help="Percentage of free VRAM to use (Default: 99)")
    parser.add_argument("--profile", action="store_true", help="Wrap execution in ncu/rocprof to dump hardware counters")
    verify_group = parser.add_mutually_exclusive_group()
    verify_group.add_argument(
        "--verify", dest="verify", action="store_true", default=True,
        help="Run validation after each workload to check for Silent Data Corruption (default)",
    )
    verify_group.add_argument(
        "--skip_verify", dest="verify", action="store_false",
        help="Skip validation after each workload",
    )
    parser.add_argument("--inject_error", action="store_true", help="Inject a hardware fault to test SDC verification")
    parser.add_argument("--platform", choices=["auto", "cuda", "hip", "mock"], default="auto", help="Compiler backend override")
    args = parser.parse_args()

    try:
        validate_run_parameters(args.duration, args.mem)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    if args.inject_error and not args.verify:
        print("[ERROR] --inject_error requires --verify")
        sys.exit(1)

    try:
        queue = resolve_test_queue(args.test)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        print(f"Available suites: {', '.join(SUITES.keys())}")
        sys.exit(1)

    # --- Setup ---
    platform = detect_platform(args.platform)
    if platform == "UNKNOWN": sys.exit("Error: No compiler found.")

    monitor = HardwareMonitor(platform)

    # --- GPU Discovery ---
    avail_count = monitor.get_gpu_count()
    try:
        target_gpus = parse_gpu_selection(args.gpu, avail_count)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    queue, skipped_tests = filter_gpu_compatible_tests(queue, len(target_gpus))
    for test_name, required_gpu_count in skipped_tests:
        log(
            f"Skipping {test_name}: requires at least {required_gpu_count} "
            f"selected GPUs; found {len(target_gpus)}."
        )

    if args.profile:
        try:
            validate_profile_tools(platform, discover_profile_tools(platform))
        except ProfileUnavailableError as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)

    build_failures = build_kernels(platform)

    # --- NEW: Display GPU Information ---
    print("\n" + "="*60)
    print(f"PANTHEON SYSTEM DETECTED (v{PANTHEON_VERSION})")
    print("="*60)
    gpu_info = get_gpu_static_info(platform)
    if not gpu_info:
        print(f"Platform: {platform} (No detailed GPU info available via SMI)")
    else:
        for g in gpu_info:
            print(f"GPU {g['id']}: [{g['manufacturer']}] {g['name']} | {g['memory_total']} VRAM | UUID: {g['uuid']}")
    print("="*60 + "\n")

    # --- Result Folder Setup ---
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(RESULTS_BASE_DIR, timestamp_str)
    ensure_dir(run_dir)
    ensure_dir(DATABASE_DIR)
    
    log(f"Run ID: {timestamp_str}")
    incremental_snapshot = get_system_snapshot(platform)
    incremental_sequence = 0

    if args.test in SUITES:
        log(f"Running test suite: {args.test.upper()} ({len(queue)} tests)")
    if args.profile:
        log("--profile collects exhaustive counters and traces; selected GPUs run serially.")

    # --- Main Loop ---
    final_results = []

    for test in queue:
        log(f"--- STARTING TEST: {test.upper()} ---")
        if test in build_failures:
            reason = build_failures[test]
            for gpu in target_gpus:
                final_results.append(build_failure_row(
                    test, gpu, args.duration, args.mem, reason, "compile",
                ))
            run_had_errors = True
            incremental_sequence += 1
            write_incremental_workload_reports(
                incremental_snapshot, final_results[-len(target_gpus):],
                timestamp_str, incremental_sequence,
            )
            log(f"--- SKIPPED TEST: {test.upper()} (compile failure) ---\n")
            continue

        gpu_batches = [[gpu] for gpu in target_gpus] if args.profile else [target_gpus]
        for gpu_batch in gpu_batches:
            try:
                rows, had_errors = execute_test(
                    test, gpu_batch, args.duration, args.mem, platform, run_dir,
                    monitor, args.profile, args.verify, args.inject_error,
                )
            except KeyboardInterrupt:
                sys.exit(0)
            except Exception as exc:
                reason = f"Unhandled workload setup error: {exc}"
                tprint(f"[ERROR] {test} on GPU(s) {gpu_batch}: {reason}")
                rows = [
                    build_failure_row(
                        test, gpu, args.duration, args.mem, reason, "launch",
                    )
                    for gpu in gpu_batch
                ]
                had_errors = True
            run_had_errors = run_had_errors or had_errors
            final_results.extend(rows)
            incremental_sequence += 1
            write_incremental_workload_reports(
                incremental_snapshot, rows, timestamp_str, incremental_sequence,
            )
            cooldown_after_workload(args.duration)

        log(f"--- FINISHED TEST: {test.upper()} ---\n")

    # --- Report Generation ---
    df = pd.DataFrame(final_results)
    
    print("\n" + "="*80)
    print("FINAL SUMMARY REPORT")
    print("="*80)
    
    # CONSOLE: Drop the noisy columns
    cols_to_hide = [
        "Description", "Max Fan (%)", "PCIe Width", "PCIe Gen",
        "Volts Core (mV)", "Volts SoC (mV)", "Limit Reason", "Efficiency",
        "Version", "Command Lines", "Profiler Command Lines",
        "Profiler Counter Files",
    ]
    
    # CONSOLE: Abbreviate headers so the table fits horizontally
    compact_headers = {
        "Duration (s)": "Dur(s)",
        "Mem Usage (%)": "Mem(%)",
        "Avg Temp (C)": "AvgT",
        "Max Temp (C)": "MaxT",
        "Avg Mem Temp (C)": "AvgMemT",
        "Max Mem Temp (C)": "MaxMemT",
        "Avg Power (W)": "AvgPwr",
        "Max Power (W)": "MaxPwr",
        "Avg Clock(MHz)": "Clk(MHz)"
    }
    
    console_df = df.drop(columns=cols_to_hide, errors='ignore').rename(columns=compact_headers)
    
    # Print the compact version to the terminal (rounded to 1 decimal place to save space)
    table_str = console_df.round(1).to_string(index=False)
    # ANSI Color Codes
    RED = '\033[91m'
    RESET = '\033[0m'
    # Print line by line, painting the row red if it contains the "ERR" unit flag
    for line in table_str.split('\n'):
        if " ERR " in line:
            print(f"{RED}{line}{RESET}")
        else:
            print(line)


    print("="*80)

    # FILES: Save EVERYTHING (including full names and hidden metrics)
    csv_path = os.path.join(run_dir, "summary.csv")
    xlsx_path = os.path.join(run_dir, "summary.xlsx")
    df.to_csv(csv_path, index=False)
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as exc:
        print(f"[PANTHEON] Warning: Could not write Excel report {xlsx_path}: {exc}")

    # 2. Save Full Database Snapshot (JSON) in database/ folder
    log("Generating Database Snapshot...")
    full_snapshot = get_system_snapshot(platform)
    full_snapshot["test_results"] = final_results
    
    db_file = os.path.join(DATABASE_DIR, f"pantheon_report_{timestamp_str}.json")
    
    with open(db_file, "w") as f:
        # Added cls=NumpyEncoder to automatically convert int64/float64 types
        json.dump(full_snapshot, f, indent=4, cls=NumpyEncoder)
    
    log(f"Snapshot saved to: {db_file}")

    # --- SYSTEM EXIT HOOK ---
    if run_had_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
