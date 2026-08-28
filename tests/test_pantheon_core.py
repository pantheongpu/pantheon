import os
import re
import json
from pathlib import Path

import pytest

import pantheon


def test_detect_platform_respects_explicit_override():
    assert pantheon.detect_platform("mock") == "MOCK"
    assert pantheon.detect_platform("cuda") == "CUDA"
    assert pantheon.detect_platform("hip") == "HIP"


def test_detect_platform_uses_mock_env(monkeypatch):
    monkeypatch.setenv("PANTHEON_MOCK", "1")
    assert pantheon.detect_platform() == "MOCK"


def test_detect_platform_prefers_gpu_matching_compiler(monkeypatch):
    monkeypatch.delenv("PANTHEON_MOCK", raising=False)

    def fake_which(name):
        available = {"nvcc", "hipcc", "nvidia-smi"}
        return f"/usr/bin/{name}" if name in available else None

    monkeypatch.setattr(pantheon.shutil, "which", fake_which)
    assert pantheon.detect_platform() == "CUDA"


def test_detect_platform_falls_back_to_mock_when_only_gpp(monkeypatch):
    monkeypatch.delenv("PANTHEON_MOCK", raising=False)

    def fake_which(name):
        return "/usr/bin/g++" if name == "g++" else None

    monkeypatch.setattr(pantheon.shutil, "which", fake_which)
    assert pantheon.detect_platform() == "MOCK"


def test_multi_gpu_workloads_are_skipped_when_not_available():
    queue = ["all_reduce", "p2p_thrasher", "memory_write"]

    runnable, skipped = pantheon.filter_gpu_compatible_tests(queue, 1)

    assert runnable == ["memory_write"]
    assert skipped == [("all_reduce", 2), ("p2p_thrasher", 2)]

    runnable, skipped = pantheon.filter_gpu_compatible_tests(queue, 2)
    assert runnable == queue
    assert skipped == []


def test_ras_value_preserves_unsupported_values_as_unavailable():
    assert pantheon.ras_value("[N/A]") == {
        "status": "unsupported", "value": None, "unit": "count",
    }
    assert pantheon.ras_value("[Not Supported]") == {
        "status": "unsupported", "value": None, "unit": "count",
    }


def test_ras_snapshot_delta_uses_counter_difference_and_retains_status():
    before = {
        "sources": {
            "vendor_ras": {
                "metrics": {"ecc.correctable": {"status": "supported", "value": 4, "unit": "count"}},
            },
            "linux_pcie_aer": {"metrics": {}, "events": ["old event"]},
        },
    }
    after = {
        "sources": {
            "vendor_ras": {"metrics": {"ecc.correctable": {"status": "supported", "value": 6, "unit": "count"}}},
            "linux_pcie_aer": {"metrics": {}, "events": ["old event", "new poison event"]},
        },
    }

    report = pantheon.diff_ras_snapshots(before, after)

    assert report["metrics"] == [{
        "source": "vendor_ras", "metric": "ecc.correctable", "before": 4,
        "after": 6, "delta": 2, "unit": "count", "status": "supported",
    }]
    assert report["new_aer_events"] == ["new poison event"]


def test_ras_delta_summary_distinguishes_clean_warning_error_and_unavailable():
    supported = {"sources": {"vendor_ras": {"status": "supported"}}}
    assert pantheon.summarize_ras_delta({"metrics": [], "new_aer_events": []}, supported, supported)["status"] == "CLEAN"
    warning = pantheon.summarize_ras_delta({
        "metrics": [{"source": "vendor_ras", "metric": "ecc.correctable", "delta": 1, "status": "supported"}],
        "new_aer_events": [],
    }, supported, supported)
    assert warning["status"] == "WARNING"
    error = pantheon.summarize_ras_delta({
        "metrics": [{"source": "vendor_ras", "metric": "ecc.uncorrected", "delta": 1, "status": "supported"}],
        "new_aer_events": [],
    }, supported, supported)
    assert error["status"] == "ERROR"
    assert pantheon.summarize_ras_delta({"metrics": [], "new_aer_events": []}, {}, {})["status"] == "UNAVAILABLE"


def test_nvidia_ras_snapshot_collects_detailed_ecc_and_preserves_na(monkeypatch):
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        fields = command[3].split("=", 1)[1].split(",")
        return type("Result", (), {
            "returncode": 0,
            "stdout": ", ".join(["[N/A]"] + ["0"] * (len(fields) - 1)) + "\n",
            "stderr": "",
        })()

    monkeypatch.setattr(pantheon.subprocess, "run", fake_run)
    monkeypatch.setattr(pantheon, "pynvml", None)

    report = pantheon.nvidia_ras_snapshot(0)

    assert "ecc.errors.corrected.volatile.dram" in commands[0][3]
    assert report["metrics"]["ecc.mode.current"]["status"] == "unsupported"
    assert report["metrics"]["ecc.errors.corrected.volatile.dram"]["value"] == 0
    assert report["metrics"]["pcie.lcrc"]["status"] == "unavailable"


def test_amd_cper_snapshot_reports_missing_tool_without_failing(monkeypatch):
    monkeypatch.setattr(pantheon, "find_tool", lambda _name: None)

    report = pantheon.amd_cper_snapshot(0)

    assert report["status"] == "unavailable"
    assert "amd-smi" in report["detail"]


def test_cuda_tool_discovery_adds_installed_toolkit_bins(monkeypatch, tmp_path):
    cuda_bin = tmp_path / "cuda-13.2" / "bin"
    cuda_bin.mkdir(parents=True)
    ncu = cuda_bin / "ncu"
    ncu.write_text("", encoding="utf-8")
    ncu.chmod(0o755)

    monkeypatch.setattr(pantheon, "cuda_bin_directories", lambda: [str(cuda_bin)])
    monkeypatch.setattr(pantheon, "rocm_bin_directories", lambda: [])
    monkeypatch.setenv("PATH", "")

    assert pantheon.find_tool("ncu") == str(ncu)


def test_mock_gpu_static_info_is_synthetic():
    info = pantheon.get_gpu_static_info("MOCK")
    assert info == [{
        "id": 0,
        "type": "MOCK",
        "manufacturer": "Pantheon",
        "name": "Mock GPU",
        "memory_total": "2048 MB",
        "driver_version": "Mock",
        "power_limit": "N/A",
        "uuid": "MOCK-GPU-0",
        "serial": "Mock",
    }]


def test_gpu_static_info_respects_selected_backend(monkeypatch):
    calls = []

    def fake_which(name):
        return f"/usr/bin/{name}" if name in {"nvidia-smi", "rocm-smi"} else None

    def fake_check_output(cmd, **_kwargs):
        calls.append(cmd)
        if cmd[0] == "nvidia-smi":
            raise AssertionError("NVIDIA discovery should not run for HIP")
        return json.dumps({
            "card0": {
                "Card Series": "AMD Test GPU",
                "VRAM Total Memory (B)": str(16 * 1024 * 1024),
                "Unique ID": "AMD-0",
            },
        })

    monkeypatch.setattr(pantheon.shutil, "which", fake_which)
    monkeypatch.setattr(pantheon.subprocess, "check_output", fake_check_output)

    info = pantheon.get_gpu_static_info("HIP")

    assert info[0]["type"] == "AMD"
    assert info[0]["name"] == "AMD Test GPU"
    assert all(call[0] == "rocm-smi" for call in calls)


def test_all_registered_binaries_have_sources():
    kernel_sources = {
        path.stem
        for path in Path(pantheon.KERNEL_DIR).glob("*/*.cpp")
        if "common" not in path.parts
    }

    missing = sorted(
        {config["bin"] for config in pantheon.TEST_REGISTRY.values()} - kernel_sources
    )
    assert missing == []


def test_memory_workloads_support_a_fault_map():
    # Console fault output is capped, so a run reports an accurate error count
    # but only a handful of error locations. Building a fault map needs every
    # failing address, which is what --fault_map provides.
    from pathlib import Path

    root = Path(os.path.join(pantheon.BASE_DIR, "kernels"))
    for source in (
        "memory_read/memory_read.cpp",
        "memory_write/memory_write.cpp",
        "atomic_virus/atomic_virus.cpp",
        "ras_validator/ras_validator.cpp",
        "memory_retention/memory_retention.cpp",
    ):
        text = (root / source).read_text(encoding="utf-8")
        assert "--fault_map" in text, source
        assert "pantheon_fault_log_append" in text, source


def test_ras_validator_records_addresses_not_only_a_count():
    from pathlib import Path

    source = Path(os.path.join(pantheon.BASE_DIR, "kernels/ras_validator/ras_validator.cpp"))
    text = source.read_text(encoding="utf-8")

    # A bare error tally cannot locate a defective cell.
    assert text.count("pantheon_fault_log_append") >= 4


def test_memory_retention_is_registered_and_holds_the_payload_in_process():
    from pathlib import Path

    assert "memory_retention" in pantheon.TEST_REGISTRY
    assert "memory_retention" in pantheon.SUITES["memory"]

    source = Path(os.path.join(pantheon.BASE_DIR, "kernels/memory_retention/memory_retention.cpp"))
    text = source.read_text(encoding="utf-8")

    # Device memory is freed when the process exits, so the idle period cannot
    # be orchestrated from outside; it has to happen inside this binary.
    assert "--retention_delay" in text
    assert "sleep_for" in text
    # Reading through the cache would return what we wrote, not what the cell held.
    assert "load_nt" in text and "store_nt" in text


def test_mock_atomic_add_returns_the_previous_value():
    # Real CUDA/HIP atomicAdd returns the old value; the mock returning void
    # made the atomic-append idiom impossible to express.
    from pathlib import Path

    mock = Path(os.path.join(pantheon.BASE_DIR, "kernels/common/mock_gpu.h")).read_text(encoding="utf-8")

    assert "unsigned int atomicAdd(unsigned int* address, unsigned int val)" in mock
    assert "return old;" in mock


def test_memory_agg_names_are_aliases_not_duplicate_kernels():
    # memory_read_agg and memory_write_agg were byte-identical to their base
    # workloads apart from one data-background constant, which --init_pattern
    # already parameterises. They stay as registry aliases so existing results
    # and leaderboard history keep resolving.
    from pathlib import Path

    assert pantheon.TEST_REGISTRY["memory_read_agg"]["bin"] == "memory_read"
    assert pantheon.TEST_REGISTRY["memory_read_agg"]["args"] == ["--init_pattern", "rail_to_rail"]
    assert pantheon.TEST_REGISTRY["memory_write_agg"]["bin"] == "memory_write"
    assert pantheon.TEST_REGISTRY["memory_write_agg"]["args"] == ["--init_pattern", "crosstalk"]

    kernels = Path(pantheon.KERNEL_DIR)
    assert not (kernels / "memory_read/memory_read_agg.cpp").exists()
    assert not (kernels / "memory_write/memory_write_agg.cpp").exists()


def test_memory_workload_defaults_preserve_their_historical_background():
    # The two workloads genuinely defaulted to different backgrounds. Changing
    # either default would silently move published scores, so each keeps its own.
    from pathlib import Path

    kernels = Path(pantheon.KERNEL_DIR)
    read = (kernels / "memory_read/memory_read.cpp").read_text(encoding="utf-8")
    write = (kernels / "memory_write/memory_write.cpp").read_text(encoding="utf-8")

    assert "int init_pattern = 2;" in read, "memory_read defaulted to crosstalk"
    assert "int init_pattern = 3;" in write, "memory_write defaulted to rail-to-rail"


def test_data_backgrounds_come_from_one_shared_selector():
    # Duplicated pattern constants are how the two workloads drifted into
    # separate binaries in the first place.
    from pathlib import Path

    common = Path(os.path.join(pantheon.BASE_DIR, "kernels/common/common.h")).read_text(encoding="utf-8")
    assert "pantheon_pattern_base" in common
    assert "pantheon_pattern_value" in common
    assert "0xFFFFFFFFu" in common and "0xAAAAAAAAu" in common

    for source in ("memory_read/memory_read.cpp", "memory_write/memory_write.cpp"):
        text = (Path(pantheon.KERNEL_DIR) / source).read_text(encoding="utf-8")
        assert "pantheon_pattern_value" in text, source
        # The literals must not be reintroduced locally.
        assert "0xAAAAAAAA : 0x55555555" not in text, source
        assert "0x00000000 : 0xFFFFFFFF" not in text, source


def test_all_suite_entries_are_registered_tests():
    registered = set(pantheon.TEST_REGISTRY)
    missing = {
        suite_name: sorted(set(test_names) - registered)
        for suite_name, test_names in pantheon.SUITES.items()
        if set(test_names) - registered
    }
    assert missing == {}


def test_tunable_kernels_normalize_launch_knobs():
    missing = []
    for source in Path(pantheon.KERNEL_DIR).glob("*/*.cpp"):
        text = source.read_text(encoding="utf-8")
        has_standard_knobs = all(
            knob in text
            for knob in (
                "int block_size",
                "int grid_size",
                "int kernel_loops",
                "int warmup_iters",
                "int sync_mode",
            )
        )
        if has_standard_knobs and "normalize_kernel_launch_config(" not in text:
            missing.append(str(source.relative_to(pantheon.BASE_DIR)))

    assert missing == []


def test_cuda_shim_covers_kernel_copy_kinds():
    common = Path(pantheon.KERNEL_DIR, "common", "common.h").read_text(encoding="utf-8")

    assert "#define hipMemcpyHostToDevice cudaMemcpyHostToDevice" in common
    assert "#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost" in common
    assert "#define hipMemcpyDeviceToDevice cudaMemcpyDeviceToDevice" in common


def test_scheduler_normalizes_before_deriving_stream_count():
    source = Path(
        pantheon.KERNEL_DIR,
        "scheduler_virus",
        "scheduler_virus.cpp",
    ).read_text(encoding="utf-8")

    assert source.index("normalize_kernel_launch_config(") < source.index("int NUM_STREAMS = grid_size;")
    assert "invalid scheduler inner-loop count" in source


def test_build_cache_detects_current_inputs_and_binaries(tmp_path, monkeypatch):
    base = tmp_path
    kernels = base / "kernels" / "demo"
    build = base / "build"
    kernels.mkdir(parents=True)
    build.mkdir()
    (base / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    (kernels / "demo.cpp").write_text("int main(){return 0;}\n", encoding="utf-8")
    binary = build / "demo"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)

    monkeypatch.setattr(pantheon, "BASE_DIR", str(base))
    monkeypatch.setattr(pantheon, "KERNEL_DIR", str(base / "kernels"))
    monkeypatch.setattr(pantheon, "BUILD_DIR", str(build))
    monkeypatch.setattr(pantheon, "BUILD_CACHE_FILE", str(build / ".pantheon_build_cache.json"))
    monkeypatch.setattr(pantheon, "TEST_REGISTRY", {
        "demo": {"bin": "demo", "args": [], "desc": "Demo"},
    })

    assert not pantheon.is_build_cache_current("MOCK")
    pantheon.write_build_cache("MOCK")
    assert pantheon.is_build_cache_current("MOCK")

    (kernels / "demo.cpp").write_text("int main(){return 1;}\n", encoding="utf-8")
    assert not pantheon.is_build_cache_current("MOCK")


def test_build_cache_requires_expected_binaries(tmp_path, monkeypatch):
    base = tmp_path
    kernels = base / "kernels" / "demo"
    build = base / "build"
    kernels.mkdir(parents=True)
    build.mkdir()
    (base / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    (kernels / "demo.cpp").write_text("int main(){return 0;}\n", encoding="utf-8")

    monkeypatch.setattr(pantheon, "BASE_DIR", str(base))
    monkeypatch.setattr(pantheon, "KERNEL_DIR", str(base / "kernels"))
    monkeypatch.setattr(pantheon, "BUILD_DIR", str(build))
    monkeypatch.setattr(pantheon, "BUILD_CACHE_FILE", str(build / ".pantheon_build_cache.json"))
    monkeypatch.setattr(pantheon, "TEST_REGISTRY", {
        "demo": {"bin": "demo", "args": [], "desc": "Demo"},
    })

    pantheon.write_build_cache("MOCK")
    assert not pantheon.is_build_cache_current("MOCK")


def test_configure_build_directory_uses_target_specific_cache(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"

    monkeypatch.setattr(pantheon, "BUILD_DIR", pantheon.BUILD_DIR)
    monkeypatch.setattr(pantheon, "BUILD_CACHE_FILE", pantheon.BUILD_CACHE_FILE)
    monkeypatch.setenv("PANTHEON_BUILD_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(pantheon, "PANTHEON_VERSION", "9.9.9")
    monkeypatch.setattr(pantheon, "detect_build_target", lambda _platform: "gfx950")

    build_dir = pantheon.configure_build_directory("HIP")

    assert build_dir == str(cache_root / "9.9.9" / "hip-gfx950")
    assert pantheon.BUILD_CACHE_FILE == str(cache_root / "9.9.9" / "hip-gfx950" / ".pantheon_build_cache.json")


def test_detect_build_target_normalizes_feature_suffix(monkeypatch):
    monkeypatch.delenv("TARGET_GFX", raising=False)
    monkeypatch.setattr(
        pantheon,
        "find_tool",
        lambda name: f"/opt/rocm/bin/{name}" if name == "amdgpu-arch" else None,
    )
    monkeypatch.setattr(
        pantheon.subprocess,
        "check_output",
        lambda *_args, **_kwargs: "gfx942:sramecc+:xnack-\n",
    )

    assert pantheon.detect_build_target("HIP") == "gfx942"


def test_detect_build_target_skips_rocm_cpu_agent(monkeypatch):
    monkeypatch.delenv("TARGET_GFX", raising=False)
    monkeypatch.setattr(
        pantheon,
        "find_tool",
        lambda name: f"/opt/rocm/bin/{name}" if name == "rocm_agent_enumerator" else None,
    )
    monkeypatch.setattr(
        pantheon.subprocess,
        "check_output",
        lambda *_args, **_kwargs: "gfx000\ngfx90a:sramecc+:xnack-\n",
    )

    assert pantheon.detect_build_target("HIP") == "gfx90a"



def test_build_kernels_passes_selected_build_dir_to_make(tmp_path, monkeypatch):
    base = tmp_path / "base"
    kernels = base / "kernels" / "demo"
    cache_root = tmp_path / "cache"
    kernels.mkdir(parents=True)
    (base / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    (kernels / "demo.cpp").write_text("int main(){return 0;}\n", encoding="utf-8")

    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        build_dir = Path(cmd[3].split("=", 1)[1])
        build_dir.mkdir(parents=True, exist_ok=True)
        binary = build_dir / "demo"
        binary.write_text("#!/bin/sh\n", encoding="utf-8")
        binary.chmod(0o755)
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setenv("PANTHEON_BUILD_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(pantheon, "BUILD_DIR", pantheon.BUILD_DIR)
    monkeypatch.setattr(pantheon, "BUILD_CACHE_FILE", pantheon.BUILD_CACHE_FILE)
    monkeypatch.setattr(pantheon, "BASE_DIR", str(base))
    monkeypatch.setattr(pantheon, "KERNEL_DIR", str(base / "kernels"))
    monkeypatch.setattr(pantheon, "PANTHEON_VERSION", "9.9.9")
    monkeypatch.setattr(pantheon, "detect_build_target", lambda _platform: "gfx942")
    monkeypatch.setattr(pantheon, "TEST_REGISTRY", {
        "demo": {"bin": "demo", "args": [], "desc": "Demo"},
    })
    monkeypatch.setattr(pantheon.subprocess, "run", fake_run)
    monkeypatch.setattr(pantheon, "find_tool", lambda name: "/opt/rocm/bin/hipcc" if name == "hipcc" else None)

    pantheon.build_kernels("HIP")

    expected_build_dir = str(cache_root / "9.9.9" / "hip-gfx942")
    assert calls == [[
        "make",
        "-k",
        "PLATFORM=HIP",
        f"BUILD_DIR={expected_build_dir}",
        "HIPCC=/opt/rocm/bin/hipcc",
        "TARGET_GFX=gfx942",
        "ROCM_PATH=/opt/rocm",
    ]]
    assert pantheon.is_build_cache_current("HIP")


def test_build_kernels_returns_only_unavailable_workloads_after_a_partial_failure(tmp_path, monkeypatch):
    base = tmp_path / "base"
    kernels = base / "kernels" / "demo"
    cache_root = tmp_path / "cache"
    kernels.mkdir(parents=True)
    (base / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    (kernels / "good.cpp").write_text("int main(){return 0;}\n", encoding="utf-8")
    (kernels / "bad.cpp").write_text("int main(){return 0;}\n", encoding="utf-8")

    def fake_run(cmd, **_kwargs):
        build_dir = Path(cmd[3].split("=", 1)[1])
        build_dir.mkdir(parents=True, exist_ok=True)
        binary = build_dir / "good"
        binary.write_text("#!/bin/sh\n", encoding="utf-8")
        binary.chmod(0o755)
        return type("Result", (), {"returncode": 2, "stdout": "good built", "stderr": "bad failed"})()

    monkeypatch.setenv("PANTHEON_BUILD_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(pantheon, "BUILD_DIR", pantheon.BUILD_DIR)
    monkeypatch.setattr(pantheon, "BUILD_CACHE_FILE", pantheon.BUILD_CACHE_FILE)
    monkeypatch.setattr(pantheon, "BASE_DIR", str(base))
    monkeypatch.setattr(pantheon, "KERNEL_DIR", str(base / "kernels"))
    monkeypatch.setattr(pantheon, "PANTHEON_VERSION", "9.9.9")
    monkeypatch.setattr(pantheon, "detect_build_target", lambda _platform: "gfx942")
    monkeypatch.setattr(pantheon, "TEST_REGISTRY", {
        "good": {"bin": "good", "args": [], "desc": "Good"},
        "bad": {"bin": "bad", "args": [], "desc": "Bad"},
    })
    monkeypatch.setattr(pantheon.subprocess, "run", fake_run)
    monkeypatch.setattr(pantheon, "find_tool", lambda name: "/opt/rocm/bin/hipcc" if name == "hipcc" else None)

    unavailable = pantheon.build_kernels("HIP")

    assert set(unavailable) == {"bad"}
    assert "build.log" in unavailable["bad"]
    assert (cache_root / "9.9.9" / "hip-gfx942" / "build.log").read_text(encoding="utf-8") == "good built\n--- stderr ---\nbad failed"


def test_resolve_test_queue_handles_all_suite_and_single():
    assert pantheon.resolve_test_queue("memory") == pantheon.SUITES["memory"]
    assert pantheon.resolve_test_queue("tensor_virus") == ["tensor_virus"]
    assert pantheon.resolve_test_queue("all") == list(pantheon.TEST_REGISTRY)


def test_resolve_test_queue_rejects_unknown_selector():
    with pytest.raises(ValueError):
        pantheon.resolve_test_queue("not_a_real_test")


def test_parse_gpu_selection_validates_ids():
    assert pantheon.parse_gpu_selection("all", 3) == [0, 1, 2]
    assert pantheon.parse_gpu_selection("0,2,2", 3) == [0, 2]

    with pytest.raises(ValueError):
        pantheon.parse_gpu_selection("3", 3)
    with pytest.raises(ValueError):
        pantheon.parse_gpu_selection("-1", 3)
    with pytest.raises(ValueError):
        pantheon.parse_gpu_selection("not_gpu", 3)


def test_validate_run_parameters_rejects_bad_ranges():
    pantheon.validate_run_parameters(1, 1)
    pantheon.validate_run_parameters(30, 99)

    with pytest.raises(ValueError):
        pantheon.validate_run_parameters(0, 50)
    with pytest.raises(ValueError):
        pantheon.validate_run_parameters(-1, 50)
    with pytest.raises(ValueError):
        pantheon.validate_run_parameters(30, 0)
    with pytest.raises(ValueError):
        pantheon.validate_run_parameters(30, 100)


def test_cooldown_after_workload_is_duration_based_and_bounded(monkeypatch):
    sleeps = []
    monkeypatch.setattr(pantheon.time, "sleep", sleeps.append)

    assert pantheon.cooldown_after_workload(10) == 1.0
    assert pantheon.cooldown_after_workload(1200) == 60.0
    assert sleeps == [1.0, 60.0]


def test_kernel_memory_limit_matches_the_cli_contract():
    sources = [
        Path(pantheon.KERNEL_DIR, "common", "ai_workload_template.h"),
        Path(pantheon.KERNEL_DIR, "graph_replay", "graph_replay.cpp"),
        Path(pantheon.KERNEL_DIR, "llm_prefill", "llm_prefill.cpp"),
        Path(pantheon.KERNEL_DIR, "llm_decode", "llm_decode.cpp"),
        Path(pantheon.KERNEL_DIR, "kv_cache_churn", "kv_cache_churn.cpp"),
    ]

    for source in sources:
        contents = source.read_text(encoding="utf-8")
        assert "mem_pct > 99" in contents
        assert "mem_pct > 90" not in contents


def test_cache_latency_reports_dependent_load_rate():
    source = Path(
        pantheon.KERNEL_DIR,
        "cache_latency",
        "cache_latency.cpp",
    ).read_text(encoding="utf-8")

    assert '"Throughput: " << dependent_loads / seconds' in source
    assert '" dependent-loads/s"' in source
    assert "inject_latency_sink_error" in source
    assert "return exit_code;" in source


def test_default_profile_metrics_cover_stress_dimensions():
    cuda_metrics = pantheon.get_profile_metrics("CUDA")
    hip_metrics = pantheon.get_profile_metrics("HIP")

    assert "sm__throughput.avg.pct_of_peak_sustained_elapsed" in cuda_metrics
    assert "sm__inst_executed_pipe_tensor.sum" in cuda_metrics
    assert "dram__throughput.avg.pct_of_peak_sustained_elapsed" in cuda_metrics
    assert "lts__t_sector_hit_rate.pct" in cuda_metrics
    assert "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum" in cuda_metrics
    assert "smsp__warps_issue_stalled_long_scoreboard_per_warp_active.pct" in cuda_metrics
    assert "pcie__read_bytes.sum" in cuda_metrics
    assert "gpu__time_duration.sum" in cuda_metrics
    assert "dram__bytes_read.sum.per_second" in cuda_metrics
    assert "sm__ctas_active.avg.pct_of_peak_sustained_active" in cuda_metrics
    assert "smsp__thread_inst_executed_per_inst_executed.ratio" in cuda_metrics
    assert "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct" in cuda_metrics
    assert "smsp__warps_issue_stalled_lg_throttle.avg.per_cycle_active" in cuda_metrics
    assert "lts__t_requests_op_read.sum" in cuda_metrics
    assert "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum" in cuda_metrics
    assert "GRBM_GUI_ACTIVE" in hip_metrics
    assert "SQ_INSTS_MFMA" in hip_metrics
    assert "SQ_INSTS_SMEM" in hip_metrics
    assert "TCP_TOTAL_CACHE_MISSES" in hip_metrics
    assert "TCC_EA_RDREQ" in hip_metrics
    assert "SDMA0_ACTIVE" in hip_metrics
    assert "SQ_INSTS_VOPD" in hip_metrics
    assert "SQ_WAIT_INST_LDS" in hip_metrics
    assert "TCP_TCC_READ_REQ" in hip_metrics
    assert "SQ_WAIT_INST_FLAT" in hip_metrics
    assert "TCP_TCC_ATOMIC_WITH_RET_REQ" in hip_metrics


def test_profile_metrics_can_be_overridden(monkeypatch):
    monkeypatch.setenv("PANTHEON_CUDA_METRICS", "metric.a, metric.b\nmetric.a")
    monkeypatch.setenv("PANTHEON_HIP_METRICS", "SQ_WAVES")

    assert pantheon.get_profile_metrics("CUDA") == ["metric.a", "metric.b"]
    assert pantheon.get_profile_metrics("HIP") == ["SQ_WAVES"]


def test_profile_metrics_can_be_appended_without_duplicates(monkeypatch):
    monkeypatch.delenv("PANTHEON_CUDA_METRICS", raising=False)
    monkeypatch.delenv("PANTHEON_HIP_METRICS", raising=False)
    monkeypatch.setenv("PANTHEON_CUDA_METRICS_APPEND", "custom.nvidia, sm__throughput.avg.pct_of_peak_sustained_elapsed")
    monkeypatch.setenv("PANTHEON_HIP_METRICS_APPEND", "CUSTOM_AMD, SQ_WAVES")

    cuda_metrics = pantheon.get_profile_metrics("CUDA")
    hip_metrics = pantheon.get_profile_metrics("HIP")

    assert cuda_metrics[-1] == "custom.nvidia"
    assert cuda_metrics.count("sm__throughput.avg.pct_of_peak_sustained_elapsed") == 1
    assert hip_metrics[-1] == "CUSTOM_AMD"
    assert hip_metrics.count("SQ_WAVES") == 1


def test_parse_kernel_output_preserves_parameter_lines():
    out = "\n".join([
        "[PANTHEON] GPU 0: Running FP64 VIRUS",
        "  -> Block Size:    256",
        "  -> Kernel Loops:  10000",
        "Throughput: 1.25 TFLOPS",
        "Verification: PASS (0 errors)",
    ])

    throughput, unit, status, had_error, lines = pantheon.parse_kernel_output(out, "", 0)

    assert throughput == 1.25
    assert unit == "TFLOPS"
    assert status == "PASS"
    assert not had_error
    assert "-> Block Size:    256" in lines
    assert "-> Kernel Loops:  10000" in lines
    assert "Verification: PASS (0 errors)" in lines


def test_build_result_row_includes_profile_metadata_and_counter_summary():
    row = pantheon.build_result_row(
        "fp64_virus",
        gpu=0,
        duration=10,
        mem_pct=99,
        throughput=1.0,
        unit="TFLOPS",
        stats={},
        commands=["./build/fp64_virus 0 10 99 --kernel_loops 100000"],
        profile_commands=["ncu --csv -- ./build/fp64_virus 0 10 99"],
        profile_files=["results/run/fp64_gpu0_hardware_counters.csv"],
        counter_summary={"Counter sm__inst_executed_pipe_fma.sum": "123 count"},
    )

    assert row["Command Lines"] == "./build/fp64_virus 0 10 99 --kernel_loops 100000"
    assert row["Profiler Command Lines"] == "ncu --csv -- ./build/fp64_virus 0 10 99"
    assert row["Profiler Counter Files"] == "results/run/fp64_gpu0_hardware_counters.csv"
    assert row["Counter sm__inst_executed_pipe_fma.sum"] == "123 count"


def test_throughput_variance_uses_repeated_samples():
    assert pantheon.throughput_variance_percent(
        "Throughput: 100 GB/s\nThroughput: 110 GB/s\n"
    ) == 4.76
    assert pantheon.throughput_variance_percent("Throughput: 100 GB/s") == "N/A"


def test_run_test_drains_large_child_output(tmp_path, monkeypatch):
    executable = tmp_path / "chatty"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "print('x' * 1000000)\n"
        "print('Throughput: 1.0 GB/s')\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)

    monkeypatch.setattr(pantheon, "BUILD_DIR", str(tmp_path))
    monkeypatch.setattr(pantheon, "TEST_REGISTRY", {
        "chatty": {"bin": "chatty", "args": [], "desc": "Chatty"},
    })

    processes = pantheon.run_test("chatty", [0], 0, 1, "MOCK", str(tmp_path))
    assert not pantheon.wait_for_processes(processes, 10)

    process_info = processes[0]
    process_info["output_thread"].join(timeout=5)
    assert not process_info["output_thread"].is_alive()
    assert len(process_info["stdout"]) > 1000000
    assert "Throughput: 1.0 GB/s" in process_info["stdout"]


def test_profile_request_without_profiler_fails_instead_of_running_unprofiled(tmp_path, monkeypatch):
    executable = tmp_path / "quiet"
    executable.write_text(
        "#!/usr/bin/env python3\nprint('Throughput: 1.0 GB/s')\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)

    monkeypatch.setattr(pantheon, "BUILD_DIR", str(tmp_path))
    monkeypatch.setattr(pantheon, "TEST_REGISTRY", {
        "quiet": {"bin": "quiet", "args": [], "desc": "Quiet"},
    })
    monkeypatch.setattr(pantheon.shutil, "which", lambda _name: None)

    with pytest.raises(pantheon.ProfileUnavailableError, match="refusing to run an incomplete profile"):
        pantheon.run_test("quiet", [0], 0, 1, "CUDA", str(tmp_path), profile=True)


def test_profiled_workloads_disable_the_normal_watchdog(tmp_path, monkeypatch):
    executable = tmp_path / "quiet"
    executable.write_text(
        "#!/usr/bin/env python3\nprint('Throughput: 1.0 GB/s')\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setattr(pantheon, "BUILD_DIR", str(tmp_path))
    monkeypatch.setattr(pantheon, "TEST_REGISTRY", {"quiet": {"bin": "quiet", "args": [], "desc": "Quiet"}})
    monkeypatch.setattr(pantheon, "discover_profile_tools", lambda _platform: {"counter": "/bin/true", "trace": "/bin/true"})
    monkeypatch.setattr(pantheon, "filter_supported_profile_metrics", lambda *_args: (["metric.one"], [], "validated"))
    monkeypatch.setattr(pantheon, "build_counter_command", lambda *_args: [str(executable)])
    monkeypatch.setattr(pantheon, "build_trace_command", lambda *_args: [])
    monkeypatch.setattr(pantheon, "generate_profile_html", lambda *_args: str(tmp_path / "profile_summary.html"))
    observed_timeouts = []
    monkeypatch.setattr(pantheon, "wait_for_processes", lambda _procs, timeout: observed_timeouts.append(timeout) or False)

    class Monitor:
        def start_collection(self, *_args):
            pass
        def stop_collection(self):
            return {0: {}}

    pantheon.execute_test("quiet", [0], 1, 1, "CUDA", str(tmp_path), Monitor(), profile=True)

    assert observed_timeouts == [None]


def test_normal_workload_writes_ras_report_without_profile(tmp_path, monkeypatch):
    executable = tmp_path / "quiet"
    executable.write_text("#!/usr/bin/env python3\nprint('Throughput: 1.0 GB/s')\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setattr(pantheon, "BUILD_DIR", str(tmp_path))
    monkeypatch.setattr(pantheon, "TEST_REGISTRY", {"quiet": {"bin": "quiet", "args": [], "desc": "Quiet"}})

    snapshots = iter([
        {"sources": {"vendor_ras": {"status": "supported", "metrics": {"ecc.correctable": {"status": "supported", "value": 2, "unit": "count"}}}}},
        {"sources": {"vendor_ras": {"status": "supported", "metrics": {"ecc.correctable": {"status": "supported", "value": 3, "unit": "count"}}}}},
    ])
    monkeypatch.setattr(pantheon, "collect_ras_snapshot", lambda *_args: next(snapshots))

    class Monitor:
        def start_collection(self, *_args):
            pass
        def stop_collection(self):
            return {0: {}}

    rows, had_errors = pantheon.execute_test("quiet", [0], 1, 1, "MOCK", str(tmp_path), Monitor())

    assert not had_errors
    assert rows[0]["RAS Status"] == "WARNING"
    report_path = Path(rows[0]["RAS Report"])
    assert report_path == tmp_path / "reliability" / "quiet" / "gpu0" / "ras.json"
    assert json.loads(report_path.read_text(encoding="utf-8"))["delta"]["metrics"][0]["delta"] == 1


def test_incremental_workload_report_is_atomic_and_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(pantheon, "DATABASE_DIR", str(tmp_path / "database"))
    row = {
        "Test Name": "memory_write",
        "GPU ID": 0,
        "Version": "1.0.16",
        "Score": 123.4,
        "Unit": "GB/s",
    }

    pantheon.write_incremental_workload_reports(
        {"pantheon_version": "1.0.16", "gpu_static_info": []},
        [row],
        "20260822-120000",
        1,
    )

    reports = list((tmp_path / "database").glob("pantheon_report_*.json"))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert payload["run_status"] == "complete"
    assert payload["record_kind"] == "completed_workload"
    assert payload["test_results"] == [row]
    assert not list((tmp_path / "database").glob("*.tmp"))


def test_launch_failure_is_reported_without_aborting_the_workload_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(pantheon, "BUILD_DIR", str(tmp_path))
    monkeypatch.setattr(pantheon, "TEST_REGISTRY", {
        "quiet": {"bin": "quiet", "args": [], "desc": "Quiet"},
    })
    monkeypatch.setattr(
        pantheon.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("missing workload binary")),
    )
    monkeypatch.setattr(pantheon, "collect_ras_snapshot", lambda *_args: {})

    class Monitor:
        def start_collection(self, *_args):
            pass

        def stop_collection(self):
            return {0: {}}

    rows, had_errors = pantheon.execute_test("quiet", [0], 1, 1, "MOCK", str(tmp_path), Monitor())

    assert had_errors
    assert rows[0]["Status"] == "FAIL"
    assert rows[0]["Failure Stage"] == "launch"
    assert "missing workload binary" in rows[0]["Failure Reason"]
    assert rows[0]["RAS Status"] == "UNAVAILABLE"


def test_profile_artifacts_are_isolated_by_workload_and_gpu(tmp_path):
    assert pantheon.profile_artifact_dir(str(tmp_path), "tensor_virus", 0) == str(
        tmp_path / "profiles" / "tensor_virus" / "gpu0"
    )
    assert pantheon.profile_artifact_dir(str(tmp_path), "memory_write", 1) == str(
        tmp_path / "profiles" / "memory_write" / "gpu1"
    )


def test_runner_uses_python36_compatible_subprocess_text_mode():
    source = Path(pantheon.__file__).read_text(encoding="utf-8")
    assert "text=True" not in source
    assert "capture_output=True" not in source
    assert "universal_newlines=True" in source


def test_profile_manifest_records_supported_and_skipped_metrics(tmp_path):
    _artifact_dir, manifest_path, _manifest = pantheon.create_profile_manifest(
        str(tmp_path), "memory_read", 0, "CUDA", {"counter": "/usr/bin/ncu"},
        ["metric.supported", "metric.skipped"], ["workload"],
        supported_metrics=["metric.supported"], skipped_metrics=["metric.skipped"],
    )

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert manifest["supported_metrics"] == ["metric.supported"]
    assert manifest["skipped_metrics"] == ["metric.skipped"]


def test_cuda_profile_metrics_are_filtered_to_the_target_gpu(monkeypatch):
    class Result:
        returncode = 0
        stdout = "Metric Name\nsm__throughput.avg.pct_of_peak_sustained_elapsed %\ndram__bytes_read.sum byte\n"

    monkeypatch.setattr(pantheon.subprocess, "run", lambda *_args, **_kwargs: Result())

    supported, skipped, validation = pantheon.filter_supported_profile_metrics(
        "CUDA",
        {"counter": "/usr/bin/ncu", "trace": "/usr/bin/nsys"},
        0,
        [
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__bytes_read.sum",
            "nvlink__rx_data.sum",
        ],
    )

    assert supported == [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__bytes_read.sum",
    ]
    assert skipped == ["nvlink__rx_data.sum"]
    assert validation == "validated"


def test_counter_summary_skips_nsight_compute_preamble(tmp_path):
    counter_file = tmp_path / "hardware_counters.csv"
    counter_file.write_text(
        "==PROF== Profiling application: ./build/memory_write\n"
        "[PANTHEON] GPU 0: NVIDIA RTX 3060\n"
        '"ID","Metric Name","Metric Unit","Metric Value"\n'
        '"0","dram__bytes_write.sum","byte","58830556928"\n',
        encoding="utf-8",
    )

    assert pantheon.summarize_hardware_counter_file(counter_file) == {
        "Counter dram__bytes_write.sum": "58830556928 byte",
    }


def test_profile_html_summary_contains_aggregated_counters(tmp_path):
    counters = tmp_path / "hardware_counters.csv"
    counters.write_text(
        '"ID","Metric Name","Metric Unit","Metric Value"\n'
        '"0","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","95.5"\n'
        '"1","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","96.5"\n',
        encoding="utf-8",
    )
    report = pantheon.generate_profile_html(
        str(tmp_path),
        {"test": "memory_write", "gpu_id": 0, "status": "PASS", "result": {
            "Score": 1.0, "Unit": "GB/s", "Duration (s)": 1, "Mem Usage (%)": 50,
            "Avg Temp (C)": 60, "Max Temp (C)": 70, "Avg Power (W)": 100,
        }},
        [str(counters)],
    )

    contents = Path(report).read_text(encoding="utf-8")
    assert "Pantheon profile" in contents
    assert "dram__throughput.avg.pct_of_peak_sustained_elapsed" in contents
    assert "96 %" in contents
    assert "hardware_counters.csv" in contents


def test_cuda_profile_plan_collects_exhaustive_counters_and_a_separate_trace(tmp_path):
    tools = {"counter": "/usr/bin/ncu", "trace": "/usr/bin/nsys"}
    workload = ["./build/tensor_virus", "0", "10", "99"]
    artifact_dir = str(tmp_path / "profiles" / "tensor_virus" / "gpu0")

    counter_cmd = pantheon.build_counter_command(
        "CUDA", tools, ["metric.one", "metric.two"], artifact_dir, workload,
    )
    trace_cmd = pantheon.build_trace_command("CUDA", tools, artifact_dir, workload)

    assert counter_cmd[:7] == [
        "/usr/bin/ncu", "--csv", "--set", "full", "--metrics", "metric.one,metric.two", "./build/tensor_virus",
    ]
    assert "-c" not in counter_cmd
    assert trace_cmd[:10] == [
        "/usr/bin/nsys", "profile", "-t", "cuda,nvtx,osrt", "-s", "none",
        "--cpuctxsw=none", "--force-overwrite=true", "--output", os.path.join(artifact_dir, "trace"),
    ]
    assert trace_cmd[-4:] == workload


def test_profile_telemetry_pass_uses_clean_workload_output(monkeypatch, tmp_path):
    class FakeProcess:
        returncode = 0

        def communicate(self, timeout=None):
            assert timeout is not None
            return "Throughput: 869.0 GB/s\nVerification: PASSED\n", ""

    calls = []

    def fake_popen(argv, **_kwargs):
        calls.append(argv)
        return FakeProcess()

    class FakeMonitor:
        def start_collection(self, gpu_ids, output_dir, test_name):
            assert gpu_ids == [0]
            assert output_dir == str(tmp_path)
            assert test_name == "memory_read"

        def stop_collection(self):
            return {0: {"avg_pwr": 236.0}}

    monkeypatch.setattr(pantheon.subprocess, "Popen", fake_popen)
    stats, outputs, errors = pantheon.run_profile_telemetry_pass(
        "memory_read",
        [{"gpu": 0, "workload_argv": ["./build/memory_read", "0", "10", "99", "--verify"]}],
        [0], 10, FakeMonitor(), str(tmp_path),
    )

    assert not errors
    assert stats[0]["avg_pwr"] == 236.0
    assert outputs[0][0].startswith("Throughput: 869.0 GB/s")
    assert calls == [["./build/memory_read", "0", "10", "99", "--verify"]]


def test_hip_profile_plan_collects_counters_and_runtime_trace_in_one_pass(tmp_path):
    tools = {"counter": "/opt/rocm/bin/rocprofv3", "trace": "/opt/rocm/bin/rocprofv3"}
    workload = ["./build/memory_write", "0", "10", "99"]
    artifact_dir = str(tmp_path / "profiles" / "memory_write" / "gpu0")

    counter_cmd = pantheon.build_counter_command("HIP", tools, ["SQ_WAVES"], artifact_dir, workload)

    assert "--pmc" in counter_cmd
    assert "--runtime-trace" in counter_cmd
    assert "--output-config" in counter_cmd
    assert pantheon.build_trace_command("HIP", tools, artifact_dir, workload) == []


def test_summarize_hardware_counter_file_flattens_ncu_metrics(tmp_path):
    counters = tmp_path / "counters.csv"
    counters.write_text(
        "Metric Name,Metric Unit,Metric Value\n"
        "sm__inst_executed_pipe_fma.sum,count,123\n"
        "lts__t_sectors.avg.pct_of_peak_sustained_elapsed,%,45.5\n",
        encoding="utf-8",
    )

    summary = pantheon.summarize_hardware_counter_file(str(counters))

    assert summary["Counter sm__inst_executed_pipe_fma.sum"] == "123 count"
    assert summary["Counter lts__t_sectors.avg.pct_of_peak_sustained_elapsed"] == "45.5 %"


def test_counter_summary_ignores_empty_profiler_analysis_rows(tmp_path):
    counters = tmp_path / "counters.csv"
    counters.write_text(
        "Metric Name,Metric Unit,Metric Value\n"
        "sm__inst_executed.sum,inst,123\n"
        ",,\n",
        encoding="utf-8",
    )

    assert pantheon.summarize_hardware_counter_file(str(counters)) == {
        "Counter sm__inst_executed.sum": "123 inst",
    }


def test_get_app_version_reads_repo_version():
    assert pantheon.get_app_version() == Path(
        os.path.join(pantheon.BASE_DIR, "VERSION")
    ).read_text(encoding="utf-8").strip()


def test_system_snapshot_contains_no_host_identifiers():
    # Snapshots become public website reports; hostnames and IP addresses
    # must never be collected into them.
    snapshot = pantheon.get_system_snapshot("MOCK")

    assert "network_info" not in snapshot
    flattened = json.dumps(snapshot).lower()
    assert "hostname" not in flattened
    assert "ip_address" not in flattened


def test_version_option_reports_version(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["pantheon.py", "--version"])

    with pytest.raises(SystemExit) as exc:
        pantheon.main()

    assert exc.value.code == 0
    assert pantheon.PANTHEON_VERSION in capsys.readouterr().out


def test_main_rejects_invalid_gpu_before_build(monkeypatch):
    build_called = False

    class FakeMonitor:
        def __init__(self, _platform):
            pass

        def get_gpu_count(self):
            return 1

    def fake_build(_platform):
        nonlocal build_called
        build_called = True

    monkeypatch.setattr(pantheon, "build_kernels", fake_build)
    monkeypatch.setattr(pantheon, "HardwareMonitor", FakeMonitor)
    monkeypatch.setattr("sys.argv", [
        "pantheon.py",
        "--platform", "mock",
        "--test", "baseline_metrics",
        "--duration", "1",
        "--gpu", "99",
        "--mem", "1",
    ])

    with pytest.raises(SystemExit) as exc:
        pantheon.main()

    assert exc.value.code == 1
    assert not build_called


def test_main_rejects_invalid_test_before_platform_detection_or_build(monkeypatch):
    monkeypatch.setattr("sys.argv", ["pantheon.py", "--test", "not_a_test"])
    monkeypatch.setattr(
        pantheon,
        "detect_platform",
        lambda *_args, **_kwargs: pytest.fail("platform detection should not run"),
    )
    monkeypatch.setattr(
        pantheon,
        "build_kernels",
        lambda *_args, **_kwargs: pytest.fail("build should not run"),
    )

    with pytest.raises(SystemExit) as exc:
        pantheon.main()

    assert exc.value.code == 1


def test_main_requires_verification_for_error_injection_when_skipped(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", [
        "pantheon.py",
        "--test", "baseline_metrics",
        "--inject_error",
        "--skip_verify",
    ])

    with pytest.raises(SystemExit) as exc:
        pantheon.main()

    assert exc.value.code == 1
    assert "--inject_error requires --verify" in capsys.readouterr().out


def test_every_registered_workload_has_a_readme():
    """A workload nobody can read the docs for is a workload nobody will run."""
    repo = Path(__file__).resolve().parent.parent
    def documented(binary):
        # Variant binaries (e.g. compute_virus_agg) are built from the base
        # source directory and documented in that directory's README.
        for candidate in (binary, binary[:-4] if binary.endswith("_agg") else binary):
            if (repo / "kernels" / candidate / "README.md").is_file():
                return True
        return False

    missing = sorted(
        name for name, spec in pantheon.TEST_REGISTRY.items()
        if not documented(spec["bin"])
    )
    assert missing == [], f"workloads with no README: {missing}"


def test_diagnostics_suite_members_are_registered():
    for name in pantheon.SUITES["diagnostics"]:
        assert name in pantheon.TEST_REGISTRY, f"{name} is in the suite but not the registry"


def test_documented_init_patterns_are_all_implemented():
    """The README pattern table and the shared selector must not drift apart."""
    repo = Path(__file__).resolve().parent.parent
    selector = (repo / "kernels" / "common" / "common.h").read_text()
    body = selector[selector.index("pantheon_pattern_value"):]
    patterns_doc = (repo / "docs" / "memory_diagnostics.md").read_text()
    documented = re.findall(r"^\| `[a-z_]+` \| (\d) \|", patterns_doc, re.M)
    assert documented, "docs/memory_diagnostics.md no longer documents an --init_pattern table"
    handled = set(re.findall(r"case (\d+):", body))
    # 2 and 3 fall through to pantheon_pattern_base rather than having a case arm.
    handled |= {"2", "3"}
    assert set(documented) <= handled, (
        f"documented but not implemented: {sorted(set(documented) - handled)}"
    )


def test_memory_hammer_reports_its_aggressor_footprint():
    """The cache-hit caveat is only checkable if the binary prints the comparison."""
    repo = Path(__file__).resolve().parent.parent
    src = (repo / "kernels" / "memory_hammer" / "memory_hammer.cpp").read_text()
    assert "Aggressor Set" in src and "l2CacheSize" in src
    assert "aggressor-reads/s" in src, "metric must not claim confirmed DRAM activations"

def test_pattern_names_resolve_to_the_numbers_the_aliases_replaced():
    """The names are only safe if they still select the original backgrounds."""
    repo = Path(__file__).resolve().parent.parent
    header = (repo / "kernels" / "common" / "common.h").read_text()
    table = re.search(r"PANTHEON_PATTERN_NAMES\[\]\s*=\s*\{(.*?)\};", header, re.S).group(1)
    mapping = dict(
        (m.group(1), int(m.group(2)))
        for m in re.finditer(r'\{"([a-z_]+)",\s*(\d+)\}', table)
    )
    # These two are what memory_read_agg / memory_write_agg used to hardcode.
    assert mapping["rail_to_rail"] == 3
    assert mapping["crosstalk"] == 2
    assert mapping["zeros"] == 0 and mapping["ones"] == 1

    # Every name the README documents must exist in the header table.
    patterns_doc = (repo / "docs" / "memory_diagnostics.md").read_text()
    documented = set(re.findall(r"^\| `([a-z_]+)` \| \d+ \|", patterns_doc, re.M))
    assert documented, "docs/memory_diagnostics.md no longer documents a named pattern table"
    assert documented <= set(mapping), f"documented but unknown: {documented - set(mapping)}"


def test_unknown_pattern_name_is_refused_not_defaulted():
    """A typo must not silently fall back to zeros."""
    repo = Path(__file__).resolve().parent.parent
    header = (repo / "kernels" / "common" / "common.h").read_text()
    parse = header[header.index("pantheon_parse_init_pattern"):]
    assert "return false;" in parse, "parser must be able to reject an argument"
    src = (repo / "kernels" / "memory_read" / "memory_read.cpp").read_text()
    assert "Unknown --init_pattern" in src and "return 1;" in src


def test_ai_workloads_do_not_claim_semantic_units():
    """The shared AI harness measures thread-iterations, not AI work units.

    Ten workloads share one loop and land within ~1% of each other on the same
    GPU. Labelling that number "requests/s" or "train-steps/s" invites it to be
    read as an inference or training rate, which it is not.
    """
    repo = Path(__file__).resolve().parent.parent
    template = (repo / "kernels" / "common" / "ai_workload_template.h").read_text()
    units = set(re.findall(r'#define AI_UNIT "([^"]+)"', template))
    assert units == {"ai-ops/s"}, f"AI workloads claim semantic units: {sorted(units)}"


def test_metric_units_are_declared_once_per_workload():
    """A workload reporting two different units would make results ambiguous."""
    repo = Path(__file__).resolve().parent.parent
    offenders = []
    for kernel in sorted((repo / "kernels").glob("*/*.cpp")):
        body = kernel.read_text(errors="ignore")
        units = re.findall(r'"Throughput: " <<[^;]*?<< " ([a-zA-Z/\-]+)"', body)
        if len(set(units)) > 1:
            offenders.append((kernel.parent.name, sorted(set(units))))
    assert offenders == [], f"workloads reporting multiple units: {offenders}"


def test_runtime_skip_is_not_recorded_as_a_zero_measurement():
    """A kernel that skips itself still prints Throughput: 0.0.

    Recorded as a real sample, that is indistinguishable from hardware which
    is catastrophically slow, and it drags any aggregate down with it.
    """
    out = ("[PANTHEON] GPU 0: Skipping P2P_THRASHER (Bidirectional P2P routing "
           "not supported to GPU 1).\nThroughput: 0.0 GB/s\n")
    throughput, unit, status, had_error, _ = pantheon.parse_kernel_output(out, "", 0)
    assert status == "SKIP" and unit == "SKIP"
    assert not had_error, "a skip is not a failure"


def test_normal_and_failed_runs_are_unaffected_by_skip_detection():
    ok = "[PANTHEON] GPU 0: Memory Read\nThroughput: 348.4 GB/s\n"
    throughput, unit, status, _, _ = pantheon.parse_kernel_output(ok, "", 0)
    assert (throughput, unit, status) == (348.4, "GB/s", "PASS")

    bad = "[PANTHEON] GPU 0: Memory Read\n[SDC FAULT] mismatch\n"
    _, unit, status, had_error, _ = pantheon.parse_kernel_output(bad, "", 0)
    assert (unit, status, had_error) == ("ERR", "FAIL", True)


def test_pingpong_warmup_does_not_inject():
    """Injecting during warmup made the fault cancel itself.

    memory_pc_pingpong injects by XOR and fires on launch_idx 0, which both
    the warmup loop and the active loop have. Two XORs of the same mask are a
    no-op, so --inject_error silently passed verification -- the workload's
    own self-test could not fail.
    """
    repo = Path(__file__).resolve().parent.parent
    src = (repo / "kernels" / "memory_pc_pingpong" / "memory_pc_pingpong.cpp").read_text()
    warmup = src[src.index("safe_warmups"):src.index("Starting active telemetry")]
    launch = [ln for ln in warmup.splitlines() if "LAUNCH_KERNEL" in ln]
    assert launch, "warmup launch not found"
    assert "inject_error" not in launch[0], "warmup must not inject"


def test_launch_parameters_are_bounded_above():
    """Unbounded launch knobs present as a GPU hang, not as an error.

    block_size was already clamped; grid_size, kernel_loops and hammer_pairs
    were not, and a large value on any of them made the launch stop returning.
    """
    repo = Path(__file__).resolve().parent.parent
    common = (repo / "kernels" / "common" / "common.h").read_text()

    assert "PANTHEON_MAX_GRID_BLOCKS" in common
    assert "PANTHEON_MAX_KERNEL_LOOPS" in common
    # The user-supplied grid must be clamped, not just the internal fill grids.
    norm = common[common.index("inline void normalize_kernel_launch_config"):]
    norm = norm[:norm.index("\n}")]
    assert "PANTHEON_MAX_GRID_BLOCKS" in norm, "grid_size is not bounded above"
    assert "PANTHEON_MAX_KERNEL_LOOPS" in norm, "kernel_loops is not bounded above"

    hammer = (repo / "kernels" / "memory_hammer" / "memory_hammer.cpp").read_text()
    assert "MAX_HAMMER_PAIRS" in hammer, "hammer_pairs is not bounded above"


def test_work_per_launch_is_bounded_not_just_loop_count():
    """Per-loop cost differs by orders of magnitude between workloads, so a
    single loop limit cannot bound both a bandwidth and an ALU kernel."""
    repo = Path(__file__).resolve().parent.parent
    common = (repo / "kernels" / "common" / "common.h").read_text()
    assert "PANTHEON_MAX_BYTES_PER_LAUNCH" in common
    scaler = common[common.index("inline void scale_kernel_loops_for_large_alloc"):]
    scaler = scaler[:scaler.index("\n}")]
    assert "PANTHEON_MAX_BYTES_PER_LAUNCH" in scaler
    assert "alloc_bytes" in scaler, "the bound must be sized by the allocation"

def test_terminate_process_tree_kills_children_not_just_the_parent():
    """Under --profile the launched process is the profiler and the GPU
    workload is its child. Killing only the profiler orphans the workload,
    which keeps saturating the GPU with nothing watching it."""
    import signal as _signal
    import subprocess as _sp
    import time as _time

    proc = _sp.Popen(["sh", "-c", "sleep 37 & wait"], start_new_session=True)
    try:
        _time.sleep(0.5)
        child_pgid = os.getpgid(proc.pid)
        pantheon.terminate_process_tree(proc)
        try:
            proc.wait(timeout=5)
        except _sp.TimeoutExpired:
            pytest.fail("parent survived termination")
        _time.sleep(0.3)
        # The whole group must be gone, not just the process we launched.
        with pytest.raises(ProcessLookupError):
            os.killpg(child_pgid, 0)
    finally:
        for pid in _sp.run(["pgrep", "-f", "sleep 37"], capture_output=True,
                           text=True).stdout.split():
            try:
                os.kill(int(pid), _signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass


def test_every_workload_launch_starts_its_own_session():
    """A process group is only killable as a unit if one was created."""
    repo = Path(__file__).resolve().parent.parent
    src = (repo / "pantheon.py").read_text()
    launches = src.count("subprocess.Popen(")
    sessions = src.count("start_new_session=True")
    assert sessions >= launches, (
        f"{launches} Popen sites but only {sessions} create a session")


def test_every_ai_kind_has_its_own_kernel_body():
    """Ten workloads once shared one loop; six were byte-identical.

    A kind that falls through to the shared fallback is not a distinct
    diagnostic -- running it tells you exactly what another one already did.
    """
    repo = Path(__file__).resolve().parent.parent
    template = (repo / "kernels" / "common" / "ai_workload_template.h").read_text()

    declared = set(re.findall(r"#(?:el)?if PANTHEON_AI_WORKLOAD_KIND == (\d+)\n#define AI_NAME", template))
    body = template[template.index("__global__ void ai_workload_kernel"):]
    body = body[:body.index("__global__ void ai_verify")]
    implemented = set(re.findall(r"#(?:el)?if PANTHEON_AI_WORKLOAD_KIND == (\d+)", body))

    # Kind 8 left the template entirely: allocation_fragmentation is a real
    # allocator workload now, not a kernel variant.
    missing = declared - implemented - {"8"}
    assert missing == set(), f"AI kinds with no kernel of their own: {sorted(missing)}"


def test_allocation_fragmentation_actually_allocates():
    """It was named for allocator behaviour while running a shared FMA loop and
    never calling an allocator once."""
    repo = Path(__file__).resolve().parent.parent
    src = (repo / "kernels" / "allocation_fragmentation" / "allocation_fragmentation.cpp").read_text()
    assert "ai_workload_template.h" not in src, "still the shared AI kernel"
    assert src.count("hipMalloc") >= 1 and src.count("hipFree") >= 1
    assert "alloc-events/s" in src, "must report what it actually counts"


def test_ai_workloads_differ_by_load_profile_not_only_arithmetic():
    """ALU work inside a memory-fed loop hides behind load latency.

    The original design gave four kinds a bespoke FP expression and they still
    measured within 0.9% of the six that had none. What separates these
    workloads has to be how they touch memory.
    """
    repo = Path(__file__).resolve().parent.parent
    body = (repo / "kernels" / "common" / "ai_workload_template.h").read_text()
    body = body[body.index("__global__ void ai_workload_kernel"):]
    body = body[:body.index("__global__ void ai_verify")]
    strides = set(re.findall(r"offset \+= (\d+)", body)) | set(re.findall(r"step \+= (\d+)", body))
    assert len(strides) > 1, f"every kind walks memory identically: {strides}"


def test_fail_verdict_is_honoured_even_on_a_clean_exit():
    """A workload reporting FAIL must be recorded as failed.

    Only [SDC FAULT] and a non-zero exit used to mark a run bad, so a
    "Verification: FAIL" line was decorative -- a failed verification with a
    clean exit was filed as PASS, throughput and all. That was safe only
    because every workload printing a verdict happened to also exit 1.
    """
    out = "Throughput: 340.1 GB/s\nVerification: FAIL (17 errors)\n"
    throughput, unit, status, had_error, _ = pantheon.parse_kernel_output(out, "", 0)
    assert status == "FAIL" and had_error
    assert throughput == 0.0 and unit == "ERR", "a failed run must not report a throughput"


def test_pass_verdict_does_not_trip_the_fail_check():
    """The verdict token is read, not searched for, so a PASS line that happens
    to mention failure is still a pass."""
    for verdict in ("PASS (0 errors)", "PASSED", "PASS (0 failed cells)"):
        out = f"Throughput: 340.1 GB/s\nVerification: {verdict}\n"
        throughput, _, status, had_error, _ = pantheon.parse_kernel_output(out, "", 0)
        assert status == "PASS" and not had_error, verdict
        assert throughput == 340.1, verdict


def test_every_verifying_workload_reports_a_verdict():
    """A clean --verify must say so, not just stay quiet.

    27 workloads used to print only on failure, so a passing verification and
    a verification that never ran produced byte-identical output: nothing.
    That is exactly how memory_pc_pingpong hid a self-test that could not
    fail. Silence is now a signal rather than the norm.
    """
    repo = Path(__file__).resolve().parent.parent
    silent = []
    for kernel in sorted((repo / "kernels").glob("*/*.cpp")):
        body = kernel.read_text(errors="ignore")
        verifies = "SDC FAULT" in body or "Verification:" in body
        if not verifies:
            continue
        # Either construction counts: streaming the verdict in
        # (`"Verification: " << ...`) or emitting it whole.
        reports = 'Verification: "' in body or "Verification: PASS" in body
        if not reports:
            silent.append(kernel.parent.name)
    assert silent == [], f"workloads that verify but never report a verdict: {silent}"
