import csv
import json

from monitor import HardwareMonitor


def test_parse_metric_handles_common_smi_formats():
    assert HardwareMonitor._parse_metric("(800Mhz)") == 800
    assert HardwareMonitor._parse_metric("72.5 W") == 72.5
    assert HardwareMonitor._parse_metric("45%") == 45
    assert HardwareMonitor._parse_metric("N/A") == 0


def test_mock_monitor_is_cpu_only_and_writes_named_time_series(tmp_path, monkeypatch):
    monitor = HardwareMonitor("MOCK")

    def stop_after_first_sample(_seconds):
        monitor.running = False

    monkeypatch.setattr("monitor.time.sleep", stop_after_first_sample)
    monitor.start_collection([0], tmp_path, "baseline_metrics")
    stats = monitor.stop_collection()

    assert monitor.get_gpu_count() == 1
    # The mock backend has no GPU, so no sensor is ever read. These report
    # "N/A" rather than 0 for the same reason a card with no memory-temperature
    # sensor does: a number here would be a measurement nobody took.
    assert stats[0]["avg_temp"] == "N/A"
    assert stats[0]["avg_pwr"] == "N/A"
    assert stats[0]["avg_gpu_util"] == "N/A"
    assert stats[0]["peak_mem_used"] == "N/A"
    # Energy is integrated from the power series, so an empty series gives 0
    # joules rather than "unknown" -- no power was drawn by a GPU that is not there.
    assert stats[0]["energy_wh"] == 0

    with (tmp_path / "time_series.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["Test_Name"] == "baseline_metrics"
    assert rows[0]["GPU_ID"] == "0"
    assert "GPU_Utilization(%)" in rows[0]
    assert "Memory_Used(MiB)" in rows[0]
    assert "Memory_Total(MiB)" in rows[0]


def test_time_series_appends_without_rewriting_header(tmp_path, monkeypatch):
    monitor = HardwareMonitor("MOCK")

    def run_once(test_name):
        def stop_after_first_sample(_seconds):
            monitor.running = False

        monkeypatch.setattr("monitor.time.sleep", stop_after_first_sample)
        monitor.start_collection([0], tmp_path, test_name)
        monitor.stop_collection()

    run_once("first")
    run_once("second")

    lines = (tmp_path / "time_series.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("Test_Name,Timestamp,GPU_ID")
    assert sum(1 for line in lines if line.startswith("Test_Name,")) == 1
    assert any(line.startswith("first,") for line in lines)
    assert any(line.startswith("second,") for line in lines)


def test_profile_measurement_window_excludes_setup_samples():
    monitor = HardwareMonitor("MOCK")
    monitor.history = {0: {
        "temp_core": [40, 42, 50, 51, 52],
        "temp_mem": [0, 0, 0, 0, 0],
        "pwr": [50, 60, 200, 210, 220],
        "clk_core": [210, 300, 1500, 1600, 1700],
        "fan_pct": [0, 0, 50, 50, 50],
        "volts_core": [0, 0, 0, 0, 0],
        "volts_soc": [0, 0, 0, 0, 0],
        "pcie_gen": [4, 4, 4, 4, 4],
        "pcie_width": [16, 16, 16, 16, 16],
        "throttle": ["Idle", "Idle", "None", "None", "None"],
        "gpu_util": [0, 0, 90, 95, 100],
        "mem_used": [100, 100, 1000, 1100, 1200],
        "mem_total": [12000, 12000, 12000, 12000, 12000],
        "elapsed": [0, 5, 10, 15, 20],
    }}

    stats = monitor._aggregate(measurement_window_seconds=10)[0]

    assert stats["avg_pwr"] == 210.0
    assert stats["avg_gpu_util"] == 95.0
    assert stats["min_clk"] == 1500.0
    assert stats["thermal_rise"] == 2.0


def test_nvidia_cli_gpu_count_counts_index_rows(monkeypatch):
    monkeypatch.setattr("monitor.shutil.which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr("monitor.pynvml", None)
    monkeypatch.setattr(
        "monitor.subprocess.check_output",
        lambda _cmd: b"0\n1\n2\n",
    )

    monitor = HardwareMonitor("CUDA")

    assert monitor.get_gpu_count() == 3


def test_hip_monitor_does_not_use_nvidia_devices(monkeypatch):
    monkeypatch.setattr(
        "monitor.shutil.which",
        lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None,
    )
    monkeypatch.setattr("monitor.pynvml", None)
    monkeypatch.setattr(
        "monitor.subprocess.check_output",
        lambda _cmd: (_ for _ in ()).throw(
            AssertionError("HIP monitoring should not call nvidia-smi")
        ),
    )

    monitor = HardwareMonitor("HIP")

    assert monitor.get_gpu_count() == 1


def test_amd_poll_prefers_junction_temperature_over_edge(monkeypatch):
    monkeypatch.setattr("monitor.shutil.which", lambda name: "/usr/bin/rocm-smi" if name == "rocm-smi" else None)
    monkeypatch.setattr(
        "monitor.subprocess.check_output",
        lambda _cmd: json.dumps({
            "card0": {
                "Temperature (Sensor junction) (C)": "91.0",
                "Temperature (Sensor edge) (C)": "67.0",
                "Temperature (Sensor memory) (C)": "82.0",
            },
        }).encode(),
    )

    monitor = HardwareMonitor("HIP")
    monitor.history = {0: {
        "temp_core": [], "temp_mem": [], "pwr": [], "clk_core": [],
        "fan_pct": [], "volts_core": [], "volts_soc": [],
        "pcie_gen": [], "pcie_width": [], "throttle": [],
    }}

    monitor._poll_amd([0])

    assert monitor.history[0]["temp_core"] == [91.0]
    assert monitor.history[0]["temp_mem"] == [82.0]


def test_absent_sensor_reports_na_not_zero():
    """A sensor the card does not have must not become a reading of zero.

    Appending 0 when NVML raised made "no memory-temperature sensor"
    indistinguishable from "VRAM at 0 C". The published leaderboard carried
    that zero for 1234 of 1315 rows, next to guidance telling readers to keep
    memory temperature under 100 C.
    """
    mon = HardwareMonitor("MOCK")
    mon.history = {
        0: {
            "temp_core": [60.0, 62.0, 64.0],
            "temp_mem": [],          # this card has no memory temperature sensor
            "pwr": [100.0, 110.0, 120.0],
            "clk_core": [1800, 1800, 1750],
            "fan_pct": [40, 42, 44],
            "gpu_util": [99, 99, 98],
            "elapsed": [0.0, 1.0, 2.0],
            "throttle": ["None", "None", "None"],
            "pcie_gen": [4, 4, 4],
            "pcie_width": [16, 16, 16],
            "temp_hotspot": [],
            "volts_core": [],
            "volts_soc": [],
            "mem_used": [100, 100, 100],
        }
    }
    stats = mon._aggregate()[0]

    assert stats["avg_mem_temp"] == "N/A"
    assert stats["max_mem_temp"] == "N/A"
    # Sensors that did report must still produce numbers.
    assert stats["avg_temp"] == 62.0
    assert stats["max_pwr"] == 120.0


def test_efficiency_survives_an_absent_power_sensor():
    """avg_pwr is "N/A" when no power sensor exists, and comparing a string to
    a number raises in Python 3. This crashed the run rather than skipping the
    efficiency figure."""
    import pantheon

    row = pantheon.build_result_row("memory_read", 0, 10, 50, 340.0, "GB/s",
                                    {"avg_pwr": "N/A"})
    assert row is not None
    row_ok = pantheon.build_result_row("memory_read", 0, 10, 50, 340.0, "GB/s",
                                       {"avg_pwr": 250.0})
    assert row_ok is not None
