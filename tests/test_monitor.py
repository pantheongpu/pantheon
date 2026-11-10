import csv
import json

import pytest

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


# --- NVIDIA sensor decoding ---------------------------------------------------
# The published v1.2.x data had these consequences: 0 of 5,630 NVIDIA rows carry
# a memory temperature (H100, GH200 and L40S all have the sensor), and 120 of the
# 186 rows measured at 85 C or hotter report no throttling at all.

import types


def _history():
    return {key: [] for key in (
        "temp_core", "temp_mem", "pwr", "clk_core", "fan_pct", "volts_core",
        "volts_soc", "pcie_gen", "pcie_width", "throttle", "gpu_util",
        "mem_used", "mem_total", "elapsed", "clk_mem", "temp_edge", "temp_hotspot")}


def _fake_nvml(mask=0, memory_temp=71):
    """Just enough of pynvml, with NVML's real argument rules.

    `nvmlDeviceGetTemperature` accepts only NVML_TEMPERATURE_GPU; NVML defines no
    other sensor (NVML_TEMPERATURE_COUNT is 1), so anything else is refused the
    way the driver refuses it.
    """
    def temperature(handle, sensor):
        if sensor != 0:
            raise RuntimeError("NVML_ERROR_INVALID_ARGUMENT")
        return 60

    def field_values(handle, ids):
        fields = []
        for field_id in ids:
            value = types.SimpleNamespace(uiVal=memory_temp, ullVal=memory_temp, dVal=0.0)
            supported = field_id == 0x52 and memory_temp is not None
            fields.append(types.SimpleNamespace(
                fieldId=field_id, valueType=1, value=value,
                nvmlReturn=0 if supported else 3))
        return fields

    return types.SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda index: object(),
        NVML_TEMPERATURE_GPU=0, NVML_CLOCK_GRAPHICS=0, NVML_SUCCESS=0,
        NVML_FI_DEV_MEMORY_TEMP=0x52, NVML_VALUE_TYPE_UNSIGNED_INT=1,
        nvmlDeviceGetTemperature=temperature,
        nvmlDeviceGetFieldValues=field_values,
        nvmlDeviceGetPowerUsage=lambda h: 250000,
        nvmlDeviceGetClockInfo=lambda h, clock: 1980,
        nvmlDeviceGetUtilizationRates=lambda h: types.SimpleNamespace(gpu=99),
        nvmlDeviceGetMemoryInfo=lambda h: types.SimpleNamespace(used=1 << 30, total=80 << 30),
        nvmlDeviceGetFanSpeed=lambda h: 40,
        nvmlDeviceGetCurrPcieLinkGeneration=lambda h: 4,
        nvmlDeviceGetCurrPcieLinkWidth=lambda h: 16,
        nvmlDeviceGetCurrentClocksThrottleReasons=lambda h: mask,
        nvmlClocksThrottleReasonGpuIdle=0x1,
        nvmlClocksThrottleReasonSwPowerCap=0x4,
        nvmlClocksThrottleReasonHwSlowdown=0x8,
    )


def _nvml_monitor(monkeypatch, **fake):
    monkeypatch.setattr("monitor.shutil.which",
                        lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr("monitor.pynvml", _fake_nvml(**fake))
    mon = HardwareMonitor("CUDA")
    mon.history = {0: _history()}
    mon._poll_nvidia([0])
    return mon.history[0]


def _cli_monitor(monkeypatch, line):
    monkeypatch.setattr("monitor.shutil.which",
                        lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr("monitor.pynvml", None)
    monkeypatch.setattr("monitor.subprocess.check_output", lambda _cmd: line.encode())
    mon = HardwareMonitor("CUDA")
    mon.history = {0: _history()}
    mon._poll_nvidia([0])
    return mon.history[0]


def test_nvml_reads_memory_temperature_from_its_field_value(monkeypatch):
    # Sensor index 2 does not exist in NVML, so every read raised and was
    # skipped: memory temperature was never collected on the path real rigs use.
    assert _nvml_monitor(monkeypatch)["temp_mem"] == [71.0]


def test_a_card_without_a_memory_sensor_still_reports_none(monkeypatch):
    assert _nvml_monitor(monkeypatch, memory_temp=None)["temp_mem"] == []


def test_nvml_names_software_thermal_slowdown(monkeypatch):
    # 0x20 is the thermal throttle a card reaches first. It decoded as "None".
    assert _nvml_monitor(monkeypatch, mask=0x20)["throttle"] == ["Thermal"]


def test_nvml_names_hardware_thermal_slowdown_and_the_power_brake(monkeypatch):
    assert _nvml_monitor(monkeypatch, mask=0x40)["throttle"] == ["Thermal"]
    assert _nvml_monitor(monkeypatch, mask=0x80)["throttle"] == ["Power Brake"]


def test_cli_fallback_reads_the_power_cap_from_its_own_bit(monkeypatch):
    # The fallback tested 0x2 -- the application clock setting -- for "Power",
    # and never looked at 0x4, where NVML reports the power cap.
    line = "0, 60, 71, 250.0, 1980, 40, 4, 16, 0x0000000000000004, 99, 1024, 81559\n"
    assert _cli_monitor(monkeypatch, line)["throttle"] == ["Power"]


def test_cli_fallback_does_not_call_an_application_clock_setting_power(monkeypatch):
    line = "0, 60, 71, 250.0, 1980, 40, 4, 16, 0x0000000000000002, 99, 1024, 81559\n"
    assert _cli_monitor(monkeypatch, line)["throttle"] == ["None"]


def test_cli_fallback_records_an_absent_sensor_as_absent(monkeypatch):
    # nvidia-smi prints "[N/A]", which the fallback parsed as 0 -- the same
    # "no sensor" reported as "0 C" that the NVML path was fixed for.
    line = "0, 60, [N/A], [N/A], 1980, 40, 4, 16, 0x0000000000000000, 99, 1024, 81559\n"
    history = _cli_monitor(monkeypatch, line)
    assert history["temp_mem"] == []
    assert history["pwr"] == []
    assert history["temp_core"] == [60.0]


# --- Missing sensors are N/A, never 0 ------------------------------------------

def _raise(*_args, **_kwargs):
    raise RuntimeError("NVML_ERROR_NOT_SUPPORTED")


def test_nvml_skips_fan_pcie_and_memory_when_unreadable(monkeypatch):
    fake = _fake_nvml()
    fake.nvmlDeviceGetFanSpeed = _raise
    fake.nvmlDeviceGetCurrPcieLinkGeneration = _raise
    fake.nvmlDeviceGetCurrPcieLinkWidth = _raise
    fake.nvmlDeviceGetMemoryInfo = _raise
    monkeypatch.setattr("monitor.shutil.which",
                        lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr("monitor.pynvml", fake)
    mon = HardwareMonitor("CUDA")
    mon.history = {0: _history()}
    mon._poll_nvidia([0])
    h = mon.history[0]
    for key in ("fan_pct", "pcie_gen", "pcie_width", "mem_used", "mem_total",
                "volts_core", "volts_soc"):
        assert h[key] == [], key
    assert h["pwr"] == [250.0]

    stats = mon._aggregate()[0]
    for key in ("max_fan", "pcie_gen", "pcie_width", "peak_mem_used", "mem_total",
                "max_volts_core", "max_volts_soc"):
        assert stats[key] == "N/A", key


def test_nvml_still_reports_fan_pcie_and_memory_when_readable(monkeypatch):
    h = _nvml_monitor(monkeypatch)
    assert h["fan_pct"] == [40]
    assert h["pcie_gen"] == [4]
    assert h["pcie_width"] == [16]
    assert h["mem_total"] == [80 * 1024.0]


def test_cli_fallback_records_absent_fan_pcie_and_memory_as_absent(monkeypatch):
    line = "0, 60, 71, 250.0, 1980, [N/A], [N/A], [N/A], 0x0, [N/A], [N/A], [N/A]\n"
    h = _cli_monitor(monkeypatch, line)
    for key in ("fan_pct", "pcie_gen", "pcie_width", "gpu_util", "mem_used", "mem_total"):
        assert h[key] == [], key
    assert h["temp_core"] == [60.0]


def test_cli_fallback_reads_fan_pcie_and_memory_when_present(monkeypatch):
    line = "0, 60, 71, 250.0, 1980, 40, 4, 16, 0x0, 99, 1024, 81559\n"
    h = _cli_monitor(monkeypatch, line)
    assert h["fan_pct"] == [40.0]
    assert h["pcie_gen"] == [4.0]
    assert h["pcie_width"] == [16.0]
    assert h["mem_used"] == [1024.0]
    assert h["mem_total"] == [81559.0]


def test_empty_history_aggregates_to_na_everywhere_but_energy():
    mon = HardwareMonitor("MOCK")
    mon.history = {0: _history()}
    stats = mon._aggregate()[0]
    for key in ("avg_temp", "max_temp", "avg_mem_temp", "max_mem_temp", "avg_pwr",
                "avg_clk", "min_clk", "avg_gpu_util", "peak_mem_used", "mem_total",
                "thermal_rise", "max_fan", "max_volts_core", "max_volts_soc",
                "pcie_gen", "pcie_width", "throttle_reason"):
        assert stats[key] == "N/A", key


def test_time_series_csv_writes_na_for_sensors_never_read(tmp_path, monkeypatch):
    monitor = HardwareMonitor("MOCK")
    monkeypatch.setattr("monitor.time.sleep", lambda _s: setattr(monitor, "running", False))
    monitor.start_collection([0], tmp_path, "t")
    monitor.stop_collection()
    with (tmp_path / "time_series.csv").open(newline="", encoding="utf-8") as f:
        row = list(csv.DictReader(f))[0]
    for column in ("Temp_Core(C)", "Temp_Mem(C)", "Power(W)", "Fan(%)", "PCIe_Gen",
                   "GPU_Utilization(%)", "Memory_Used(MiB)", "Memory_Total(MiB)"):
        assert row[column] == "N/A", column


def _amd_monitor(monkeypatch, card, fail_usage_flags=False):
    """A HIP monitor whose rocm-smi prints `card`; records the argument lists."""
    calls = []

    def fake_check_output(cmd):
        calls.append(list(cmd))
        if fail_usage_flags and "--showuse" in cmd:
            raise RuntimeError("unrecognized arguments: --showuse")
        return json.dumps({"card0": card}).encode()

    monkeypatch.setattr("monitor.shutil.which",
                        lambda name: "/usr/bin/rocm-smi" if name == "rocm-smi" else None)
    monkeypatch.setattr("monitor.subprocess.check_output", fake_check_output)
    mon = HardwareMonitor("HIP")
    # These tests are about rocm-smi's output. Point sysfs at nothing so a host that
    # has an AMD card does not add fan, voltage and link readings of its own.
    mon.sysfs_drm_root = mon.sysfs_pci_root = "/nonexistent-sysfs"
    mon.history = {0: _history()}
    mon._poll_amd([0])
    return mon, calls


AMD_FULL = {
    "Temperature (Sensor junction) (C)": "70.0",
    "Temperature (Sensor edge) (C)": "60.0",
    "Temperature (Sensor memory) (C)": "66.0",
    "Average Graphics Package Power (W)": "300.0",
    "sclk clock speed:": "(2100Mhz)",
    "GPU use (%)": "97",
    "VRAM Total Memory (B)": str(16 * 1024 ** 3),
    "VRAM Total Used Memory (B)": str(2 * 1024 ** 3),
}


def test_amd_poll_asks_for_utilization_and_vram_and_reads_them(monkeypatch):
    mon, calls = _amd_monitor(monkeypatch, AMD_FULL)
    assert "--showuse" in calls[0] and "--showmeminfo" in calls[0] and "--json" in calls[0]
    assert "--showvoltage" in calls[0] and "--showbus" in calls[0]
    h = mon.history[0]
    assert h["gpu_util"] == [97.0]
    assert h["mem_used"] == [2048.0]      # bytes reported, MiB stored
    assert h["mem_total"] == [16384.0]
    stats = mon._aggregate()[0]
    assert stats["avg_gpu_util"] == 97.0
    assert stats["mem_total"] == 16384.0


def test_amd_poll_missing_sensors_are_absent_not_zero(monkeypatch):
    card = {"Temperature (Sensor edge) (C)": "55.0"}
    mon, _ = _amd_monitor(monkeypatch, card)
    h = mon.history[0]
    assert h["temp_core"] == [55.0]
    for key in ("temp_mem", "pwr", "clk_core", "gpu_util", "mem_used", "mem_total", "fan_pct"):
        assert h[key] == [], key
    stats = mon._aggregate()[0]
    assert stats["avg_mem_temp"] == "N/A"
    assert stats["mem_total"] == "N/A"
    assert stats["avg_gpu_util"] == "N/A"


def test_amd_poll_falls_back_when_usage_flags_are_rejected(monkeypatch):
    mon, calls = _amd_monitor(monkeypatch, AMD_FULL, fail_usage_flags=True)
    assert len(calls) == 2 and "--showuse" not in calls[1]
    h = mon.history[0]
    assert h["temp_core"] == [70.0]
    assert h["pwr"] == [300.0]


# --- AMD: PCIe link, voltage rails and fan from sysfs, rocm-smi for the rest ---------
# The layout below is what an RX 6800 shows under /sys (read on a real host): the PCI
# device directory holds current_link_*, and its hwmon/hwmon<N> holds the rails and fan.

def _fake_amd_device(root, bus="0000:03:00.0", speed="16.0 GT/s PCIe", width="16",
                     rails=(("in0", "vddgfx", "768"),), pwm="128", pwm_max="255", hwmon_name="amdgpu"):
    """A fake /sys/bus/pci/devices/<bus> tree; returns the PCI root."""
    device = root / bus
    hwmon = device / "hwmon" / "hwmon8"
    hwmon.mkdir(parents=True)
    (device / "vendor").write_text("0x1002\n")
    for name, value in (("current_link_speed", speed), ("current_link_width", width)):
        if value is not None:
            (device / name).write_text(value + "\n")
    (hwmon / "name").write_text(hwmon_name + "\n")
    for prefix, label, millivolts in rails:
        (hwmon / f"{prefix}_label").write_text(label + "\n")
        (hwmon / f"{prefix}_input").write_text(millivolts + "\n")
    if pwm is not None:
        (hwmon / "pwm1").write_text(pwm + "\n")
        (hwmon / "pwm1_max").write_text(pwm_max + "\n")
    return root


def test_amd_sysfs_readings_report_link_rails_and_fan(tmp_path):
    from monitor import amd_sysfs_readings

    pci = _fake_amd_device(tmp_path, rails=(("in0", "vddgfx", "768"), ("in1", "vddnb", "1015")))

    assert amd_sysfs_readings(str(pci / "0000:03:00.0")) == {
        "pcie_gen": 4, "pcie_width": 16, "volts_core": 768.0, "volts_soc": 1015.0, "fan_pct": 50.2}


@pytest.mark.parametrize("speed,gen", [("2.5 GT/s PCIe", 1), ("5.0 GT/s PCIe", 2), ("8.0 GT/s PCIe", 3),
                                       ("16.0 GT/s PCIe", 4), ("32.0 GT/s PCIe", 5), ("64.0 GT/s PCIe", 6)])
def test_amd_sysfs_link_speed_maps_to_the_pcie_generation(tmp_path, speed, gen):
    from monitor import amd_sysfs_readings

    pci = _fake_amd_device(tmp_path, speed=speed, rails=(), pwm=None)

    assert amd_sysfs_readings(str(pci / "0000:03:00.0"))["pcie_gen"] == gen


def test_amd_sysfs_leaves_out_what_is_absent_unreadable_or_zero(tmp_path):
    from monitor import amd_sysfs_readings

    # Unknown link speed, a zero width, a zero rail, no fan: nothing was measured.
    pci = _fake_amd_device(tmp_path, speed="Unknown speed", width="0",
                           rails=(("in0", "vddgfx", "0"),), pwm=None)
    assert amd_sysfs_readings(str(pci / "0000:03:00.0")) == {}

    # A file the kernel refuses to read (EBUSY while the card sleeps) is skipped, not fatal:
    # a directory stands in for it, which raises an OSError the same way.
    device = pci / "0000:03:00.0"
    (device / "current_link_speed").unlink()
    (device / "current_link_speed").mkdir()
    assert "pcie_gen" not in amd_sysfs_readings(str(device))
    assert amd_sysfs_readings(None) == {}
    assert amd_sysfs_readings(str(tmp_path / "no-such-device")) == {}


def test_amd_sysfs_ignores_another_drivers_hwmon(tmp_path):
    from monitor import amd_sysfs_readings

    pci = _fake_amd_device(tmp_path, hwmon_name="nvme", rails=(("in0", "vddgfx", "768"),))

    readings = amd_sysfs_readings(str(pci / "0000:03:00.0"))
    assert "volts_core" not in readings and "fan_pct" not in readings


def test_amd_device_dir_finds_the_card_by_bus_address_or_by_rank_among_amd_cards(tmp_path):
    from monitor import amd_device_dir

    pci = _fake_amd_device(tmp_path / "pci")
    assert amd_device_dir("0000:03:00.0", pci_root=str(pci)) == str(pci / "0000:03:00.0")
    assert amd_device_dir("0000:99:00.0", pci_root=str(pci)) is None

    # Without an address: rank among AMD cards, not the card number. card0 here is a BMC's
    # VGA adapter, so the first AMD GPU is card1 and the connector entry is ignored.
    drm = tmp_path / "drm"
    for name, vendor in (("card0", "0x1a03"), ("card1", "0x1002"), ("card2", "0x1002")):
        (drm / name / "device").mkdir(parents=True)
        (drm / name / "device" / "vendor").write_text(vendor + "\n")
    (drm / "card1-DP-1").mkdir()
    assert amd_device_dir(rank=0, drm_root=str(drm)) == str(drm / "card1" / "device")
    assert amd_device_dir(rank=1, drm_root=str(drm)) == str(drm / "card2" / "device")
    assert amd_device_dir(rank=2, drm_root=str(drm)) is None


def _amd_sysfs_monitor(monkeypatch, tmp_path, card, pci_root=None):
    """A HIP monitor whose rocm-smi prints `card` and whose sysfs is `pci_root` (or empty)."""
    monkeypatch.setattr("monitor.shutil.which", lambda name: "/usr/bin/rocm-smi" if name == "rocm-smi" else None)
    monkeypatch.setattr("monitor.subprocess.check_output", lambda _cmd: json.dumps({"card0": card}).encode())
    monitor = HardwareMonitor("HIP")
    monitor.sysfs_pci_root = str(pci_root or tmp_path / "no-pci")
    monitor.sysfs_drm_root = str(tmp_path / "no-drm")
    monitor.history = {0: _history()}
    return monitor


def test_amd_poll_records_link_voltage_and_fan(monkeypatch, tmp_path):
    pci = _fake_amd_device(tmp_path, rails=(("in0", "vddgfx", "768"), ("in1", "vddnb", "1015")))
    monitor = _amd_sysfs_monitor(monkeypatch, tmp_path, {
        "PCI Bus": "0000:03:00.0", "Voltage (mV)": "862", "Fan speed (%)": "45",
        "Temperature (Sensor edge) (C)": "60.0"}, pci_root=pci)

    monitor._poll_amd([0])
    history = monitor.history[0]

    assert history["pcie_gen"] == [4] and history["pcie_width"] == [16]
    assert history["volts_core"] == [862.0]        # rocm-smi answered, so sysfs does not override it
    assert history["volts_soc"] == [1015.0]        # only sysfs has the SoC rail
    assert history["fan_pct"] == [45.0]            # the % sits in the key, not the value


def test_amd_poll_reads_a_fan_percent_printed_in_the_value(monkeypatch, tmp_path):
    for printed in ("45%", "Fan Level: 115 (45%)"):
        monitor = _amd_sysfs_monitor(monkeypatch, tmp_path, {"Fan speed": printed, "Fan Level": "115"})
        monitor._poll_amd([0])
        assert monitor.history[0]["fan_pct"] == [45.0], printed


def test_amd_poll_falls_back_to_sysfs_when_rocm_smi_prints_neither_fan_nor_voltage(monkeypatch, tmp_path):
    pci = _fake_amd_device(tmp_path)
    monitor = _amd_sysfs_monitor(monkeypatch, tmp_path, {"PCI Bus": "0000:03:00.0"}, pci_root=pci)

    monitor._poll_amd([0])

    assert monitor.history[0]["fan_pct"] == [50.2] and monitor.history[0]["volts_core"] == [768.0]


def test_amd_card_without_these_sensors_reports_na_never_zero(monkeypatch, tmp_path):
    # A card with no fan, no readable rails and no PCIe attributes (a datacenter card,
    # a sleeping one): every series stays empty and the aggregate says N/A.
    monitor = _amd_sysfs_monitor(monkeypatch, tmp_path, {"Temperature (Sensor edge) (C)": "55.0"})
    monitor._poll_amd([0])
    monitor._poll_amd([0])
    for series in ("fan_pct", "volts_core", "volts_soc", "pcie_gen", "pcie_width", "throttle"):
        assert monitor.history[0][series] == [], series

    monitor.history[0]["elapsed"] = [0.0, 1.0]
    stats = monitor._aggregate()[0]

    for key in ("max_fan", "max_volts_core", "max_volts_soc", "pcie_gen", "pcie_width", "throttle_reason"):
        assert stats[key] == "N/A", key
    assert stats["max_temp"] == 55.0


def test_amd_link_downshifted_at_idle_does_not_hide_the_link_under_load(monkeypatch, tmp_path):
    pci = _fake_amd_device(tmp_path, speed="2.5 GT/s PCIe")
    monitor = _amd_sysfs_monitor(monkeypatch, tmp_path, {"PCI Bus": "0000:03:00.0"}, pci_root=pci)
    monitor._poll_amd([0])                                           # idle sample: gen 1
    (pci / "0000:03:00.0" / "current_link_speed").write_text("16.0 GT/s PCIe\n")
    monitor._poll_amd([0])                                           # sample under load: gen 4
    monitor.history[0]["elapsed"] = [0.0, 1.0]
    monitor.history[0]["temp_core"] = [50.0, 51.0]

    assert monitor._aggregate()[0]["pcie_gen"] == 4.0


class _FakePsutil:
    """psutil as the monitor sees it: per-core load and one clock reading."""

    def __init__(self, per_core, mhz):
        self.per_core, self.mhz = per_core, mhz

    def cpu_percent(self, percpu=False):
        return list(self.per_core)

    def cpu_freq(self):
        class Freq:
            current = self.mhz
        return Freq()


def _one_tick(monitor, monkeypatch, tmp_path):
    monkeypatch.setattr(HardwareMonitor, "HOST_WARMUP_S", 0)
    def stop_after_first_sample(_seconds):
        monitor.running = False

    monkeypatch.setattr("monitor.time.sleep", stop_after_first_sample)
    monitor.start_collection([0], tmp_path, "scheduler")
    return monitor.stop_collection()


def test_host_cpu_is_sampled_with_the_busiest_core(tmp_path, monkeypatch):
    # One launch thread saturating one core of four: the machine-wide average
    # hides it, the busiest core shows it.
    monkeypatch.setattr("monitor.psutil", _FakePsutil([100.0, 4.0, 2.0, 2.0], 3100.0))
    stats = _one_tick(HardwareMonitor("MOCK"), monkeypatch, tmp_path)[0]

    assert stats["avg_cpu_util"] == 27.0
    assert stats["avg_cpu_busiest_core"] == 100.0
    assert stats["max_cpu_busiest_core"] == 100.0
    assert stats["avg_cpu_clk"] == 3100.0

    with (tmp_path / "time_series.csv").open(newline="", encoding="utf-8") as f:
        row = list(csv.DictReader(f))[0]
    assert row["CPU_Utilization(%)"] == "27.0"
    assert row["CPU_Busiest_Core(%)"] == "100.0"
    assert row["CPU_Clock(MHz)"] == "3100.0"


def test_host_cpu_is_na_without_psutil(tmp_path, monkeypatch):
    monkeypatch.setattr("monitor.psutil", None)
    stats = _one_tick(HardwareMonitor("MOCK"), monkeypatch, tmp_path)[0]

    for key in ("avg_cpu_util", "max_cpu_util", "avg_cpu_busiest_core",
                "max_cpu_busiest_core", "avg_cpu_clk"):
        assert stats[key] == "N/A"


def test_host_cpu_uses_the_same_measurement_window_as_the_gpu():
    monitor = HardwareMonitor("MOCK")
    monitor.history = {0: {key: [] for key in (
        "temp_core", "temp_mem", "pwr", "clk_core", "fan_pct", "volts_core",
        "volts_soc", "pcie_gen", "pcie_width", "throttle", "gpu_util",
        "mem_used", "mem_total", "elapsed")}}
    monitor.host = {
        "cpu_util": [90.0, 90.0, 20.0, 30.0],        # setup is busy, the run is not
        "cpu_busiest_core": [100.0, 100.0, 40.0, 60.0],
        "cpu_freq": [2000.0, None, 3000.0, 3200.0],  # a tick with no clock reading
        "elapsed": [0, 5, 10, 15],
    }

    stats = monitor._aggregate(measurement_window_seconds=5)[0]

    assert stats["avg_cpu_util"] == 25.0
    assert stats["max_cpu_busiest_core"] == 60.0
    assert stats["avg_cpu_clk"] == 3100.0


def test_first_host_tick_inside_the_warmup_is_not_a_reading(tmp_path, monkeypatch):
    monkeypatch.setattr("monitor.psutil", _FakePsutil([0.0, 0.0], 3000.0))
    monitor = HardwareMonitor("MOCK")

    def stop_after_first_sample(_seconds):
        monitor.running = False

    monkeypatch.setattr("monitor.time.sleep", stop_after_first_sample)
    monitor.start_collection([0], tmp_path, "scheduler")
    stats = monitor.stop_collection()[0]

    # The only tick came before the warm-up elapsed: no number is invented.
    assert stats["avg_cpu_util"] == "N/A"


def test_host_cpu_is_primed_on_the_sampling_thread(tmp_path, monkeypatch):
    # psutil keeps the cpu_percent() baseline per thread; a baseline set on
    # another thread makes the first real reading 0.0.
    import threading
    calls = []

    class ThreadAwarePsutil(_FakePsutil):
        def cpu_percent(self, percpu=False):
            calls.append(threading.get_ident())
            return super().cpu_percent(percpu)

    monkeypatch.setattr("monitor.psutil", ThreadAwarePsutil([50.0, 50.0], 3000.0))
    stats = _one_tick(HardwareMonitor("MOCK"), monkeypatch, tmp_path)[0]

    assert len(calls) == 2 and calls[0] == calls[1] != threading.get_ident()
    assert stats["avg_cpu_util"] == 50.0


# --- Memory clock, both temperature sensors, and throttle seconds by reason -----------------

def test_nvml_records_the_memory_clock_and_the_edge_temperature(monkeypatch):
    h = _nvml_monitor(monkeypatch)
    assert h["clk_mem"] == [1980]
    assert h["temp_edge"] == [60] and h["temp_core"] == [60]
    assert h["temp_hotspot"] == []        # NVML has no hotspot sensor: nothing was read, so N/A


def test_nvml_skips_the_memory_clock_when_it_cannot_be_read(monkeypatch):
    def clock(handle, which):
        if which == 2:
            raise RuntimeError("NVML_ERROR_NOT_SUPPORTED")
        return 1980

    fake = _fake_nvml()
    fake.nvmlDeviceGetClockInfo = clock
    monkeypatch.setattr("monitor.shutil.which",
                        lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr("monitor.pynvml", fake)
    mon = HardwareMonitor("CUDA")
    mon.history = {0: _history()}
    mon._poll_nvidia([0])
    assert mon.history[0]["clk_mem"] == [] and mon.history[0]["clk_core"] == [1980]


def test_cli_fallback_reads_the_memory_clock_when_nvidia_smi_prints_it(monkeypatch):
    h = _cli_monitor(monkeypatch, "0, 60, 71, 250.0, 1980, 40, 4, 16, 0x0, 99, 1024, 81559, 9501\n")
    assert h["clk_mem"] == [9501.0] and h["temp_edge"] == [60.0]
    h = _cli_monitor(monkeypatch, "0, 60, 71, 250.0, 1980, 40, 4, 16, 0x0, 99, 1024, 81559, [N/A]\n")
    assert h["clk_mem"] == []
    # A twelve-field answer (an older query) still parses.
    h = _cli_monitor(monkeypatch, "0, 60, 71, 250.0, 1980, 40, 4, 16, 0x0, 99, 1024, 81559\n")
    assert h["clk_mem"] == [] and h["temp_core"] == [60.0]


def test_amd_poll_keeps_edge_and_hotspot_apart_and_reads_the_memory_clock(monkeypatch):
    mon, _ = _amd_monitor(monkeypatch, {
        "Temperature (Sensor edge) (C)": "61.0", "Temperature (Sensor junction) (C)": "74.0",
        "sclk clock speed:": "(2600Mhz)", "mclk clock speed:": "(1000Mhz)", "mclk clock level:": "3",
    })
    h = mon.history[0]
    assert h["temp_edge"] == [61.0] and h["temp_hotspot"] == [74.0]
    assert h["temp_core"] == [74.0]       # unchanged: the published "Peak Temp" is still the hotspot on AMD
    assert h["clk_mem"] == [1000.0] and h["clk_core"] == [2600.0]


def test_amd_poll_without_a_hotspot_sensor_leaves_it_empty(monkeypatch):
    mon, _ = _amd_monitor(monkeypatch, {"Temperature (Sensor edge) (C)": "61.0"})
    assert mon.history[0]["temp_edge"] == [61.0] and mon.history[0]["temp_hotspot"] == []


def test_amd_sysfs_reports_the_memory_clock_from_the_mclk_label(tmp_path):
    from monitor import amd_sysfs_readings

    pci = _fake_amd_device(tmp_path, rails=(), pwm=None)
    hwmon = pci / "0000:03:00.0" / "hwmon" / "hwmon8"
    (hwmon / "freq1_label").write_text("sclk\n")
    (hwmon / "freq1_input").write_text("2600000000\n")
    (hwmon / "freq2_label").write_text("mclk\n")
    (hwmon / "freq2_input").write_text("1000000000\n")
    assert amd_sysfs_readings(str(pci / "0000:03:00.0"))["clk_mem"] == 1000.0
    (hwmon / "freq2_input").write_text("0\n")
    assert "clk_mem" not in amd_sysfs_readings(str(pci / "0000:03:00.0"))


def _aggregate_of(**series):
    mon = HardwareMonitor("MOCK")
    base = _history()
    for key, values in series.items():
        base[key] = values
    if "elapsed" not in series:
        base["elapsed"] = list(range(len(next(iter(series.values())))))
    mon.history = {0: base}
    return mon


def test_throttle_seconds_are_counted_per_reason():
    mon = _aggregate_of(throttle=["None", "Power", "Power|Thermal", "Thermal", "Power Brake"])
    stats = mon._aggregate()[0]
    assert stats["throttle_power_s"] == 2.0       # "Power" and "Power|Thermal"; "Power Brake" is its own reason
    assert stats["throttle_thermal_s"] == 2.0
    assert stats["throttle_brake_s"] == 1.0


def test_throttle_seconds_are_na_when_the_card_never_reported_a_reason():
    stats = _aggregate_of(throttle=["N/A", "N/A", "N/A"])._aggregate()[0]
    assert (stats["throttle_power_s"], stats["throttle_thermal_s"], stats["throttle_brake_s"]) == ("N/A",) * 3
    # ...but a card that did report, and was never held back, has zero seconds, not N/A.
    stats = _aggregate_of(throttle=["None", "None"])._aggregate()[0]
    assert stats["throttle_power_s"] == 0.0


def test_thermal_margin_is_the_slowdown_temperature_minus_the_hottest_reading():
    mon = _aggregate_of(temp_core=[70, 84, 80])
    mon.temp_limits = {0: 88}
    assert mon._aggregate()[0]["thermal_margin"] == 4.0
    mon.temp_limits = {0: 80}
    assert mon._aggregate()[0]["thermal_margin"] == -4.0      # past the slowdown temperature: negative, not hidden
    for unknown in ({}, {0: "N/A"}, {0: None}, {1: 88}):
        mon.temp_limits = unknown
        assert mon._aggregate()[0]["thermal_margin"] == "N/A", unknown
    quiet = _aggregate_of(temp_core=[])
    quiet.temp_limits = {0: 88}
    assert quiet._aggregate()[0]["thermal_margin"] == "N/A"


def test_start_state_is_the_first_reading_not_the_run_average():
    mon = _aggregate_of(temp_core=[41, 70, 80], pwr=[30.0, 200.0, 210.0], gpu_util=[0, 99, 100],
                        mem_used=[500, 9000, 9100], clk_core=[210, 1700, 1700], clk_mem=[405, 1215, 1215])
    stats = mon._aggregate()[0]
    assert (stats["start_temp"], stats["start_pwr"], stats["start_gpu_util"]) == (41.0, 30.0, 0.0)
    assert (stats["start_mem_used"], stats["start_clk"], stats["start_mem_clk"]) == (500.0, 210.0, 405.0)
    assert stats["avg_mem_clk"] == 945.0 and stats["max_mem_clk"] == 1215.0
    empty = _aggregate_of(temp_core=[])._aggregate()[0]
    assert empty["start_temp"] == empty["start_mem_clk"] == empty["avg_mem_clk"] == "N/A"


def test_max_edge_and_hotspot_temperatures_are_separate_columns():
    stats = _aggregate_of(temp_core=[70, 80], temp_edge=[60, 66], temp_hotspot=[70, 80])._aggregate()[0]
    assert stats["max_edge_temp"] == 66.0 and stats["max_hotspot_temp"] == 80.0
    stats = _aggregate_of(temp_core=[70], temp_edge=[70])._aggregate()[0]
    assert stats["max_edge_temp"] == 70.0 and stats["max_hotspot_temp"] == "N/A"


def _violation_nvml(counters):
    """A fake NVML whose violation counters step through `counters` (policy -> nanoseconds per call)."""
    import types

    fake = _fake_nvml()
    calls = {}

    def status(handle, policy):
        values = counters.get(policy)
        if values is None:
            raise RuntimeError("NVML_ERROR_NOT_SUPPORTED")
        index = calls.get(policy, 0)
        calls[policy] = index + 1
        return types.SimpleNamespace(referenceTime=0, violationTime=values[min(index, len(values) - 1)])

    fake.nvmlDeviceGetViolationStatus = status
    fake.nvmlDeviceGetComputeRunningProcesses = lambda h: [types.SimpleNamespace(pid=10), types.SimpleNamespace(pid=11)]
    fake.nvmlDeviceGetGraphicsRunningProcesses = lambda h: [types.SimpleNamespace(pid=11)]
    return fake


def _nvml_context_monitor(monkeypatch, fake):
    monkeypatch.setattr("monitor.shutil.which",
                        lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr("monitor.pynvml", fake)
    mon = HardwareMonitor("CUDA")
    mon._begin_context([0])
    mon._end_context()
    mon.history = {0: _history()}
    return mon._aggregate()[0]


def test_violation_counters_give_the_seconds_held_back_between_start_and_end(monkeypatch):
    stats = _nvml_context_monitor(monkeypatch, _violation_nvml({0: [5_000_000_000, 6_500_000_000], 1: [0, 0]}))
    assert stats["power_violation_s"] == 1.5
    assert stats["thermal_violation_s"] == 0.0
    # Two processes already held the card (pid 11 is in both lists and counts once).
    assert stats["other_gpu_processes"] == 2


def test_violation_counters_are_na_where_the_driver_does_not_support_them(monkeypatch):
    stats = _nvml_context_monitor(monkeypatch, _violation_nvml({}))
    assert stats["power_violation_s"] == stats["thermal_violation_s"] == "N/A"


def test_other_processes_and_violations_are_na_without_nvml():
    mon = HardwareMonitor("MOCK")
    mon._begin_context([0])
    mon._end_context()
    mon.history = {0: _history()}
    stats = mon._aggregate()[0]
    assert stats["other_gpu_processes"] == "N/A"
    assert stats["power_violation_s"] == stats["thermal_violation_s"] == "N/A"


def test_host_load_at_start_is_recorded_and_na_when_the_platform_has_none(monkeypatch):
    mon = HardwareMonitor("MOCK")
    monkeypatch.setattr("monitor.os.getloadavg", lambda: (3.456, 2.0, 1.0))
    mon._begin_context([0])
    assert mon._host_stats()["host_load_start"] == 3.46

    def no_loadavg():
        raise OSError("no loadavg")

    monkeypatch.setattr("monitor.os.getloadavg", no_loadavg)
    mon._begin_context([0])
    assert mon._host_stats()["host_load_start"] == "N/A"


def test_time_series_csv_carries_the_new_columns_at_the_end(tmp_path, monkeypatch):
    monitor = HardwareMonitor("MOCK")
    monkeypatch.setattr("monitor.time.sleep", lambda _s: setattr(monitor, "running", False))
    monitor.start_collection([0], tmp_path, "x")
    monitor.stop_collection()
    with (tmp_path / "time_series.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)
    assert header[-3:] == ["Mem_Clock(MHz)", "Temp_Edge(C)", "Temp_Hotspot(C)"]
    assert row[-3:] == ["N/A", "N/A", "N/A"]          # nothing was read on the mock backend


# --- Rated limits read once at start-up ------------------------------------------------------

def _spec_nvml(**overrides):
    import types

    answers = dict(
        nvmlDeviceGetMaxClockInfo=lambda h, clock: {0: 1590, 2: 5001}[clock],
        nvmlDeviceGetMemoryBusWidth=lambda h: 256,
        nvmlDeviceGetMaxPcieLinkGeneration=lambda h: 3,
        nvmlDeviceGetMaxPcieLinkWidth=lambda h: 16,
        nvmlDeviceGetNumGpuCores=lambda h: 2560,
        nvmlDeviceGetVbiosVersion=lambda h: "90.04.96.00.01",
        nvmlDeviceGetPowerManagementDefaultLimit=lambda h: 70000,
        nvmlDeviceGetEnforcedPowerLimit=lambda h: 60000,
        nvmlDeviceGetTemperatureThreshold=lambda h, kind: {0: 105, 1: 96}[kind],
        nvmlDeviceGetPersistenceMode=lambda h: 1,
        nvmlDeviceGetComputeMode=lambda h: 0,
        nvmlDeviceGetMigMode=_raise,
    )
    answers.update(overrides)
    return types.SimpleNamespace(nvmlInit=lambda: None, nvmlShutdown=lambda: None,
                                 nvmlDeviceGetHandleByIndex=lambda i: object(), **answers)


def test_nvidia_device_specs_reads_the_rated_limits_from_nvml(tmp_path):
    from monitor import DEVICE_SPEC_KEYS, nvidia_device_specs

    node = tmp_path / "0000:00:1e.0"
    node.mkdir()
    (node / "numa_node").write_text("1\n")
    specs = nvidia_device_specs(0, "00000000:00:1E.0", nvml=_spec_nvml(), smi_field=lambda i, f: None,
                                pci_root=str(tmp_path))
    assert set(specs) == set(DEVICE_SPEC_KEYS)
    assert specs["max_graphics_clock_mhz"] == 1590 and specs["max_memory_clock_mhz"] == 5001
    assert specs["memory_bus_width_bits"] == 256 and specs["cuda_cores"] == 2560
    assert (specs["pcie_max_gen"], specs["pcie_max_width"]) == (3, 16)
    assert specs["vbios_version"] == "90.04.96.00.01"
    assert (specs["power_limit_default"], specs["power_limit_enforced"]) == (70.0, 60.0)
    assert (specs["temp_slowdown_c"], specs["temp_shutdown_c"]) == (96, 105)
    assert (specs["persistence_mode"], specs["compute_mode"]) == ("Enabled", "Default")
    assert specs["mig_mode"] == "N/A" and specs["compute_units"] == "N/A"     # a T4 has no MIG; compute_units is AMD's
    assert specs["numa_node"] == 1


def test_nvidia_device_specs_falls_back_to_nvidia_smi_per_field_and_never_raises():
    import types

    from monitor import nvidia_device_specs

    asked = []

    def smi(index, field):
        asked.append(field)
        return {"clocks.max.mem": 9501, "persistence_mode": "Disabled"}.get(field)

    broken = _spec_nvml(nvmlDeviceGetMaxClockInfo=_raise, nvmlDeviceGetPersistenceMode=_raise)
    specs = nvidia_device_specs(0, nvml=broken, smi_field=smi)
    assert specs["max_memory_clock_mhz"] == 9501 and specs["persistence_mode"] == "Disabled"
    assert specs["max_graphics_clock_mhz"] == "N/A"          # neither NVML nor nvidia-smi knew
    assert "pcie.link.gen.max" not in asked                  # NVML answered it, so nvidia-smi is not asked

    unusable = types.SimpleNamespace(nvmlInit=_raise, nvmlShutdown=lambda: None)
    assert set(nvidia_device_specs(0, nvml=unusable, smi_field=lambda i, f: None).values()) == {"N/A"}


def test_numa_node_is_na_for_no_address_or_no_numa(tmp_path):
    from monitor import numa_node_of

    (tmp_path / "0000:03:00.0").mkdir()
    (tmp_path / "0000:03:00.0" / "numa_node").write_text("-1\n")
    assert numa_node_of("0000:03:00.0", str(tmp_path)) is None      # -1 is the kernel's "no node"
    assert numa_node_of("", str(tmp_path)) is None and numa_node_of(None, str(tmp_path)) is None
    assert numa_node_of("0000:99:00.0", str(tmp_path)) is None


def test_a_history_without_the_newer_series_still_polls(monkeypatch):
    # A caller that builds its own history (older code, a test) must not lose the whole poll to a missing key.
    mon, _ = _amd_monitor(monkeypatch, {"Temperature (Sensor edge) (C)": "61.0", "mclk clock speed:": "(1000Mhz)"})
    old_style = {key: [] for key in ("temp_core", "temp_mem", "pwr", "clk_core", "fan_pct", "volts_core",
                                     "volts_soc", "pcie_gen", "pcie_width", "throttle")}
    mon.history = {0: old_style}
    mon._poll_amd([0])
    assert old_style["temp_core"] == [61.0] and old_style["temp_edge"] == [61.0] and old_style["clk_mem"] == [1000.0]


# --- The start state is read before the workload launches --------------------------------------

def test_nvml_start_state_is_read_before_launch_not_from_the_first_tick(monkeypatch, tmp_path):
    temps, utils = iter([41, 70, 71]), iter([0, 99, 99])
    fake = _fake_nvml()
    fake.nvmlDeviceGetTemperature = lambda handle, sensor: next(temps)
    fake.nvmlDeviceGetUtilizationRates = lambda handle: types.SimpleNamespace(gpu=next(utils))
    monkeypatch.setattr("monitor.shutil.which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr("monitor.pynvml", fake)
    mon = HardwareMonitor("CUDA")
    monkeypatch.setattr("monitor.time.sleep", lambda _s: setattr(mon, "running", False))

    mon.start_collection([0], tmp_path, "t")
    stats = mon.stop_collection()[0]

    assert stats["start_temp"] == 41.0 and stats["start_gpu_util"] == 0.0     # before launch
    assert mon.history[0]["temp_core"] == [70] and stats["max_temp"] == 70.0   # the run's own first tick is the run
    # The scratch reading left no extra entry behind: every series is still parallel to the elapsed times.
    assert len(mon.history[0]["elapsed"]) == len(mon.history[0]["temp_core"]) == len(mon.history[0]["gpu_util"]) == 1


def test_amd_start_state_is_read_before_launch_even_though_rocm_smi_is_slow(monkeypatch, tmp_path):
    idle = {"Temperature (Sensor edge) (C)": "29.0", "Temperature (Sensor junction) (C)": "31.0", "GPU use (%)": "0",
            "VRAM Total Memory (B)": str(12 * 1024 ** 3), "VRAM Total Used Memory (B)": str(300 * 1024 ** 2),
            "mclk clock speed:": "(96Mhz)", "sclk clock speed:": "(500Mhz)"}
    busy = dict(idle, **{"GPU use (%)": "97", "VRAM Total Used Memory (B)": str(12 * 1024 ** 3 - 1024 ** 2),
                         "mclk clock speed:": "(1000Mhz)", "sclk clock speed:": "(2600Mhz)"})
    answers = iter([idle, busy, busy])
    monkeypatch.setattr("monitor.shutil.which", lambda name: "/usr/bin/rocm-smi" if name == "rocm-smi" else None)
    monkeypatch.setattr("monitor.subprocess.check_output", lambda cmd: json.dumps({"card0": next(answers)}).encode())
    mon = HardwareMonitor("HIP")
    mon.sysfs_drm_root = mon.sysfs_pci_root = "/nonexistent-sysfs"
    monkeypatch.setattr("monitor.time.sleep", lambda _s: setattr(mon, "running", False))

    mon.start_collection([0], tmp_path, "t")
    stats = mon.stop_collection()[0]

    assert (stats["start_gpu_util"], stats["start_mem_used"]) == (0.0, 300.0)          # the idle card, not the loaded one
    assert (stats["start_clk"], stats["start_mem_clk"], stats["start_temp"]) == (500.0, 96.0, 31.0)
    assert stats["avg_mem_clk"] == 1000.0 and stats["max_hotspot_temp"] == 31.0
    assert len(mon.history[0]["elapsed"]) == len(mon.history[0]["gpu_util"]) == 1


def test_a_start_reading_that_could_not_be_taken_is_na_not_the_first_run_sample(monkeypatch, tmp_path):
    # Nothing readable before launch (a sleeping card, an error), readable once the run is under way.
    # rocm-smi is asked twice per poll (the full flags, then the short ones), so both of the first poll fail.
    calls = iter([RuntimeError("busy"), RuntimeError("busy"),
                  {"Temperature (Sensor edge) (C)": "60.0"}, {"Temperature (Sensor edge) (C)": "60.0"}])

    def check_output(cmd):
        answer = next(calls)
        if isinstance(answer, Exception):
            raise answer
        return json.dumps({"card0": answer}).encode()

    monkeypatch.setattr("monitor.shutil.which", lambda name: "/usr/bin/rocm-smi" if name == "rocm-smi" else None)
    monkeypatch.setattr("monitor.subprocess.check_output", check_output)
    mon = HardwareMonitor("HIP")
    mon.sysfs_drm_root = mon.sysfs_pci_root = "/nonexistent-sysfs"
    monkeypatch.setattr("monitor.time.sleep", lambda _s: setattr(mon, "running", False))

    mon.start_collection([0], tmp_path, "t")
    stats = mon.stop_collection()[0]

    assert stats["start_temp"] == "N/A" and stats["max_temp"] == 60.0
