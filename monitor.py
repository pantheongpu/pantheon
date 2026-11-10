import subprocess
import time
import json
import shutil
import threading
import numpy as np
import csv
import glob
import os
import re

# Try to import NVIDIA native bindings
try:
    import pynvml
except ImportError:
    pynvml = None

# Host CPU sampling is optional in the same way: without psutil the CPU
# columns report "N/A" and everything else is unaffected.
try:
    import psutil
except ImportError:
    psutil = None


# NVML's clock-event reasons, by bit (nvml.h). Decoded in one place because the
# NVML and nvidia-smi paths used to decode them separately, and disagreed: the
# nvidia-smi path tested 0x2 -- the application clock setting -- for "Power" and
# never looked at 0x4, and neither path knew the thermal bits past 0x8. A card in
# software thermal slowdown (0x20, the throttle reached first) read "None": 120 of
# the 186 published v1.2.x rows measured at 85 C or hotter say no throttling.
THROTTLE_BITS = (
    (0x01, "Idle"),         # GpuIdle
    (0x04, "Power"),        # SwPowerCap
    (0x08, "Thermal"),      # HwSlowdown
    (0x20, "Thermal"),      # SwThermalSlowdown
    (0x40, "Thermal"),      # HwThermalSlowdown
    (0x80, "Power Brake"),  # HwPowerBrakeSlowdown
)

# Memory temperature is a field value, not a temperature sensor index.
NVML_FI_DEV_MEMORY_TEMP = 82

# c_nvmlValue_t members, by NVML_VALUE_TYPE.
_NVML_VALUE_MEMBERS = {0: "dVal", 1: "uiVal", 2: "ulVal", 3: "ullVal",
                       4: "sllVal", 5: "siVal", 6: "usVal"}


def decode_throttle_mask(mask):
    """Name the clock-event reasons in an NVML bitmask; "N/A" if it was unreadable."""
    if mask is None:
        return "N/A"
    labels = []
    for bit, label in THROTTLE_BITS:
        if mask & bit and label not in labels:
            labels.append(label)
    return "|".join(labels) if labels else "None"


def _nvml_field_number(field):
    """The numeric value of one NVML field reading, by its declared value type."""
    return getattr(field.value, _NVML_VALUE_MEMBERS.get(getattr(field, "valueType", 1), "uiVal"))

# Where the amdgpu driver publishes what rocm-smi does not print for every card.
SYSFS_DRM_ROOT = "/sys/class/drm"
SYSFS_PCI_ROOT = "/sys/bus/pci/devices"
# PCIe link speed in GT/s as current_link_speed prints it, by generation.
PCIE_GENERATIONS = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}
# hwmon in*_label values that name the two rails the report carries.
AMD_VOLTAGE_LABELS = {"vddgfx": "volts_core", "vddnb": "volts_soc"}

# NVML enums, spelled out so a pynvml release that renames one does not take the reading with it.
NVML_CLOCK_GRAPHICS = 0
NVML_CLOCK_MEM = 2
NVML_TEMPERATURE_THRESHOLD_SHUTDOWN = 0
NVML_TEMPERATURE_THRESHOLD_SLOWDOWN = 1
NVML_PERF_POLICY_POWER = 0
NVML_PERF_POLICY_THERMAL = 1
_NVML_COMPUTE_MODES = {0: "Default", 1: "Exclusive_Thread", 2: "Prohibited", 3: "Exclusive_Process"}
SYSFS_KFD_ROOT = "/sys/class/kfd/kfd/topology/nodes"

# What each card is rated for, read once at start-up: the denominators that turn a measured clock, link or
# temperature into "how close to the limit". Every key is always present; a value nothing could read is "N/A".
DEVICE_SPEC_KEYS = (
    "max_graphics_clock_mhz", "max_memory_clock_mhz", "memory_bus_width_bits",
    "pcie_max_gen", "pcie_max_width", "cuda_cores", "compute_units", "vbios_version",
    "persistence_mode", "compute_mode", "mig_mode", "power_limit_default", "power_limit_enforced",
    "temp_slowdown_c", "temp_shutdown_c", "numa_node",
)

# The same facts from nvidia-smi, for a host where NVML cannot answer one of them.
_SMI_SPEC_FIELDS = {
    "max_graphics_clock_mhz": "clocks.max.gr", "max_memory_clock_mhz": "clocks.max.mem",
    "pcie_max_gen": "pcie.link.gen.max", "pcie_max_width": "pcie.link.width.max",
    "vbios_version": "vbios_version", "persistence_mode": "persistence_mode",
    "compute_mode": "compute_mode", "mig_mode": "mig.mode.current",
    "power_limit_default": "power.default_limit", "power_limit_enforced": "enforced.power.limit",
}
_SMI_NO_VALUE = ("", "n/a", "[n/a]", "not supported", "[not supported]", "[unknown error]")


def _spec_number(value):
    """A number from a tool's text ("1215", "300.00", "1215 MHz"), else the text, else None."""
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    text = str(value).strip()
    if text.lower() in _SMI_NO_VALUE:
        return None
    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)(?:\s*(?:MHz|W|C|MiB))?", text, re.I)
    if match:
        number = float(match.group(1))
        return int(number) if number == int(number) else round(number, 1)
    return text


def _smi_spec_field(index, field):
    """One nvidia-smi --query-gpu field for one card, or None when it printed a placeholder or failed."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", str(index), f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=20).decode().strip().splitlines()
    except Exception:
        return None
    return _spec_number(out[0]) if out else None


def numa_node_of(bus_id, pci_root=SYSFS_PCI_ROOT):
    """The NUMA node a PCI device sits on, from its address (nvidia-smi's 00000000:00:1E.0 form works), or None."""
    text = str(bus_id or "").strip().lower()
    if not text:
        return None
    domain, _, rest = text.partition(":")
    path = os.path.join(pci_root, f"{domain[-4:].rjust(4, '0')}:{rest}" if rest else text, "numa_node")
    value = _sysfs_number(path)
    return None if value is None or value < 0 else int(value)


def nvidia_device_specs(index, bus_id=None, nvml=None, smi_field=None, pci_root=SYSFS_PCI_ROOT):
    """The rated limits and settings of one NVIDIA card: NVML first, nvidia-smi for what it cannot say.

    Always returns every DEVICE_SPEC_KEYS key. Reading is best-effort and never raises: a card or driver that
    does not answer one question (MIG on a card without it, a bus width on an old driver) reads "N/A" for it.
    """
    lib = pynvml if nvml is None else nvml
    smi = _smi_spec_field if smi_field is None else smi_field
    specs = {key: "N/A" for key in DEVICE_SPEC_KEYS}
    started, handle = False, None
    if lib is not None:
        try:
            lib.nvmlInit()
            started = True
            handle = lib.nvmlDeviceGetHandleByIndex(index)
        except Exception:
            handle = None

    def ask(name, *args):
        if handle is None:
            return None
        try:
            return getattr(lib, name)(handle, *args)
        except Exception:
            return None

    def milliwatts(name):
        value = ask(name)
        return None if value is None else round(value / 1000.0, 1)

    found = {
        "max_graphics_clock_mhz": ask("nvmlDeviceGetMaxClockInfo", NVML_CLOCK_GRAPHICS),
        "max_memory_clock_mhz": ask("nvmlDeviceGetMaxClockInfo", NVML_CLOCK_MEM),
        "memory_bus_width_bits": ask("nvmlDeviceGetMemoryBusWidth"),
        "pcie_max_gen": ask("nvmlDeviceGetMaxPcieLinkGeneration"),
        "pcie_max_width": ask("nvmlDeviceGetMaxPcieLinkWidth"),
        "cuda_cores": ask("nvmlDeviceGetNumGpuCores"),
        "vbios_version": _spec_number(ask("nvmlDeviceGetVbiosVersion")),
        "power_limit_default": milliwatts("nvmlDeviceGetPowerManagementDefaultLimit"),
        "power_limit_enforced": milliwatts("nvmlDeviceGetEnforcedPowerLimit"),
        "temp_slowdown_c": ask("nvmlDeviceGetTemperatureThreshold", NVML_TEMPERATURE_THRESHOLD_SLOWDOWN),
        "temp_shutdown_c": ask("nvmlDeviceGetTemperatureThreshold", NVML_TEMPERATURE_THRESHOLD_SHUTDOWN),
    }
    persistence = ask("nvmlDeviceGetPersistenceMode")
    found["persistence_mode"] = None if persistence is None else ("Enabled" if persistence else "Disabled")
    found["compute_mode"] = _NVML_COMPUTE_MODES.get(ask("nvmlDeviceGetComputeMode"))
    mig = ask("nvmlDeviceGetMigMode")
    current = mig[0] if isinstance(mig, (tuple, list)) and mig else None
    found["mig_mode"] = None if current is None else ("Enabled" if current else "Disabled")

    for key, value in found.items():
        if value is not None:
            specs[key] = value
    for key, field in _SMI_SPEC_FIELDS.items():
        if specs[key] == "N/A":
            value = smi(index, field)
            if value is not None:
                specs[key] = value
    node = numa_node_of(bus_id, pci_root)
    if node is not None:
        specs["numa_node"] = node
    if started:
        try:
            lib.nvmlShutdown()
        except Exception:
            pass
    return specs


def _sysfs_text(path):
    """A sysfs file's text, or None: absent where the card has no such sensor, and
    EBUSY while it sleeps in a low-power state. Either way the sample is skipped."""
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _sysfs_number(path):
    text = _sysfs_text(path)
    try:
        return float(text.split()[0]) if text else None
    except (ValueError, IndexError):
        return None


def amd_device_dir(bus_id=None, rank=None, drm_root=SYSFS_DRM_ROOT, pci_root=SYSFS_PCI_ROOT):
    """The sysfs directory of one AMD GPU, or None.

    Found by PCI address when rocm-smi gave one. Otherwise by rank among the AMD
    cards in /sys/class/drm, which is how rocm-smi numbers them; a card number
    alone would be wrong on a host where another DRM device, a BMC's VGA for
    example, takes card0.
    """
    if bus_id:
        path = os.path.join(pci_root, str(bus_id).strip().lower())
        if os.path.isdir(path):
            return path
    if rank is None:
        return None
    cards = sorted((int(name[4:]), os.path.join(drm_root, name, "device"))
                   for name in (os.path.basename(p) for p in glob.glob(os.path.join(drm_root, "card*")))
                   if re.fullmatch(r"card\d+", name))
    amd = [path for _, path in cards if _sysfs_text(os.path.join(path, "vendor")) == "0x1002"]
    return amd[rank] if 0 <= rank < len(amd) else None


def amd_sysfs_readings(device_dir, have=()):
    """What an amdgpu device directory shows, as {series: value} using the monitor's
    series names. A sensor that is absent, unreadable or reads zero is left out of
    the dict: nothing was measured, so nothing is recorded.

    `have` names the series the caller already holds (from rocm-smi), and their files are
    not read: a hwmon sensor read goes to the card's firmware, so asking again for a reading
    rocm-smi has just supplied is a needless round trip every second. The PCIe link is always
    read; rocm-smi does not print it and it is plain PCI configuration.

    pcie_gen / pcie_width: the link as negotiated right now. It downshifts at idle,
    so only the samples taken under load carry the real link; the aggregate keeps
    the maximum. volts_core / volts_soc: hwmon rails labelled vddgfx / vddnb, in mV
    (vddnb exists on APUs only). fan_pct: pwm1 as a share of pwm1_max.
    """
    found = {}
    if not device_dir:
        return found
    gen = PCIE_GENERATIONS.get(_sysfs_number(os.path.join(device_dir, "current_link_speed")))
    if gen:
        found["pcie_gen"] = gen
    width = _sysfs_number(os.path.join(device_dir, "current_link_width"))
    if width and width > 0:
        found["pcie_width"] = int(width)
    for hwmon in sorted(glob.glob(os.path.join(device_dir, "hwmon", "hwmon*"))):
        if _sysfs_text(os.path.join(hwmon, "name")) not in (None, "amdgpu"):
            continue
        for label_file in sorted(glob.glob(os.path.join(hwmon, "in*_label"))):
            series = AMD_VOLTAGE_LABELS.get(_sysfs_text(label_file))
            if series and series not in found and series not in have:
                millivolts = _sysfs_number(label_file[:-len("_label")] + "_input")
                if millivolts and millivolts > 0:
                    found[series] = millivolts
        pwm = _sysfs_number(os.path.join(hwmon, "pwm1")) if "fan_pct" not in have else None
        if pwm is not None:
            found["fan_pct"] = round(100.0 * pwm / (_sysfs_number(os.path.join(hwmon, "pwm1_max")) or 255.0), 1)
        # freq2_input is the memory clock in hertz, labelled "mclk".
        for label_file in (sorted(glob.glob(os.path.join(hwmon, "freq*_label"))) if "clk_mem" not in have else ()):
            if _sysfs_text(label_file) == "mclk":
                hertz = _sysfs_number(label_file[:-len("_label")] + "_input")
                if hertz and hertz > 0:
                    found["clk_mem"] = round(hertz / 1e6, 1)
                break
        break
    return found


class HardwareMonitor:
    # psutil measures CPU load between consecutive calls. The first tick lands
    # a few milliseconds after the priming call, too short an interval to mean
    # anything, so host samples taken before this much time has passed are
    # recorded as missing rather than as a number.
    HOST_WARMUP_S = 0.5

    def __init__(self, platform):
        self.platform = platform
        self.sysfs_drm_root = SYSFS_DRM_ROOT
        self.sysfs_pci_root = SYSFS_PCI_ROOT
        self.has_rocm_smi = shutil.which("rocm-smi") is not None
        self.has_nvidia_smi = shutil.which("nvidia-smi") is not None
        self.running = False
        self.history = {} 
        self.thread = None
        self.warned = set()
        self.test_name = ""
        
        # Initialize NVML (NVIDIA Native) if available
        self.nvml_active = False
        if self.platform == "CUDA" and self.has_nvidia_smi and pynvml:
            try:
                pynvml.nvmlInit()
                self.nvml_active = True
            except Exception as e:
                print(f"[MONITOR] NVML Init failed: {e}. Falling back to nvidia-smi CLI.")
                self.nvml_active = False

    def _warn_once(self, key, msg):
        if key not in self.warned:
            print(msg)
            self.warned.add(key)

    @staticmethod
    def _parse_metric(value, default=0.0):
        try:
            text = str(value).strip()
            if text in ("", "N/A", "[Not Supported]"):
                return default
            for token in ("Mhz", "MHz", "W", "C", "%", "mV", "(", ")"):
                text = text.replace(token, "")
            return float(text.strip())
        except Exception:
            return default

    @staticmethod
    def _smi_number(value):
        """A number nvidia-smi printed, or None for a placeholder such as "[N/A]"."""
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _smi_mask(value):
        """The clock-event bitmask nvidia-smi printed, or None if it printed none."""
        text = str(value).strip()
        try:
            return int(text, 16) if text.lower().startswith("0x") else int(text)
        except ValueError:
            return None

    def get_gpu_count(self):
        if self.platform == "MOCK":
            return 1
        if self.platform == "HIP" and self.has_rocm_smi:
            try:
                out = subprocess.check_output(["rocm-smi", "-i", "--json"]).decode()
                return len(json.loads(out))
            except Exception as e:
                self._warn_once("rocm_count", f"[MONITOR] Unable to count AMD GPUs via rocm-smi: {e}. Falling back to 1.")
                return 1
        elif self.platform == "CUDA" and self.has_nvidia_smi:
            if self.nvml_active:
                try:
                    return pynvml.nvmlDeviceGetCount()
                except Exception as e:
                    self._warn_once("nvml_count", f"[MONITOR] Unable to count NVIDIA GPUs via NVML: {e}. Falling back to 1.")
                    return 1
            else:
                try:
                    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"]).decode()
                    return len([line for line in out.splitlines() if line.strip()])
                except Exception as e:
                    self._warn_once("nvidia_count", f"[MONITOR] Unable to count NVIDIA GPUs via nvidia-smi: {e}. Falling back to 1.")
                    return 1
        return 1

    def start_collection(self, gpu_ids, output_dir=".", test_name=""):
        self.running = True
        self.test_name = test_name
        # Reset history for this run
        self.history = {gid: {
            'temp_core': [], 'temp_mem': [], 
            'pwr': [], 'clk_core': [], 
            'fan_pct': [], 'volts_core': [], 'volts_soc': [],
            'pcie_gen': [], 'pcie_width': [], 'throttle': [],
            'gpu_util': [], 'mem_used': [], 'mem_total': [], 'elapsed': [],
            'clk_mem': [], 'temp_edge': [], 'temp_hotspot': []
        } for gid in gpu_ids}
        # The host is sampled on the same ticks as the GPUs. One set for the
        # machine, not one per GPU: every GPU in the run shares the CPU.
        self.host = {'cpu_util': [], 'cpu_busiest_core': [], 'cpu_freq': [], 'elapsed': []}
        self._host_primed_at = time.time()
        
        csv_path = os.path.join(output_dir, "time_series.csv")
        write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        self.csv_file = open(csv_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        if write_header:
            self.csv_writer.writerow([
                "Test_Name", "Timestamp", "GPU_ID",
                "Temp_Core(C)", "Temp_Mem(C)",
                "Power(W)", "Clock(MHz)",
                "Fan(%)", "Volts_Core(mV)", "Volts_SoC(mV)",
                "PCIe_Gen", "PCIe_Width", "Throttle_Reason",
                "GPU_Utilization(%)", "Memory_Used(MiB)", "Memory_Total(MiB)",
                "CPU_Utilization(%)", "CPU_Busiest_Core(%)", "CPU_Clock(MHz)",
                "Mem_Clock(MHz)", "Temp_Edge(C)", "Temp_Hotspot(C)"
            ])

        self._begin_context(gpu_ids)
        self.thread = threading.Thread(target=self._loop, args=(gpu_ids,))
        self.thread.start()

    def _prime_host(self):
        """Start psutil's CPU-load interval on the thread that will sample it.

        psutil answers cpu_percent() relative to the previous call, and since
        psutil 6 it keeps that previous call per thread. Priming anywhere else
        leaves this thread without a baseline, and its first reading is 0.0 --
        observed as a spurious idle second at the start of every workload.
        """
        if psutil is not None:
            try:
                psutil.cpu_percent(percpu=True)
            except Exception:
                pass
        self._host_primed_at = time.time()

    def _loop(self, gpu_ids):
        self._prime_host()
        start_t = time.time()
        # At least one tick even when stop_collection() lands before this
        # thread gets going, so a very short workload still has a sample.
        first_tick = True
        while self.running or first_tick:
            first_tick = False
            try:
                self._poll_platform(gpu_ids)
            except Exception as e:
                # Prevent thread death if a single poll fails
                print(f"[MONITOR DEBUG] Polling error: {e}")
            
            elapsed = round(time.time() - start_t, 2)
            if time.time() - self._host_primed_at >= self.HOST_WARMUP_S:
                cpu_util, cpu_core, cpu_freq = self._poll_host()
            else:
                cpu_util = cpu_core = cpu_freq = None
            self.host['cpu_util'].append(cpu_util)
            self.host['cpu_busiest_core'].append(cpu_core)
            self.host['cpu_freq'].append(cpu_freq)
            self.host['elapsed'].append(elapsed)
            for gid in gpu_ids:
                h = self.history[gid]
                
                # Safe retrieval with defaults
                # A series nothing was read into is written as N/A, not 0: the
                # time series must not claim a measurement that was never taken.
                def get_last(key, default="N/A"):
                    return h[key][-1] if h[key] else default

                t_c = get_last('temp_core')
                t_m = get_last('temp_mem')
                p   = get_last('pwr')
                c   = get_last('clk_core')
                f   = get_last('fan_pct')
                v_c = get_last('volts_core')
                v_s = get_last('volts_soc')
                pg  = get_last('pcie_gen')
                pw  = get_last('pcie_width')
                tr  = get_last('throttle', "N/A")
                gu  = get_last('gpu_util')
                mu  = get_last('mem_used')
                mt  = get_last('mem_total')
                mc  = get_last('clk_mem')
                te  = get_last('temp_edge')
                th  = get_last('temp_hotspot')
                h['elapsed'].append(elapsed)
                    
                self.csv_writer.writerow([self.test_name, elapsed, gid, t_c, t_m, p, c, f, v_c, v_s, pg, pw, tr, gu, mu, mt,
                                          *("N/A" if v is None else round(v, 1)
                                            for v in (cpu_util, cpu_core, cpu_freq)),
                                          mc, te, th])
            
            self.csv_file.flush()
            time.sleep(1)
        
        self.csv_file.close()

    def _poll_platform(self, gpu_ids):
        if self.platform == "HIP" and self.has_rocm_smi:
            self._poll_amd(gpu_ids)
        elif self.platform == "CUDA" and self.has_nvidia_smi:
            self._poll_nvidia(gpu_ids)

    def _capture_start_state(self, gpu_ids):
        """One reading of every card before the workload exists.

        The first tick of the sampling thread is not that reading: rocm-smi alone takes a second or two to answer,
        by which time the workload has launched, allocated its memory and started its kernels, and a "start" state
        that already shows 99% of VRAM in use is the run, not what it started from. So the reading is taken here,
        synchronously, into a scratch history that is thrown away, which also keeps the series the same length as
        the elapsed-time series.
        """
        self._start_state = {}
        live = self.history
        self.history = {gid: {key: [] for key in live[gid]} for gid in gpu_ids if gid in live}
        try:
            self._poll_platform(gpu_ids)
            self._start_state = {gid: {key: values[-1] for key, values in series.items() if values and key != 'elapsed'}
                                 for gid, series in self.history.items()}
        except Exception:
            self._start_state = {}
        finally:
            self.history = live

    @staticmethod
    def _poll_host():
        """Host CPU for one tick: machine-wide load, the busiest core, and clock.

        The busiest core is what exposes a host bottleneck. A benchmark that is
        waiting on one launch thread pins a single core while the machine-wide
        average stays low -- one saturated core of eight reads as 12.5%.
        Returns None for anything that could not be read.
        """
        if psutil is None:
            return None, None, None
        try:
            per_core = psutil.cpu_percent(percpu=True)
        except Exception:
            per_core = []
        util = sum(per_core) / len(per_core) if per_core else None
        busiest = max(per_core) if per_core else None
        try:
            freq = psutil.cpu_freq()
            freq = freq.current if freq and freq.current else None
        except Exception:
            freq = None
        return util, busiest, freq

    def _begin_context(self, gpu_ids):
        """What the host and the cards were doing just before the workload starts.

        Called before the workload is launched, so anything already using a GPU is somebody else's, and the
        cumulative throttle counters are read here and again at the end to give the seconds spent held back.
        """
        try:
            load = os.getloadavg()[0]
        except (OSError, AttributeError):
            load = None
        self.host_load_start = None if load is None else round(load, 2)
        self.other_procs = {}
        self._violation_start = {}
        self._violation_end = {}
        if self.platform == "CUDA" and self.nvml_active:
            for gid in gpu_ids:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gid)
                except Exception:
                    continue
                self.other_procs[gid] = self._count_gpu_processes(handle)
                self._violation_start[gid] = self._violation_times(handle)
        self._capture_start_state(gpu_ids)

    def _end_context(self):
        if self.platform == "CUDA" and self.nvml_active:
            for gid in getattr(self, "_violation_start", {}):
                try:
                    self._violation_end[gid] = self._violation_times(pynvml.nvmlDeviceGetHandleByIndex(gid))
                except Exception:
                    continue

    @staticmethod
    def _count_gpu_processes(handle):
        """How many processes hold a context on the card, or None when the driver would not say."""
        pids, answered = set(), False
        for name in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
            try:
                found = getattr(pynvml, name)(handle)
            except Exception:
                continue
            answered = True
            pids.update(getattr(process, "pid", None) for process in found)
        pids.discard(os.getpid())
        pids.discard(None)
        return len(pids) if answered else None

    @staticmethod
    def _violation_times(handle):
        """Cumulative seconds the card has been held back by power and by thermal limits, None where unsupported."""
        seen = {}
        for name, policy in (("power", NVML_PERF_POLICY_POWER), ("thermal", NVML_PERF_POLICY_THERMAL)):
            try:
                seen[name] = pynvml.nvmlDeviceGetViolationStatus(handle, policy).violationTime / 1e9
            except Exception:
                seen[name] = None
        return seen

    def stop_collection(self, measurement_window_seconds=None):
        self.running = False
        if self.thread: self.thread.join()
        self._end_context()
        return self._aggregate(measurement_window_seconds)

    # --- NVIDIA POLLING (Crash-Proof Version) ---
    def _poll_nvidia(self, gpu_ids):
        # The GPU index is the only CLI field that must parse; sensor readings
        # go through _smi_number, which returns None for an absent one.
        def parse_index(val):
            try: return int(val.strip())
            except Exception: return None

        if self.nvml_active:
            # NVML Library Mode (Best)
            try:
                for gid in gpu_ids:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gid)
                    h = self.history[gid]
                    
                    # A sensor this card does not have must not be recorded as a
                    # reading of zero. Appending 0 on failure made "no memory
                    # temperature sensor" indistinguishable from "VRAM at 0 C",
                    # and the latter is what got published. Skip the sample
                    # instead; the aggregate reports N/A when nothing was read.
                    try:
                        edge = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        h['temp_core'].append(edge)
                        h.setdefault('temp_edge', []).append(edge)
                    except Exception: pass

                    # A field value, not sensor index 2. nvmlDeviceGetTemperature
                    # accepts only NVML_TEMPERATURE_GPU (NVML_TEMPERATURE_COUNT is
                    # 1), so the old call raised on every card and memory
                    # temperature was never read on this path -- 0 of 5,630
                    # published v1.2.x NVIDIA rows carry one, H100 and GH200
                    # included. A card without the sensor returns NOT_SUPPORTED
                    # for the field and is skipped, as before.
                    try:
                        field_id = getattr(pynvml, "NVML_FI_DEV_MEMORY_TEMP", NVML_FI_DEV_MEMORY_TEMP)
                        (field,) = pynvml.nvmlDeviceGetFieldValues(handle, [field_id])
                        if field.nvmlReturn == pynvml.NVML_SUCCESS:
                            h['temp_mem'].append(float(_nvml_field_number(field)))
                    except Exception: pass

                    try: h['pwr'].append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
                    except Exception: pass

                    try: h['clk_core'].append(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS))
                    except Exception: pass

                    try: h.setdefault('clk_mem', []).append(pynvml.nvmlDeviceGetClockInfo(handle, NVML_CLOCK_MEM))
                    except Exception: pass

                    try: h['gpu_util'].append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                    except Exception: pass

                    try:
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        h['mem_used'].append(mem.used / (1024.0 * 1024.0))
                        h['mem_total'].append(mem.total / (1024.0 * 1024.0))
                    except Exception: pass

                    # Fan, PCIe link and voltage follow the same rule as every
                    # sensor above: an unreadable one is skipped, never stored as
                    # 0. Passively cooled datacenter cards have no fan, and a
                    # report of "0 % fan" or "PCIe x0" is a reading nobody took.
                    # Voltage is not exposed by NVML at all, so those series stay
                    # empty and the report says N/A.
                    try: h['fan_pct'].append(pynvml.nvmlDeviceGetFanSpeed(handle))
                    except Exception: pass

                    try: h['pcie_gen'].append(pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle))
                    except Exception: pass

                    try: h['pcie_width'].append(pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle))
                    except Exception: pass

                    # Throttle Reason
                    try:
                        h['throttle'].append(decode_throttle_mask(
                            pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)))
                    except Exception: h['throttle'].append("N/A")
            except Exception as e:
                self._warn_once("nvml_poll", f"[MONITOR] NVML polling failed: {e}.")
        else:
            # CLI Fallback Mode (Robust)
            try:
                # Query: Index, Core Temp, Mem Temp, Power, Clock, Fan, Gen, Width, Throttle, Util, Memory Used, Memory Total
                cmd = [
                    "nvidia-smi",
                    "--query-gpu=index,temperature.gpu,temperature.memory,power.draw,clocks.gr,fan.speed,pcie.link.gen.current,pcie.link.width.current,clocks_throttle_reasons.active,utilization.gpu,memory.used,memory.total,clocks.mem",
                    "--format=csv,noheader,nounits"
                ]
                out = subprocess.check_output(cmd).decode()
                
                for line in out.strip().split('\n'):
                    parts = line.split(',')
                    if len(parts) < 12: continue # Skip malformed lines

                    idx = parse_index(parts[0])
                    if idx in self.history:
                        h = self.history[idx]
                        
                        # Absent sensors are skipped rather than recorded as 0, the
                        # rule the NVML path already follows. nvidia-smi prints
                        # "[N/A]", which safe_parse turned into 0 C and 0 W.
                        # The same goes for fan, PCIe link and memory. Voltage is
                        # not queried, so its series stay empty (N/A in the report).
                        columns = [('temp_core', parts[1]), ('temp_edge', parts[1]), ('temp_mem', parts[2]),
                                   ('pwr', parts[3]), ('clk_core', parts[4]),
                                   ('fan_pct', parts[5]), ('pcie_gen', parts[6]),
                                   ('pcie_width', parts[7]), ('gpu_util', parts[9]),
                                   ('mem_used', parts[10]), ('mem_total', parts[11])]
                        if len(parts) > 12:
                            columns.append(('clk_mem', parts[12]))
                        for key, raw in columns:
                            reading = self._smi_number(raw)
                            if reading is not None:
                                h.setdefault(key, []).append(reading)

                        h['throttle'].append(decode_throttle_mask(self._smi_mask(parts[8])))

            except Exception as e:
                self._warn_once("nvidia_poll", f"[MONITOR] nvidia-smi polling failed: {e}.")


    # Without --showuse and --showmeminfo rocm-smi never prints utilization or
    # memory use, so those columns used to be 0.0 on every AMD row.
    _ROCM_BASE = ["rocm-smi", "-v", "-P", "-t", "-c", "-f"]
    # --showvoltage ("Voltage (mV)", the graphics rail) and --showbus ("PCI Bus",
    # which locates the card's sysfs directory) ride along with the usage flags.
    _ROCM_USAGE = ["--showuse", "--showmeminfo", "vram", "--showvoltage", "--showbus"]

    def _rocm_json(self):
        """rocm-smi's JSON for the sensors we read; the extra usage flags are
        dropped if this rocm-smi release rejects them, so a missing option costs
        only utilization and memory, never the temperature and power readings."""
        try:
            return json.loads(subprocess.check_output(
                self._ROCM_BASE + self._ROCM_USAGE + ["--json"]).decode())
        except Exception:
            return json.loads(subprocess.check_output(self._ROCM_BASE + ["--json"]).decode())

    def _poll_amd(self, gpu_ids):
        # Every reading below is appended only when rocm-smi printed a number. A
        # sensor the card lacks (or an option this rocm-smi does not know) leaves
        # its series empty, which the aggregate reports as N/A. It used to append
        # 0, which published "0.0 C" memory temperatures and zero memory size.
        parse = lambda v: self._parse_metric(v, default=None)
        try:
            data = self._rocm_json()

            for gid in gpu_ids:
                card_key = f"card{gid}"
                if card_key not in data:
                    keys = list(data.keys())
                    if len(keys) > gid: card_key = keys[gid]

                if card_key in data:
                    c = data[card_key]
                    h = self.history[gid]
                    for key in ('gpu_util', 'mem_used', 'mem_total', 'elapsed'):
                        h.setdefault(key, [])

                    t_junction = t_edge = t_memory = None
                    for k, v in c.items():
                        kl = k.lower()
                        if "temperature" in kl:
                            val = parse(v)
                            if val is None:
                                continue
                            # New drivers use "junction" as the primary hotspot
                            if "junction" in kl:
                                t_junction = val
                            elif "edge" in kl:
                                t_edge = val
                            # Capture dedicated memory sensor
                            if "memory" in kl:
                                t_memory = val

                    t_core = t_junction if t_junction is not None else t_edge
                    if t_core is not None:
                        h['temp_core'].append(t_core)
                    # Both sensors on their own, because "Peak Temp" is the hotspot on AMD and the edge on NVIDIA.
                    if t_edge is not None:
                        h.setdefault('temp_edge', []).append(t_edge)
                    if t_junction is not None:
                        h.setdefault('temp_hotspot', []).append(t_junction)
                    if t_memory is not None:
                        h['temp_mem'].append(t_memory)

                    # --- 2. POWER (Fuzzy Search) ---
                    p_val = None
                    for k, v in c.items():
                        kl = k.lower()
                        # Matches "Average Graphics Package Power" OR "Current Socket Graphics Package Power"
                        if "power" in kl and ("average" in kl or "socket" in kl):
                            val = parse(v)
                            if val is not None:
                                p_val = val
                    if p_val is not None:
                        h['pwr'].append(p_val)

                    # --- 3. CLOCK (Standard Search) ---
                    clk_val = None
                    for k, v in c.items():
                        if "sclk" in k.lower() and "(" in str(v):
                            # Extracts 800 from (800Mhz)
                            val = parse(v)
                            if val is not None:
                                clk_val = val
                    if clk_val is not None:
                        h['clk_core'].append(clk_val)

                    # --- 4. UTILIZATION AND VRAM (--showuse, --showmeminfo vram) ---
                    # "GPU use (%)", "VRAM Total Memory (B)", "VRAM Total Used
                    # Memory (B)". The memory figures are bytes; the report is MiB.
                    util_val = mem_used = mem_total = None
                    for k, v in c.items():
                        kl = k.lower()
                        val = parse(v)
                        if val is None:
                            continue
                        if kl.startswith("gpu use"):
                            util_val = val
                        elif "vram" in kl and "used" in kl:
                            mem_used = val / (1024.0 * 1024.0)
                        elif "vram" in kl and "total" in kl:
                            mem_total = val / (1024.0 * 1024.0)
                    if util_val is not None:
                        h['gpu_util'].append(util_val)
                    if mem_used is not None:
                        h['mem_used'].append(mem_used)
                    if mem_total is not None:
                        h['mem_total'].append(mem_total)

                    # --- 5. FAN, VOLTAGE (rocm-smi first) ---
                    # The percent sign sits in the value ("45%") on some releases and
                    # in the key ("Fan speed (%)": "45") on others.
                    readings = {}
                    for k, v in c.items():
                        kl = k.lower()
                        if "fan" in kl and ("%" in kl or "%" in str(v)):
                            match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", str(v))
                            val = float(match.group(1)) if match else parse(v)
                            if val is not None:
                                readings['fan_pct'] = val
                        elif kl.startswith("voltage"):
                            val = parse(v)
                            if val is not None and val > 0:
                                readings['volts_core'] = val
                        elif "mclk" in kl and "(" in str(v):
                            # "mclk clock speed:": "(1000Mhz)"; the "clock level" key beside it has no parentheses.
                            val = parse(v)
                            if val is not None and val > 0:
                                readings['clk_mem'] = val

                    # --- 6. PCIE LINK, VOLTAGE RAILS, FAN (amdgpu sysfs for what rocm-smi omitted) ---
                    device_dir = amd_device_dir(str(c.get("PCI Bus", "")).strip() or None, rank=gid,
                                                drm_root=self.sysfs_drm_root, pci_root=self.sysfs_pci_root)
                    for series, val in amd_sysfs_readings(device_dir, have=readings).items():
                        readings.setdefault(series, val)
                    for series, val in readings.items():
                        h.setdefault(series, []).append(val)

        except Exception as e:
            self._warn_once("amd_poll", f"[MONITOR] rocm-smi polling failed: {e}.")

    def _host_stats(self, measurement_window_seconds=None):
        host = getattr(self, "host", None) or {}
        elapsed = list(host.get('elapsed', []))
        keep = [True] * len(elapsed)
        if measurement_window_seconds is not None and len(elapsed) > 1 and measurement_window_seconds > 0:
            cutoff = elapsed[-1] - float(measurement_window_seconds)
            keep = [t >= cutoff for t in elapsed]

        def series(key):
            return [v for v, k in zip(host.get(key, []), keep) if k and v is not None]

        def mean(values): return float(round(np.mean(values), 1)) if values else "N/A"
        def peak(values): return float(round(np.max(values), 1)) if values else "N/A"
        util, busiest, freq = series('cpu_util'), series('cpu_busiest_core'), series('cpu_freq')
        return {
            "avg_cpu_util": mean(util),
            "max_cpu_util": peak(util),
            "avg_cpu_busiest_core": mean(busiest),
            "max_cpu_busiest_core": peak(busiest),
            "avg_cpu_clk": mean(freq),
            "host_load_start": ("N/A" if getattr(self, "host_load_start", None) is None
                                else float(self.host_load_start)),
        }

    def _aggregate(self, measurement_window_seconds=None):
        stats = {}
        host_stats = self._host_stats(measurement_window_seconds)
        for gid, data in self.history.items():
            samples = data
            if measurement_window_seconds is not None:
                elapsed_values = np.asarray(data.get('elapsed', []), dtype=float)
                if len(elapsed_values) > 1 and measurement_window_seconds > 0:
                    # Keep the complete time series, but use only the final
                    # workload-duration window for profiled telemetry. This
                    # excludes profiler setup, replay, and teardown time.
                    cutoff = elapsed_values[-1] - float(measurement_window_seconds)
                    mask = elapsed_values >= cutoff
                    samples = {
                        key: [value for value, keep in zip(values, mask) if keep]
                        for key, values in data.items()
                    }
            # "N/A" rather than 0 when a series is empty: nothing was read, and
            # reporting a number invents a measurement that was never taken.
            # Consumers already treat "N/A" as missing.
            def safe_max(l): return float(round(np.max(l), 1)) if l else "N/A"
            def safe_mean(l): return float(round(np.mean(l), 1)) if l else "N/A"
            def safe_min(l): return float(round(np.min(l), 1)) if l else "N/A"
            def mode_str(l): return max(set(l), key=l.count) if l else "N/A"
            power = np.asarray(samples.get('pwr', []), dtype=float)
            elapsed = np.asarray(samples.get('elapsed', []), dtype=float)
            if len(elapsed) != len(power):
                elapsed = np.arange(len(power), dtype=float)
            integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
            energy_wh = float(round(integrate(power, elapsed) / 3600.0, 3)) if len(power) > 1 else 0
            throttle_samples = sum(1 for reason in samples.get('throttle', [])
                                   if reason not in ("N/A", "None", "Idle", "") and reason is not None)
            sample_interval = (elapsed[-1] - elapsed[0]) / max(1, len(elapsed) - 1) if len(elapsed) > 1 else 1

            # Seconds held back by each cause, from the same samples as "Throttle Time". N/A when the series never
            # said anything at all (no throttle reading on this card), so a card that cannot tell is not "clean".
            throttle_series = [reason for reason in samples.get('throttle', []) if isinstance(reason, str)]
            throttle_read = any(reason not in ("N/A", "") for reason in throttle_series)

            def held_back(label):
                if not throttle_read:
                    return "N/A"
                count = sum(1 for reason in throttle_series if label in reason.split("|"))
                return float(round(count * sample_interval, 1))

            def violation(name):
                begin = getattr(self, "_violation_start", {}).get(gid, {}).get(name)
                end = getattr(self, "_violation_end", {}).get(gid, {}).get(name)
                if begin is None or end is None:
                    return "N/A"
                return float(round(max(0.0, end - begin), 2))

            captured = getattr(self, "_start_state", {}).get(gid)

            def first(key):
                # The reading taken before launch when there is one (a key it could not read is N/A, not the first
                # in-run sample); a history built by hand has no such reading and uses its first sample.
                if captured is not None:
                    value = captured.get(key)
                else:
                    values = data.get(key, [])
                    value = values[0] if values else None
                return "N/A" if value is None else float(round(value, 1))

            # How far the hottest reading came from the card's own slowdown temperature.
            limit = (getattr(self, "temp_limits", None) or {}).get(gid)
            hottest = samples.get('temp_core', [])
            margin = ("N/A" if not isinstance(limit, (int, float)) or isinstance(limit, bool) or not hottest
                      else float(round(limit - max(hottest), 1)))
            others = getattr(self, "other_procs", {}).get(gid)

            stats[gid] = {
                "avg_temp": safe_mean(samples['temp_core']),
                "max_temp": safe_max(samples['temp_core']),
                "avg_mem_temp": safe_mean(samples['temp_mem']),
                "max_mem_temp": safe_max(samples['temp_mem']),
                "avg_pwr":  safe_mean(samples['pwr']),
                "max_pwr":  safe_max(samples['pwr']),
                "avg_clk":  safe_mean(samples['clk_core']),
                "min_clk": safe_min([v for v in samples['clk_core'] if v > 0]),
                "max_clk": safe_max(samples['clk_core']),
                "avg_gpu_util": safe_mean(samples.get('gpu_util', [])),
                "max_gpu_util": safe_max(samples.get('gpu_util', [])),
                "peak_mem_used": safe_max(samples.get('mem_used', [])),
                "mem_total": safe_max(samples.get('mem_total', [])),
                "energy_wh": energy_wh,
                "thermal_rise": (float(round((max(samples['temp_core']) - samples['temp_core'][0]), 1))
                                 if len(samples['temp_core']) > 1 else (0 if samples['temp_core'] else "N/A")),
                "throttle_time": float(round(throttle_samples * sample_interval, 1)),
                "max_fan": safe_max(samples['fan_pct']),
                "max_volts_core": safe_max(samples['volts_core']),
                "max_volts_soc": safe_max(samples['volts_soc']),
                "pcie_gen": safe_max(samples['pcie_gen']),
                "pcie_width": safe_max(samples['pcie_width']),
                "throttle_reason": mode_str(samples['throttle']),
                "avg_mem_clk": safe_mean(samples.get('clk_mem', [])),
                "max_mem_clk": safe_max(samples.get('clk_mem', [])),
                "max_edge_temp": safe_max(samples.get('temp_edge', [])),
                "max_hotspot_temp": safe_max(samples.get('temp_hotspot', [])),
                "thermal_margin": margin,
                "throttle_power_s": held_back("Power"),
                "throttle_thermal_s": held_back("Thermal"),
                "throttle_brake_s": held_back("Power Brake"),
                "power_violation_s": violation("power"),
                "thermal_violation_s": violation("thermal"),
                "start_temp": first('temp_core'),
                "start_pwr": first('pwr'),
                "start_gpu_util": first('gpu_util'),
                "start_mem_used": first('mem_used'),
                "start_clk": first('clk_core'),
                "start_mem_clk": first('clk_mem'),
                "other_gpu_processes": "N/A" if others is None else int(others),
                **host_stats,
            }
        return stats
