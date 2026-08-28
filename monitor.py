import subprocess
import time
import json
import shutil
import threading
import numpy as np
import csv
import os

# Try to import NVIDIA native bindings
try:
    import pynvml
except ImportError:
    pynvml = None

class HardwareMonitor:
    def __init__(self, platform):
        self.platform = platform
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
            'gpu_util': [], 'mem_used': [], 'mem_total': [], 'elapsed': []
        } for gid in gpu_ids}
        
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
                "GPU_Utilization(%)", "Memory_Used(MiB)", "Memory_Total(MiB)"
            ])
        
        self.thread = threading.Thread(target=self._loop, args=(gpu_ids,))
        self.thread.start()

    def _loop(self, gpu_ids):
        start_t = time.time()
        while self.running:
            try:
                if self.platform == "MOCK":
                    pass
                elif self.platform == "HIP" and self.has_rocm_smi:
                    self._poll_amd(gpu_ids)
                elif self.platform == "CUDA" and self.has_nvidia_smi:
                    self._poll_nvidia(gpu_ids)
            except Exception as e:
                # Prevent thread death if a single poll fails
                print(f"[MONITOR DEBUG] Polling error: {e}")
            
            elapsed = round(time.time() - start_t, 2)
            for gid in gpu_ids:
                h = self.history[gid]
                
                # Safe retrieval with defaults
                def get_last(key, default=0):
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
                h['elapsed'].append(elapsed)
                    
                self.csv_writer.writerow([self.test_name, elapsed, gid, t_c, t_m, p, c, f, v_c, v_s, pg, pw, tr, gu, mu, mt])
            
            self.csv_file.flush()
            time.sleep(1)
        
        self.csv_file.close()

    def stop_collection(self, measurement_window_seconds=None):
        self.running = False
        if self.thread: self.thread.join()
        return self._aggregate(measurement_window_seconds)

    # --- NVIDIA POLLING (Crash-Proof Version) ---
    def _poll_nvidia(self, gpu_ids):
        # Helper to safely parse floats/ints from "N/A" strings
        def safe_parse(val, type_func):
            try:
                val = val.strip()
                if val == "N/A" or val == "[Not Supported]": return 0
                if type_func == int and val.startswith("0x"): return int(val, 16)
                return type_func(val)
            except: return 0

        if self.nvml_active:
            # NVML Library Mode (Best)
            try:
                for gid in gpu_ids:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gid)
                    h = self.history[gid]
                    
                    try: h['temp_core'].append(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                    except: h['temp_core'].append(0)
                    
                    try: h['temp_mem'].append(pynvml.nvmlDeviceGetTemperature(handle, 2)) # 2 = Memory
                    except: h['temp_mem'].append(0)

                    try: h['pwr'].append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
                    except: h['pwr'].append(0)
                    
                    try: h['clk_core'].append(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS))
                    except: h['clk_core'].append(0)

                    try: h['gpu_util'].append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                    except: h['gpu_util'].append(0)

                    try:
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        h['mem_used'].append(mem.used / (1024.0 * 1024.0))
                        h['mem_total'].append(mem.total / (1024.0 * 1024.0))
                    except:
                        h['mem_used'].append(0)
                        h['mem_total'].append(0)

                    try: h['fan_pct'].append(pynvml.nvmlDeviceGetFanSpeed(handle))
                    except: h['fan_pct'].append(0)

                    # Voltage (Rarely supported on Linux Consumer)
                    h['volts_core'].append(0)
                    h['volts_soc'].append(0)

                    try: h['pcie_gen'].append(pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle))
                    except: h['pcie_gen'].append(0)

                    try: h['pcie_width'].append(pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle))
                    except: h['pcie_width'].append(0)

                    # Throttle Reason
                    try:
                        reasons = []
                        mask = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                        if mask & pynvml.nvmlClocksThrottleReasonGpuIdle: reasons.append("Idle")
                        if mask & pynvml.nvmlClocksThrottleReasonSwPowerCap: reasons.append("Power")
                        if mask & pynvml.nvmlClocksThrottleReasonHwSlowdown: reasons.append("Thermal")
                        if not reasons: reasons.append("None")
                        h['throttle'].append("|".join(reasons))
                    except: h['throttle'].append("N/A")
            except Exception as e:
                self._warn_once("nvml_poll", f"[MONITOR] NVML polling failed: {e}.")
        else:
            # CLI Fallback Mode (Robust)
            try:
                # Query: Index, Core Temp, Mem Temp, Power, Clock, Fan, Gen, Width, Throttle, Util, Memory Used, Memory Total
                cmd = [
                    "nvidia-smi",
                    "--query-gpu=index,temperature.gpu,temperature.memory,power.draw,clocks.gr,fan.speed,pcie.link.gen.current,pcie.link.width.current,clocks_throttle_reasons.active,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits"
                ]
                out = subprocess.check_output(cmd).decode()
                
                for line in out.strip().split('\n'):
                    parts = line.split(',')
                    if len(parts) < 12: continue # Skip malformed lines

                    idx = safe_parse(parts[0], int)
                    if idx in self.history:
                        h = self.history[idx]
                        
                        h['temp_core'].append(safe_parse(parts[1], float))
                        h['temp_mem'].append(safe_parse(parts[2], float))
                        h['pwr'].append(safe_parse(parts[3], float))
                        h['clk_core'].append(safe_parse(parts[4], float))
                        h['fan_pct'].append(safe_parse(parts[5], float))
                        h['volts_core'].append(0)
                        h['volts_soc'].append(0)
                        h['pcie_gen'].append(safe_parse(parts[6], int))
                        h['pcie_width'].append(safe_parse(parts[7], int))
                        
                        # Throttle (Bitmask from CLI)
                        mask = safe_parse(parts[8], int)
                        reasons = []
                        if mask & 0x1: reasons.append("Idle")
                        if mask & 0x2: reasons.append("Power")
                        if mask & 0x8: reasons.append("Thermal")
                        h['throttle'].append("|".join(reasons) if reasons else "None")
                        h['gpu_util'].append(safe_parse(parts[9], float))
                        h['mem_used'].append(safe_parse(parts[10], float))
                        h['mem_total'].append(safe_parse(parts[11], float))

            except Exception as e:
                self._warn_once("nvidia_poll", f"[MONITOR] nvidia-smi polling failed: {e}.")


    def _poll_amd(self, gpu_ids):
        try:
            out = subprocess.check_output(["rocm-smi", "-v", "-P", "-t", "-c", "-f", "--json"]).decode()
            data = json.loads(out)

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

                    t_junction = 0
                    t_edge = 0
                    t_memory = 0

                    for k, v in c.items():
                        kl = k.lower()
                        if "temperature" in kl:
                            try:
                                val = self._parse_metric(v)
                                # New drivers use "junction" as the primary hotspot
                                if "junction" in kl:
                                    t_junction = val
                                elif "edge" in kl:
                                    t_edge = val
                                # Capture dedicated memory sensor
                                if "memory" in kl:
                                    t_memory = val
                            except: pass

                    h['temp_core'].append(t_junction or t_edge)
                    h['temp_mem'].append(t_memory)

                    # --- 2. POWER (Fuzzy Search) ---
                    p_val = 0
                    for k, v in c.items():
                        kl = k.lower()
                        # Matches "Average Graphics Package Power" OR "Current Socket Graphics Package Power"
                        if "power" in kl and ("average" in kl or "socket" in kl):
                            try: p_val = self._parse_metric(v)
                            except: pass
                    h['pwr'].append(p_val)

                    # --- 3. CLOCK (Standard Search) ---
                    clk_val = 0
                    for k, v in c.items():
                        if "sclk" in k.lower() and "(" in str(v):
                            try:
                                # Extracts 800 from (800Mhz)
                                clk_val = self._parse_metric(v)
                            except: pass
                    h['clk_core'].append(clk_val)

                    util_val = 0
                    mem_used = 0
                    mem_total = 0
                    for k, v in c.items():
                        kl = k.lower()
                        try:
                            if "utilization" in kl or "gpu use" in kl:
                                util_val = self._parse_metric(v)
                            elif "memory" in kl and "use" in kl:
                                mem_used = self._parse_metric(v)
                            elif "memory" in kl and ("total" in kl or "capacity" in kl):
                                mem_total = self._parse_metric(v)
                        except: pass
                    h['gpu_util'].append(util_val)
                    h['mem_used'].append(mem_used)
                    h['mem_total'].append(mem_total)

                    # --- 4. FAN ---
                    f_pct = 0
                    for k, v in c.items():
                        if "fan" in k.lower() and "%" in str(v):
                            try: f_pct = self._parse_metric(v)
                            except: pass
                    h['fan_pct'].append(f_pct)

        except Exception as e:
            self._warn_once("amd_poll", f"[MONITOR] rocm-smi polling failed: {e}.")

    def _aggregate(self, measurement_window_seconds=None):
        stats = {}
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
            def safe_max(l): return float(round(np.max(l), 1)) if l else 0
            def safe_mean(l): return float(round(np.mean(l), 1)) if l else 0
            def safe_min(l): return float(round(np.min(l), 1)) if l else 0
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
                "thermal_rise": float(round((max(samples['temp_core']) - samples['temp_core'][0]), 1)) if len(samples['temp_core']) > 1 else 0,
                "throttle_time": float(round(throttle_samples * sample_interval, 1)),
                "max_fan": safe_max(samples['fan_pct']),
                "max_volts_core": safe_max(samples['volts_core']),
                "max_volts_soc": safe_max(samples['volts_soc']),
                "pcie_gen": safe_max(samples['pcie_gen']),
                "pcie_width": safe_max(samples['pcie_width']),
                "throttle_reason": mode_str(samples['throttle'])
            }
        return stats
