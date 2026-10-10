#!/usr/bin/env python3
"""Run Pantheon on the GPUs of a node and answer with one line and an exit code.

A scheduler, a health-check daemon or a cron job wants a yes or a no, not a
table. This script runs the workloads it is given, reads the reports Pantheon
writes, and prints a verdict for the node and for each GPU.

Exit codes follow the convention of Nagios plugins:

    0  HEALTHY        every workload completed and nothing was found
       SKIPPED        the job has no GPU
    1  WATCH          the card works, and something deserves a look
    2  FAULT          wrong data, uncorrectable errors, or a failed memory test
    3  no result      nothing was tested, or the run did not finish: a workload
                      that hung until the timeout, that exited without writing
                      a result, or that skipped itself leaves the verdict
                      INCOMPLETE, never HEALTHY (the Slurm epilog logs it and
                      drains only when told to, see integrations/slurm)

The verdict is read from the reports and not from Pantheon's exit code, which
covers the workloads only: errors that a card counted during a run that
completed are in the report.

The script needs Python 3.9 or later and nothing outside the standard library.
"""
import argparse
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

# The rules below mirror assess_gpu in pantheon.py, so a node judged here and a
# report read by hand reach the same word. They are repeated because a Pantheon
# release older than the verdict does not write it into the report.
DIAGNOSTIC_TESTS = {
    "march_test", "galpat", "memory_hammer", "memory_retention",
    "memory_retention_bake", "ras_validator",
}
THERMAL_WATCH_C = 90
MEMORY_THERMAL_WATCH_C = 95
# The PCIe link cycling power states ticks this counter on healthy hardware.
BENIGN_RAS_TOKENS = ("l0_to_recovery",)

HEALTHY, WATCH, FAULT = "HEALTHY", "WATCH", "FAULT"
INCOMPLETE, NO_GPU, NOT_RUN, SKIPPED = "INCOMPLETE", "NO GPU TESTED", "NOT RUN", "SKIPPED"
SEVERITY = {HEALTHY: 0, WATCH: 1, INCOMPLETE: 2, FAULT: 3, NO_GPU: 4, NOT_RUN: 5}
EXIT_CODE = {SKIPPED: 0, HEALTHY: 0, WATCH: 1, FAULT: 2, INCOMPLETE: 3, NO_GPU: 3, NOT_RUN: 3}

DEFAULT_WORKLOADS = ("memory_read", "march_test")
# These renumber or hide GPUs. The check names GPUs as the node numbers them,
# so Pantheon runs with them unset.
DEVICE_VARIABLES = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
# The first run on a node compiles the workloads before it starts the clock.
COMPILE_ALLOWANCE_S = 300


class NoDevices(Exception):
    """The job has no GPU, so there is nothing to test."""


class BadDevices(Exception):
    """The GPUs of the job cannot be read from the environment."""


def _num(value, default=0.0):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default


def split_ras_details(delta_text):
    benign, serious = [], []
    for token in str(delta_text or "").split("||"):
        token = token.strip()
        if not token or token in ("None", "N/A"):
            continue
        (benign if any(b in token for b in BENIGN_RAS_TOKENS) else serious).append(token)
    return benign, serious


def limit_reason_set(value):
    """The lower-case labels of a Limit Reason such as "Power|Thermal"."""
    return {part.strip().lower() for part in str(value or "").split("|") if part.strip()}


def is_skipped_row(row):
    """True for the row of a workload that declined to run, which is not a pass."""
    return str(row.get("Status") or "").upper() == "SKIP" or row.get("Unit") == "SKIP"


def assess_gpu(rows, gpu_id, gpu_name, unfinished=()):
    """Turn the result rows of one GPU into a verdict with its reasons.

    `unfinished` lists the workloads that were asked for and did not finish or
    left no result. Whatever else the card showed, the node was not fully
    tested, so the verdict is at best INCOMPLETE.
    """
    mine = [r for r in rows if r.get("GPU ID") == gpu_id]
    faults, watches, notes, link_recovery = [], [], [], []
    ran = 0
    skipped = []
    for row in mine:
        test = row.get("Test Name", "?")
        if row.get("Failure Stage"):
            notes.append(f"{test} did not run ({row.get('Failure Stage')}: {row.get('Failure Reason')})")
            continue
        failed = row.get("Unit") == "ERR" or str(row.get("Status") or "PASS").upper() == "FAIL"
        if failed:
            if test in DIAGNOSTIC_TESTS:
                faults.append(f"{test} failed: memory errors detected or the workload aborted")
            else:
                watches.append(f"{test} did not complete")
            continue
        if is_skipped_row(row):
            skipped.append(test)
            reason = row.get("Skip Reason")
            notes.append(f"{test} was skipped" + (f" ({reason})" if reason else ""))
            continue
        if test != "baseline_metrics":
            ran += 1

        ras_status = str(row.get("RAS Status", "")).upper()
        if ras_status == "ERROR":
            faults.append(f"{test}: uncorrectable errors ({row.get('RAS Error Delta')})")
        elif ras_status == "WARNING":
            benign, serious = split_ras_details(row.get("RAS Error Delta"))
            if serious:
                watches.append(f"{test}: correctable errors ({', '.join(serious)})")
            if benign:
                link_recovery.append(test)

        tmax = _num(row.get("Max Temp (C)"))
        tmem = _num(row.get("Max Mem Temp (C)"))
        # The reason is the commonest label of the samples, joined with "|"
        # when a sample had several ("Power|Thermal"), so it is read as a set.
        reasons = limit_reason_set(row.get("Limit Reason"))
        throttle_s = _num(row.get("Throttle Time (s)"))
        if "thermal" in reasons:
            seconds = f", {throttle_s:.0f} s throttled" if throttle_s > 0 else ""
            watches.append(f"{test}: thermally throttled, GPU at {tmax:.0f} C{seconds}")
        elif tmax >= THERMAL_WATCH_C:
            watches.append(f"{test}: GPU reached {tmax:.0f} C")
        if throttle_s > 0 and not (reasons - {"none", "n/a", "idle"}):
            notes.append(f"{test}: throttled for {throttle_s:.0f} s, cause not recorded")
        if tmem >= MEMORY_THERMAL_WATCH_C:
            watches.append(f"{test}: memory reached {tmem:.0f} C")

    if link_recovery:
        notes.append(f"PCIe link recovery during {', '.join(link_recovery)}: "
                     "link power-state cycling, not a fault")
    if faults:
        verdict = FAULT
    elif watches:
        verdict = WATCH
    elif ran:
        verdict = HEALTHY
    else:
        verdict = INCOMPLETE
        if skipped:
            notes.append(f"nothing ran: {len(skipped)} workload(s) skipped ({', '.join(skipped)})")
        else:
            notes.append("nothing ran: no workload completed")
    unfinished = list(unfinished)
    if unfinished:
        if verdict in (HEALTHY, WATCH):
            verdict = INCOMPLETE
        watches = watches + unfinished
    scores = [f"{r.get('Test Name')} {r.get('Score')} {r.get('Unit')}" for r in mine
              if r.get("Unit") not in (None, "ERR") and not r.get("Failure Stage")]
    return {"gpu_id": gpu_id, "gpu_name": gpu_name, "verdict": verdict,
            "reasons": faults + watches, "notes": notes, "scores": scores}


def _maybe_num(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def row_result(row):
    """The numbers of one result row, for programs and metrics."""
    return {
        "gpu_id": row.get("GPU ID"),
        "workload": row.get("Test Name"),
        "score": None if row.get("Unit") in ("ERR", "SKIP") else _maybe_num(row.get("Score")),
        "unit": row.get("Unit"),
        "max_temp_c": _maybe_num(row.get("Max Temp (C)")),
        "max_power_w": _maybe_num(row.get("Max Power (W)")),
        "limit_reason": row.get("Limit Reason"),
        "ras_status": row.get("RAS Status"),
        "failed": bool(row.get("Failure Stage")) or row.get("Unit") == "ERR"
        or str(row.get("Status") or "PASS").upper() == "FAIL",
    }


def report_files(report_dir):
    """The report files of a directory, as {path: (modification time, size)}."""
    found = {}
    for path in sorted(glob.glob(os.path.join(report_dir, "*.json"))):
        try:
            info = os.stat(path)
        except OSError:
            continue
        found[path] = (info.st_mtime_ns, info.st_size)
    return found


def load_reports(report_dir, previous=None, since_ns=None):
    """Rows and GPUs from the reports in the directory, each row once.

    Pantheon writes a report for the session and a second one for each workload
    that completed, holding the same row, so a row is counted once by content.

    A reused --report-dir holds the reports of earlier runs, whose old FAULT
    rows would condemn a repaired card for good and whose old HEALTHY rows
    would hide that a new run wrote nothing. With `previous`, the listing the
    directory had before the run (see report_files), only files that are new
    or changed since are read; `since_ns` also accepts any file written after
    that moment.
    """
    rows, gpus, kinds, seen = [], {}, set(), set()
    for path, stamp in report_files(report_dir).items():
        if previous is not None and previous.get(path) == stamp and not (since_ns and stamp[0] >= since_ns):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                report = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(report, dict):
            continue
        for gpu in report.get("gpu_static_info") or []:
            if isinstance(gpu, dict):
                gpus.setdefault(gpu.get("id"), gpu.get("name") or "GPU")
                kinds.add(str(gpu.get("type") or "").upper())
        for row in report.get("test_results") or []:
            if not isinstance(row, dict):
                continue
            key = json.dumps(row, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                rows.append(row)
    return rows, gpus, kinds


def parse_ids(text, source):
    ids = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if not token.isdigit():
            raise BadDevices(f"cannot tell which GPUs to test from {source}='{text}', "
                             "name them with --gpu")
        if int(token) not in ids:
            ids.append(int(token))
    return ids


def select_devices(gpu, context, environ):
    """The GPU ids to test, or None for every GPU of the node.

    In a prolog or an epilog Slurm names the GPUs of the job in SLURM_JOB_GPUS,
    numbered as the node numbers them. Inside a job the GPUs are the ones in
    CUDA_VISIBLE_DEVICES, numbered as the job sees them.
    """
    if gpu not in (None, "", "auto"):
        return None if gpu == "all" else parse_ids(gpu, "--gpu")
    if context == "node":
        return None
    variable = "SLURM_JOB_GPUS" if context in ("prolog", "epilog") else "CUDA_VISIBLE_DEVICES"
    value = environ.get(variable)
    if value is None:
        if context == "job":
            return None
        raise NoDevices("the job has no GPU, nothing was tested")
    if value.strip() in ("", "NoDevFiles"):
        raise NoDevices("the job has no GPU, nothing was tested")
    ids = parse_ids(value, variable)
    if not ids:
        raise NoDevices("the job has no GPU, nothing was tested")
    return ids


def run_pantheon(executable, workload, ids, args, workdir, environ):
    """Run one workload. Returns the exit code, or None when it timed out."""
    command = [executable, "--test", workload, "--duration", str(args.duration),
               "--mem", str(args.mem), "--gpu", "all" if ids is None else ",".join(map(str, ids))]
    if args.platform != "auto":
        command += ["--platform", args.platform]
    env = {k: v for k, v in environ.items() if k not in DEVICE_VARIABLES}
    timeout = args.timeout if args.timeout else args.duration + COMPILE_ALLOWANCE_S
    with open(os.path.join(workdir, f"{workload}.log"), "w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=workdir, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Pantheon starts one process per GPU; stop all of them.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
            process.wait()
            return None


def log_tail(workdir, workload, lines=3):
    try:
        with open(os.path.join(workdir, f"{workload}.log"), encoding="utf-8", errors="replace") as log:
            text = [line.strip() for line in log if line.strip()]
    except OSError:
        return ""
    return " | ".join(text[-lines:])


def check(args, environ):
    """Run the workloads and return the result as a dictionary."""
    result = {"verdict": NOT_RUN, "messages": [], "gpus": [], "results": [],
              "workloads": list(args.test), "backend": "unknown", "report_dir": None}
    try:
        ids = select_devices(args.gpu, args.context, environ)
    except NoDevices as reason:
        result.update(verdict=SKIPPED, messages=[str(reason)])
        return result
    except BadDevices as reason:
        result["messages"].append(str(reason))
        return result

    executable = shutil.which(args.pantheon, path=environ.get("PATH"))
    if executable is None:
        result["messages"].append(f"'{args.pantheon}' was not found, install it with: pipx install pantheon-gpu")
        return result

    keep = args.report_dir is not None
    workdir = args.report_dir if keep else tempfile.mkdtemp(prefix="pantheon-node-check-")
    os.makedirs(workdir, exist_ok=True)
    try:
        database = os.path.join(workdir, "database")
        unfinished = {}
        rows, gpus, kinds, seen = [], {}, set(), set()
        for workload in args.test:
            # Read only what this workload wrote: the directory may hold the
            # reports of earlier runs, and a hung workload must not be
            # excused by a report that was there before it started.
            before = report_files(database)
            started = time.time_ns()
            code = run_pantheon(executable, workload, ids, args, workdir, environ)
            new_rows, new_gpus, new_kinds = load_reports(database, previous=before, since_ns=started)
            for new_row in new_rows:
                key = json.dumps(new_row, sort_keys=True, default=str)
                if key not in seen:
                    seen.add(key)
                    rows.append(new_row)
            for gpu_id, name in new_gpus.items():
                gpus.setdefault(gpu_id, name)
            kinds |= new_kinds
            if code is None:
                result["messages"].append(f"{workload}: stopped, it did not finish in time")
                unfinished[workload] = f"{workload} did not finish in time and was stopped"
            elif code != 0:
                # A workload that failed on a card has a row that says so.
                # Pantheon's own words are wanted only when it left no row.
                if not new_rows:
                    why = f"pantheon exited with code {code} ({log_tail(workdir, workload)})"
                    result["messages"].append(f"{workload}: {why}")
                    unfinished[workload] = f"{workload} left no result ({why})"
            elif not new_rows:
                unfinished[workload] = f"{workload} left no result"
    finally:
        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)
    if keep:
        result["report_dir"] = workdir

    if "MOCK" in kinds:
        result["backend"] = "cpu"
    elif "NVIDIA" in kinds:
        result["backend"] = "cuda"
    elif "AMD" in kinds:
        result["backend"] = "hip"

    if not rows:
        result["messages"].append("pantheon wrote no report")
        return result
    if result["backend"] == "cpu" and args.platform != "mock":
        result["verdict"] = NO_GPU
        result["messages"].append("pantheon found no GPU with a compiler (nvcc or hipcc) and ran on its "
                                  "CPU backend, no hardware was tested")
        return result

    # Reports are read in the order of their names; show workloads in the order asked for.
    asked = {workload: position for position, workload in enumerate(args.test)}
    rows.sort(key=lambda r: asked.get(r.get("Test Name"), len(asked)))
    judged = ids if ids is not None else sorted({r.get("GPU ID") for r in rows}, key=str)
    for gpu_id in judged:
        result["gpus"].append(assess_gpu(rows, gpu_id, gpus.get(gpu_id, "not found on this node"),
                                         unfinished=list(unfinished.values())))
    result["verdict"] = max((g["verdict"] for g in result["gpus"]), key=lambda v: SEVERITY[v])
    result["results"] = [row_result(r) for r in rows if r.get("GPU ID") in judged]
    if result["backend"] == "cpu":
        result["messages"].append("CPU backend: these results describe no GPU")
    return result


def headline(result):
    """One line that can stand alone, as the reason a node was drained does."""
    if not result["gpus"]:
        return "; ".join(result["messages"]) or "nothing was tested"
    parts = []
    for gpu in sorted(result["gpus"], key=lambda g: -SEVERITY[g["verdict"]]):
        if gpu["verdict"] != HEALTHY:
            why = (gpu["reasons"] or gpu["notes"] or ["no result"])[0]
            parts.append(f"GPU {gpu['gpu_id']} {gpu['verdict']} ({why})")
    healthy = sum(1 for gpu in result["gpus"] if gpu["verdict"] == HEALTHY)
    if healthy:
        parts.append(f"{healthy} GPU {HEALTHY}")
    return "; ".join(parts)


def render(result):
    lines = [f"PANTHEON {result['verdict']}: {headline(result)}"]
    for gpu in result["gpus"]:
        detail = "; ".join(gpu["reasons"]) or ", ".join(gpu["scores"])
        lines.append(f"GPU {gpu['gpu_id']} ({gpu['gpu_name']}): {gpu['verdict']}" + (f", {detail}" if detail else ""))
        lines += [f"  note: {note}" for note in gpu["notes"]]
    if result["gpus"]:
        lines += [f"note: {message}" for message in result["messages"]]
    if result["report_dir"]:
        lines.append(f"reports: {result['report_dir']}")
    return "\n".join(lines)


def _label(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render_textfile(result, now):
    """The result in the text format that Prometheus reads.

    node_exporter's textfile collector picks the file up on its next scrape,
    so a cron job that runs this check gives every GPU a verdict in Grafana.
    """
    code = EXIT_CODE[result["verdict"]]
    lines = [
        "# HELP pantheon_verdict_code Verdict of the last Pantheon node check: "
        "0 healthy, 1 watch, 2 fault, 3 nothing was tested.",
        "# TYPE pantheon_verdict_code gauge",
        f"pantheon_verdict_code {code}",
        "# HELP pantheon_verdict_info The verdict of the last node check as a word.",
        "# TYPE pantheon_verdict_info gauge",
        f'pantheon_verdict_info{{verdict="{_label(result["verdict"])}"}} 1',
        "# HELP pantheon_gpu_verdict_code Verdict of the last node check for one GPU: "
        "0 healthy, 1 watch, 2 fault, 3 nothing was tested.",
        "# TYPE pantheon_gpu_verdict_code gauge",
    ]
    for gpu in result["gpus"]:
        lines.append(f'pantheon_gpu_verdict_code{{gpu="{_label(gpu["gpu_id"])}",'
                     f'name="{_label(gpu["gpu_name"])}"}} {EXIT_CODE[gpu["verdict"]]}')
    metrics = [
        ("pantheon_gpu_score", "Score of a workload on a GPU, in the unit given by the label.", "score", True),
        ("pantheon_gpu_max_temperature_celsius", "Highest GPU temperature during a workload.", "max_temp_c", False),
        ("pantheon_gpu_max_power_watts", "Highest power draw during a workload.", "max_power_w", False),
    ]
    for name, help_text, key, with_unit in metrics:
        rows = [r for r in result["results"] if r.get(key) is not None]
        if not rows:
            continue
        lines += [f"# HELP {name} {help_text}", f"# TYPE {name} gauge"]
        for r in rows:
            labels = f'gpu="{_label(r["gpu_id"])}",workload="{_label(r["workload"])}"'
            if with_unit:
                labels += f',unit="{_label(r["unit"])}"'
            lines.append(f"{name}{{{labels}}} {r[key]:g}")
    if result["results"]:
        lines += ["# HELP pantheon_gpu_workload_failed 1 when the workload failed or did not complete on the GPU.",
                  "# TYPE pantheon_gpu_workload_failed gauge"]
        for r in result["results"]:
            lines.append(f'pantheon_gpu_workload_failed{{gpu="{_label(r["gpu_id"])}",'
                         f'workload="{_label(r["workload"])}"}} {1 if r["failed"] else 0}')
    lines += ["# HELP pantheon_last_run_timestamp_seconds When the last node check finished, as Unix time.",
              "# TYPE pantheon_last_run_timestamp_seconds gauge",
              f"pantheon_last_run_timestamp_seconds {now:.0f}"]
    return "\n".join(lines) + "\n"


def write_textfile(path, text):
    """Write the file in one step, so a scrape never sees half of it."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".pantheon-", suffix=".prom", dir=directory)
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        out.write(text)
    os.replace(temporary, path)


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Exit codes: 0 HEALTHY, 1 WATCH, 2 FAULT, 3 nothing was tested or the run did not finish "
               "(a workload that hung, left no result or skipped itself is never HEALTHY).")
    parser.add_argument("--test", "-t", action="append", metavar="WORKLOAD",
                        help=f"workload or suite to run, can be given several times "
                             f"(default: {' and '.join(DEFAULT_WORKLOADS)})")
    parser.add_argument("--duration", type=int, default=30, help="seconds to run each workload (default: 30)")
    parser.add_argument("--mem", type=int, default=99,
                        help="percentage of the free GPU memory that a workload may use (default: 99)")
    parser.add_argument("--gpu", default="auto",
                        help="'all', a list such as 0,1, or 'auto' to take the GPUs from --context (default: auto)")
    parser.add_argument("--context", choices=["node", "job", "prolog", "epilog"], default="node",
                        help="where the check runs: 'node' tests every GPU, 'prolog' and 'epilog' test the GPUs "
                             "in SLURM_JOB_GPUS, 'job' tests the GPUs in CUDA_VISIBLE_DEVICES (default: node)")
    parser.add_argument("--platform", choices=["auto", "cuda", "hip", "mock"], default="auto",
                        help="backend for Pantheon; 'mock' runs on the CPU, to try the check on a machine "
                             "without a GPU (default: auto)")
    parser.add_argument("--pantheon", default="pantheon", metavar="PATH",
                        help="the pantheon executable (default: pantheon, found in PATH)")
    parser.add_argument("--timeout", type=int, default=0, metavar="SECONDS",
                        help=f"stop a workload after this long (default: the duration plus {COMPILE_ALLOWANCE_S})")
    parser.add_argument("--report-dir", metavar="DIR",
                        help="keep Pantheon's reports and logs in this directory (default: a temporary "
                             "directory that is removed)")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    parser.add_argument("--textfile", metavar="PATH",
                        help="also write the result as Prometheus metrics to this file, for "
                             "node_exporter's textfile collector (for example "
                             "/var/lib/node_exporter/textfile_collector/pantheon.prom)")
    return parser


def main(argv=None, environ=None):
    args = build_parser().parse_args(argv)
    args.test = args.test or list(DEFAULT_WORKLOADS)
    result = check(args, dict(os.environ if environ is None else environ))
    if args.textfile:
        write_textfile(args.textfile, render_textfile(result, time.time()))
    print(json.dumps(result, indent=2) if args.json else render(result))
    return EXIT_CODE[result["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
