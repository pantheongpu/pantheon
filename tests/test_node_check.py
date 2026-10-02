"""Tests of integrations/pantheon_node_check.py.

A small program stands in for Pantheon: it records how it was called and
writes the reports a scenario describes. The check itself is the real one.
"""
import json
import os
import re
import stat
import sys
import textwrap

import pytest

from integrations import pantheon_node_check as node_check

FAKE_PANTHEON = textwrap.dedent('''\
    #!{python}
    import json, os, sys, time
    scenario = json.load(open(os.environ["FAKE_SCENARIO"]))
    args = sys.argv[1:]
    workload = args[args.index("--test") + 1]
    with open(os.environ["FAKE_CALLS"], "a") as calls:
        calls.write(json.dumps({{
            "args": args,
            "cwd_is_empty_of_reports": not os.path.exists("database"),
            "device_variables": sorted(v for v in os.environ
                                       if v.endswith("_VISIBLE_DEVICES")),
        }}) + "\\n")
    print("FINAL SUMMARY REPORT for", workload)
    time.sleep(scenario.get("sleep", 0))
    rows = [r for r in scenario.get("rows", []) if r["Test Name"] == workload]
    if rows or scenario.get("empty_report"):
        os.makedirs("database", exist_ok=True)
        report = {{"gpu_static_info": scenario.get("gpus", []), "test_results": rows}}
        name = "database/pantheon_report_" + workload
        json.dump(report, open(name + ".json", "w"))
        # Pantheon writes the row of a completed workload a second time.
        for row in rows:
            if row.get("Unit") != "ERR":
                twin = name + "_0001_%s_gpu%s.json" % (workload, row["GPU ID"])
                json.dump({{"gpu_static_info": scenario.get("gpus", []),
                           "test_results": [row]}}, open(twin, "w"))
    sys.exit(scenario.get("exit", {{}}).get(workload, 0))
''')

NVIDIA = [{"id": 0, "type": "NVIDIA", "name": "NVIDIA H100 PCIe"},
          {"id": 1, "type": "NVIDIA", "name": "NVIDIA H100 PCIe"}]
CPU = [{"id": 0, "type": "MOCK", "name": "Mock GPU"}]


def row(workload, gpu=0, **fields):
    """A result row as Pantheon writes it for a workload that passed."""
    result = {
        "Test Name": workload, "GPU ID": gpu, "Score": 1971.4, "Unit": "GB/s",
        "Status": None, "Failure Stage": None, "RAS Status": "CLEAN",
        "RAS Error Delta": "None", "Limit Reason": "None", "Max Temp (C)": 72.0,
    }
    result.update(fields)
    return result


class Node:
    """A machine with the stand-in for Pantheon installed."""

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        executable = self.bin / "pantheon"
        executable.write_text(FAKE_PANTHEON.format(python=sys.executable))
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
        self.scenario = tmp_path / "scenario.json"
        self.calls_file = tmp_path / "calls.jsonl"
        self.environ = {
            "PATH": f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}",
            "FAKE_SCENARIO": str(self.scenario),
            "FAKE_CALLS": str(self.calls_file),
        }

    def run(self, capsys, *argv, **scenario):
        self.scenario.write_text(json.dumps(scenario))
        code = node_check.main(list(argv), dict(self.environ))
        return code, capsys.readouterr().out

    @property
    def calls(self):
        if not self.calls_file.exists():
            return []
        return [json.loads(line) for line in self.calls_file.read_text().splitlines()]


@pytest.fixture
def node(tmp_path):
    return Node(tmp_path)


def both_workloads(**fields):
    """Two workloads on two cards; the fields change march_test on GPU 1."""
    return [row("memory_read", 0), row("memory_read", 1),
            row("march_test", 0, Unit="march-ops/s"),
            row("march_test", 1, **{"Unit": "march-ops/s", **fields})]


def test_healthy_cards_give_exit_code_zero(node, capsys):
    code, out = node.run(capsys, gpus=NVIDIA, rows=both_workloads())
    assert code == 0
    assert out.splitlines()[0] == "PANTHEON HEALTHY: 2 GPU HEALTHY"
    assert "GPU 1 (NVIDIA H100 PCIe): HEALTHY" in out


def test_each_workload_is_one_pantheon_run_on_every_gpu(node, capsys):
    node.run(capsys, "--duration", "12", "--mem", "50", gpus=NVIDIA, rows=both_workloads())
    assert [c["args"] for c in node.calls] == [
        ["--test", "memory_read", "--duration", "12", "--mem", "50", "--gpu", "all"],
        ["--test", "march_test", "--duration", "12", "--mem", "50", "--gpu", "all"],
    ]


def test_a_failed_memory_test_is_a_fault(node, capsys):
    code, out = node.run(capsys, gpus=NVIDIA, rows=both_workloads(Unit="ERR", Score=0.0),
                         exit={"march_test": 1})
    assert code == 2
    assert out.splitlines()[0] == ("PANTHEON FAULT: GPU 1 FAULT (march_test failed: memory errors detected "
                                   "or the workload aborted); 1 GPU HEALTHY")
    assert "GPU 1 (NVIDIA H100 PCIe): FAULT, march_test failed" in out
    # The row says what happened, so Pantheon's last words are left out.
    assert "exited with code" not in out


def test_counted_errors_are_a_fault_although_pantheon_exits_zero(node, capsys):
    fields = {"RAS Status": "ERROR", "RAS Error Delta": "ecc.uncorrected +2"}
    code, out = node.run(capsys, gpus=NVIDIA, rows=both_workloads(**fields))
    assert code == 2
    assert "march_test: uncorrectable errors (ecc.uncorrected +2)" in out


def test_a_workload_that_is_no_memory_test_and_fails_is_a_watch(node, capsys):
    code, out = node.run(capsys, "--test", "tensor_virus", gpus=NVIDIA,
                         rows=[row("tensor_virus", 0, Unit="ERR", Score=0.0)],
                         exit={"tensor_virus": 1})
    assert code == 1
    assert "tensor_virus did not complete" in out


@pytest.mark.parametrize("fields, reason", [
    ({"Limit Reason": "Thermal", "Max Temp (C)": 95.0}, "thermally throttled, GPU at 95 C"),
    ({"Max Temp (C)": 91.0}, "GPU reached 91 C"),
    ({"Max Mem Temp (C)": 96.0}, "memory reached 96 C"),
    ({"RAS Status": "WARNING", "RAS Error Delta": "vendor_ras.pcie.bad_tlp +1972"},
     "correctable errors (vendor_ras.pcie.bad_tlp +1972)"),
])
def test_heat_and_correctable_errors_are_a_watch(node, capsys, fields, reason):
    code, out = node.run(capsys, gpus=NVIDIA, rows=both_workloads(**fields))
    assert code == 1
    assert out.splitlines()[0] == f"PANTHEON WATCH: GPU 1 WATCH (march_test: {reason}); 1 GPU HEALTHY"


def test_a_link_changing_power_state_is_a_note(node, capsys):
    fields = {"RAS Status": "WARNING", "RAS Error Delta": "vendor_ras.pcie.l0_to_recovery +1"}
    code, out = node.run(capsys, gpus=NVIDIA, rows=both_workloads(**fields))
    assert code == 0
    assert "PCIe link recovery" in out


def test_the_cpu_backend_is_not_a_pass(node, capsys):
    code, out = node.run(capsys, "--test", "memory_read", gpus=CPU, rows=[row("memory_read")])
    assert code == 3
    assert out.startswith("PANTHEON NO GPU TESTED: ")
    assert "no hardware was tested" in out


def test_the_cpu_backend_can_be_asked_for(node, capsys):
    code, out = node.run(capsys, "--test", "memory_read", "--platform", "mock",
                         gpus=CPU, rows=[row("memory_read")])
    assert code == 0
    assert node.calls[0]["args"][-2:] == ["--platform", "mock"]
    assert "CPU backend: these results describe no GPU" in out


def test_no_report_is_not_a_pass(node, capsys):
    code, out = node.run(capsys, "--test", "no_such_workload", exit={"no_such_workload": 1})
    assert code == 3
    assert out.startswith("PANTHEON NOT RUN: ")
    assert "pantheon exited with code 1" in out
    assert "pantheon wrote no report" in out


def test_a_missing_pantheon_is_reported(node, capsys):
    code, out = node.run(capsys, "--pantheon", "pantheon-that-is-not-installed")
    assert code == 3
    assert "was not found" in out
    assert node.calls == []


def test_a_run_that_hangs_is_stopped(node, capsys):
    code, out = node.run(capsys, "--test", "memory_read", "--timeout", "1", sleep=30)
    assert code == 3
    assert "did not finish in time" in out


def test_a_requested_card_without_a_result_is_incomplete(node, capsys):
    code, out = node.run(capsys, "--test", "memory_read", "--gpu", "0,1",
                         gpus=NVIDIA, rows=[row("memory_read", 0)])
    assert code == 3
    assert out.splitlines()[0] == "PANTHEON INCOMPLETE: GPU 1 INCOMPLETE (no workload completed); 1 GPU HEALTHY"


def test_an_epilog_tests_the_cards_of_the_job_by_node_number(node, capsys):
    node.environ.update(SLURM_JOB_GPUS="1", CUDA_VISIBLE_DEVICES="0", ROCR_VISIBLE_DEVICES="0")
    code, out = node.run(capsys, "--context", "epilog", "--test", "memory_read",
                         gpus=NVIDIA, rows=[row("memory_read", 1)])
    assert code == 0
    assert node.calls[0]["args"][-2:] == ["--gpu", "1"]
    # The variables would renumber the cards, so Pantheon runs without them.
    assert node.calls[0]["device_variables"] == []
    assert out.splitlines()[0] == "PANTHEON HEALTHY: 1 GPU HEALTHY"


def test_an_epilog_of_a_job_without_a_card_tests_nothing(node, capsys):
    code, out = node.run(capsys, "--context", "epilog")
    assert code == 0
    assert out.strip() == "PANTHEON SKIPPED: the job has no GPU, nothing was tested"
    assert node.calls == []


def test_a_job_tests_the_cards_it_can_see(node, capsys):
    node.environ.update(CUDA_VISIBLE_DEVICES="0,1")
    node.run(capsys, "--context", "job", "--test", "memory_read",
             gpus=NVIDIA, rows=[row("memory_read", 0), row("memory_read", 1)])
    assert node.calls[0]["args"][-2:] == ["--gpu", "0,1"]


def test_cards_named_by_uuid_are_not_guessed(node, capsys):
    node.environ.update(CUDA_VISIBLE_DEVICES="GPU-9214ae6b")
    code, out = node.run(capsys, "--context", "job")
    assert code == 3
    assert "name them with --gpu" in out
    assert node.calls == []


def test_the_cards_on_the_command_line_win(node, capsys):
    node.environ.update(SLURM_JOB_GPUS="1")
    node.run(capsys, "--context", "epilog", "--gpu", "0", "--test", "memory_read",
             gpus=NVIDIA, rows=[row("memory_read", 0)])
    assert node.calls[0]["args"][-2:] == ["--gpu", "0"]


def test_reports_are_kept_when_asked_for(node, capsys, tmp_path):
    kept = tmp_path / "kept"
    code, out = node.run(capsys, "--test", "memory_read", "--report-dir", str(kept),
                         gpus=NVIDIA, rows=[row("memory_read", 0)])
    assert code == 0
    assert (kept / "database" / "pantheon_report_memory_read.json").exists()
    assert (kept / "memory_read.log").read_text().startswith("FINAL SUMMARY REPORT")
    assert f"reports: {kept}" in out


def test_reports_are_removed_otherwise(node, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", None)
    node.run(capsys, "--test", "memory_read", gpus=NVIDIA, rows=[row("memory_read", 0)])
    assert list((tmp_path / "tmp").iterdir()) == []


def test_the_result_can_be_read_by_a_program(node, capsys):
    code, out = node.run(capsys, "--json", gpus=NVIDIA, rows=both_workloads())
    result = json.loads(out)
    assert code == 0
    assert result["verdict"] == "HEALTHY"
    assert result["backend"] == "cuda"
    assert [g["gpu_id"] for g in result["gpus"]] == [0, 1]
    assert result["gpus"][0]["scores"] == ["memory_read 1971.4 GB/s", "march_test 1971.4 march-ops/s"]


def test_the_first_line_names_the_worst_card_first(node, capsys):
    rows = [row("memory_read", 0, **{"Max Temp (C)": 92.0}), row("memory_read", 1),
            row("march_test", 0, Unit="march-ops/s"),
            row("march_test", 1, Unit="ERR", Score=0.0)]
    code, out = node.run(capsys, gpus=NVIDIA, rows=rows, exit={"march_test": 1})
    assert code == 2
    assert out.splitlines()[0] == ("PANTHEON FAULT: GPU 1 FAULT (march_test failed: memory errors detected "
                                   "or the workload aborted); GPU 0 WATCH (memory_read: GPU reached 92 C)")


def test_the_numbers_of_each_workload_are_in_the_result(node, capsys):
    rows = [row("memory_read", 0, **{"Max Power (W)": 288.5}),
            row("march_test", 0, Unit="ERR", Score=0.0)]
    code, out = node.run(capsys, "--json", gpus=NVIDIA[:1], rows=rows, exit={"march_test": 1})
    result = json.loads(out)
    assert [r["workload"] for r in result["results"]] == ["memory_read", "march_test"]
    first, second = result["results"]
    assert first["score"] == 1971.4 and first["unit"] == "GB/s"
    assert first["max_temp_c"] == 72.0 and first["max_power_w"] == 288.5
    assert first["failed"] is False
    # A failed workload has no score, however Pantheon filled the column.
    assert second["score"] is None and second["failed"] is True


def test_the_prometheus_file_carries_the_verdict_and_the_numbers(node, capsys, tmp_path):
    prom = tmp_path / "textfile" / "pantheon.prom"
    rows = both_workloads(**{"RAS Status": "WARNING", "RAS Error Delta": "vendor_ras.pcie.bad_tlp +12"})
    code, out = node.run(capsys, "--textfile", str(prom), gpus=NVIDIA, rows=rows)
    assert code == 1
    text = prom.read_text()
    assert "pantheon_verdict_code 1\n" in text
    assert 'pantheon_verdict_info{verdict="WATCH"} 1\n' in text
    assert 'pantheon_gpu_verdict_code{gpu="0",name="NVIDIA H100 PCIe"} 0\n' in text
    assert 'pantheon_gpu_verdict_code{gpu="1",name="NVIDIA H100 PCIe"} 1\n' in text
    assert 'pantheon_gpu_score{gpu="0",workload="memory_read",unit="GB/s"} 1971.4\n' in text
    assert 'pantheon_gpu_score{gpu="1",workload="march_test",unit="march-ops/s"} 1971.4\n' in text
    assert 'pantheon_gpu_max_temperature_celsius{gpu="0",workload="memory_read"} 72\n' in text
    assert 'pantheon_gpu_workload_failed{gpu="1",workload="march_test"} 0\n' in text
    assert "pantheon_last_run_timestamp_seconds " in text
    # Prometheus is strict about the format: every metric has HELP and TYPE, and
    # the file ends with a newline.
    for name in ("pantheon_verdict_code", "pantheon_gpu_verdict_code", "pantheon_gpu_score",
                 "pantheon_gpu_max_temperature_celsius", "pantheon_gpu_workload_failed"):
        assert f"# HELP {name} " in text and f"# TYPE {name} gauge" in text
    assert text.endswith("\n")
    # The file is written in one step: no half-written file is left beside it.
    assert sorted(p.name for p in prom.parent.iterdir()) == ["pantheon.prom"]


def test_the_prometheus_file_says_when_nothing_was_tested(node, capsys, tmp_path):
    prom = tmp_path / "pantheon.prom"
    code, out = node.run(capsys, "--textfile", str(prom), "--test", "memory_read",
                         gpus=CPU, rows=[row("memory_read")])
    assert code == 3
    text = prom.read_text()
    assert "pantheon_verdict_code 3\n" in text
    assert 'pantheon_verdict_info{verdict="NO GPU TESTED"} 1\n' in text
    assert "pantheon_gpu_score" not in text


DASHBOARD = os.path.join(os.path.dirname(__file__), "..", "integrations", "prometheus",
                         "grafana-dashboard.json")


def test_grafana_dashboard_shows_every_exported_metric_and_nothing_else():
    """The dashboard and the exporter must not drift apart."""
    with open(DASHBOARD, encoding="utf-8") as fp:
        dashboard = json.load(fp)
    result = {
        "verdict": "WATCH",
        "gpus": [{"gpu_id": 0, "gpu_name": "NVIDIA H100 PCIe", "verdict": "HEALTHY"},
                 {"gpu_id": 1, "gpu_name": "NVIDIA H100 PCIe", "verdict": "WATCH"}],
        "results": [{"gpu_id": 0, "workload": "memory_read", "unit": "GB/s", "score": 1971.4,
                     "max_temp_c": 72, "max_power_w": 288.5, "failed": False}],
    }
    exported = set(re.findall(r"^# TYPE (pantheon_\w+)", node_check.render_textfile(result, 1790800000), re.M))
    queries = [t["expr"] for panel in dashboard["panels"] for t in panel["targets"]]
    queries += [v["definition"] for v in dashboard["templating"]["list"] if v["type"] == "query"]
    used = set(re.findall(r"pantheon_\w+", " ".join(queries)))
    assert used == exported

    # Every panel and variable goes through the datasource the import asks for.
    for panel in dashboard["panels"]:
        assert panel["datasource"]["uid"] == "${DS_PROMETHEUS}"
        for t in panel["targets"]:
            assert t["datasource"]["uid"] == "${DS_PROMETHEUS}"
    assert [v["name"] for v in dashboard["templating"]["list"]] == ["DS_PROMETHEUS", "instance", "gpu"]

    # The verdict mappings name every exit code the check can answer with.
    codes = {str(code) for code in node_check.EXIT_CODE.values()}
    for title in ("Node verdict", "Cards"):
        panel = next(p for p in dashboard["panels"] if p["title"] == title)
        mapping = panel["fieldConfig"]["defaults"]["mappings"][0]["options"]
        assert set(mapping) == codes
        assert mapping["0"]["text"] == "HEALTHY" and mapping["2"]["text"] == "FAULT"

    # Panels tile the grid without overlapping.
    cells = set()
    for panel in dashboard["panels"]:
        g = panel["gridPos"]
        assert g["x"] + g["w"] <= 24
        for x in range(g["x"], g["x"] + g["w"]):
            for y in range(g["y"], g["y"] + g["h"]):
                assert (x, y) not in cells, f"{panel['title']} overlaps another panel"
                cells.add((x, y))


def test_label_values_are_escaped():
    assert node_check._label('a "quoted" name\\') == 'a \\"quoted\\" name\\\\'
