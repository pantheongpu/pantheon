"""NVLink and GPU-fabric checks.

These run against a stand-in for NVML, because the machines CI uses have no
NVLink. They show the rules do what they say on recorded states; they cannot
show that NVML reports those states on a real degraded node, which is why a
finding here is a WATCH and never a FAULT.
"""
import ctypes
import types

import pytest

import pantheon

ENABLED, DISABLED = 1, 0
GPU, SWITCH = 0, 2


class _FabricInfo(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int), ("state", ctypes.c_uint), ("healthMask", ctypes.c_uint)]


def _nvml(gpus):
    """A stand-in for pynvml. gpus: list of {"links": [(enabled, remote_type), ...], "fabric": (state, status, mask)}."""

    def link_state(handle, link):
        links = gpus[handle]["links"]
        if link >= len(links):
            raise RuntimeError("invalid argument")
        return ENABLED if links[link][0] else DISABLED

    def remote_type(handle, link):
        return gpus[handle]["links"][link][1]

    def fabric(handle, ref):
        fabric_state = gpus[handle].get("fabric")
        if fabric_state is None:
            raise RuntimeError("not supported")
        info = ref._obj
        info.state, info.status, info.healthMask = fabric_state

    return types.SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetCount=lambda: len(gpus),
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetNvLinkState=link_state,
        nvmlDeviceGetNvLinkRemoteDeviceType=remote_type,
        c_nvmlGpuFabricInfoV_t=_FabricInfo,
        nvmlDeviceGetGpuFabricInfoV=fabric,
        NVML_FEATURE_ENABLED=ENABLED,
    )


def _node(count, active=18, total=18, remote=SWITCH, special=None):
    """count GPUs with `total` links of which `active` are up; special: {gpu: active_links}."""
    gpus = []
    for index in range(count):
        up = (special or {}).get(index, active)
        gpus.append({"links": [(link < up, remote) for link in range(total)]})
    return gpus


def _snapshot(monkeypatch, gpus):
    monkeypatch.setattr(pantheon, "pynvml", _nvml(gpus))
    return pantheon.nvidia_interconnect_snapshot()


def test_a_healthy_nvswitch_node_has_no_finding(monkeypatch):
    snap = _snapshot(monkeypatch, _node(8))
    assert snap["status"] == "supported"
    assert [(e["links_active"], e["links_total"]) for e in snap["gpus"]] == [(18, 18)] * 8
    assert pantheon.interconnect_findings(snap, snap) == {}
    assert pantheon.describe_interconnect(snap) == "NVLink: GPU 0,1,2,3,4,5,6,7: 18 of 18 links active"


def test_one_gpu_with_a_link_down_is_named_and_the_others_are_not(monkeypatch):
    snap = _snapshot(monkeypatch, _node(8, special={3: 17}))
    found = pantheon.interconnect_findings(snap, snap)
    assert list(found) == [3]
    assert found[3] == ["NVLink: 17 of 18 links active; the other GPUs of this node have 18"]
    assert pantheon.describe_interconnect(snap) == (
        "NVLink: GPU 0,1,2,4,5,6,7: 18 of 18 links active; GPU 3: 17 of 18 links active")


def test_a_gpu_with_every_link_down_is_a_finding_only_through_an_nvswitch(monkeypatch):
    switched = _snapshot(monkeypatch, _node(8, special={5: 0}))
    found = pantheon.interconnect_findings(switched, switched)
    assert list(found) == [5] and "none of its 18 links is active" in found[5][0]
    # Two NVLink bridges and two unbridged cards: nothing is wrong with the unbridged ones.
    bridged = _snapshot(monkeypatch, [
        {"links": [(True, GPU)] * 4}, {"links": [(True, GPU)] * 4},
        {"links": [(False, GPU)] * 4}, {"links": [(False, GPU)] * 4},
    ])
    assert pantheon.interconnect_findings(bridged, bridged) == {}


def test_gpus_are_compared_only_with_gpus_that_have_the_same_number_of_links(monkeypatch):
    gpus = _node(2, active=12, total=12) + _node(2, active=18, total=18, special={1: 17})
    snap = _snapshot(monkeypatch, gpus)
    found = pantheon.interconnect_findings(snap, snap)
    assert list(found) == [3]


def test_a_lone_gpu_has_nothing_to_be_compared_with(monkeypatch):
    snap = _snapshot(monkeypatch, _node(1, active=17))
    assert pantheon.interconnect_findings(snap, snap) == {}


def test_a_link_lost_during_the_run_is_reported(monkeypatch):
    before = _snapshot(monkeypatch, _node(2))
    after = _snapshot(monkeypatch, _node(2, special={1: 17}))
    found = pantheon.interconnect_findings(before, after)
    assert any("1 link(s) went down during the run (18 active before, 17 after)" in reason for reason in found[1])
    assert 0 not in found


def test_a_node_with_no_nvlink_is_unsupported_not_clean(monkeypatch):
    snap = _snapshot(monkeypatch, [{"links": []}, {"links": []}])
    assert snap["status"] == "unsupported" and "no NVLink" in snap["detail"]
    assert pantheon.interconnect_findings(snap, snap) == {}
    assert pantheon.describe_interconnect(snap) is None


def test_without_nvml_the_snapshot_says_why_instead_of_failing(monkeypatch):
    monkeypatch.setattr(pantheon, "pynvml", None)
    snap = pantheon.nvidia_interconnect_snapshot()
    assert snap["status"] == "unavailable" and "pynvml" in snap["detail"]
    assert pantheon.interconnect_findings(snap, snap) == {}


def test_nvml_failing_to_start_does_not_raise(monkeypatch):
    def broken():
        raise RuntimeError("driver not loaded")

    stand_in = _nvml(_node(2))
    stand_in.nvmlInit = broken
    monkeypatch.setattr(pantheon, "pynvml", stand_in)
    snap = pantheon.nvidia_interconnect_snapshot()
    assert snap["status"] == "unavailable" and "driver not loaded" in snap["detail"]


@pytest.mark.parametrize("fabric, expected", [
    ((3, 0, 0), None),                              # completed, success
    ((1, 0, 0), "GPU fabric not started, not completed"),
    ((2, 0, 0), "GPU fabric in progress, not completed"),
    ((3, 7, 0), "GPU fabric reports an error (status 7)"),
    ((3, 0, 1), "the driver reports degraded NVLink bandwidth for this GPU"),
    ((0, 0, 0), None),                              # not supported: nothing to say
])
def test_fabric_state_is_judged_only_when_the_fabric_is_supported(monkeypatch, fabric, expected):
    gpus = _node(2)
    gpus[0]["fabric"] = fabric
    snap = _snapshot(monkeypatch, gpus)
    found = pantheon.interconnect_findings(snap, snap)
    if expected is None:
        assert found == {}
    else:
        assert found == {0: [expected]}


def test_a_driver_without_fabric_calls_leaves_the_links_readable(monkeypatch):
    stand_in = _nvml(_node(2))
    del stand_in.c_nvmlGpuFabricInfoV_t
    del stand_in.nvmlDeviceGetGpuFabricInfoV
    monkeypatch.setattr(pantheon, "pynvml", stand_in)
    snap = pantheon.nvidia_interconnect_snapshot()
    assert snap["status"] == "supported" and snap["gpus"][0]["fabric"] is None


def test_the_snapshot_is_plain_json(monkeypatch):
    import json
    gpus = _node(2)
    gpus[0]["fabric"] = (3, 0, 2)
    snap = _snapshot(monkeypatch, gpus)
    assert json.loads(json.dumps(snap)) == snap
    assert snap["gpus"][0]["fabric"] == {"state": 3, "status": 0, "degraded_bandwidth": False}


# --- the verdict --------------------------------------------------------------

def _row(test, score, unit="GB/s", gpu=0):
    row = pantheon.build_result_row(test, gpu, 60, 99, score, unit,
                                    {"max_temp": 60, "max_mem_temp": 0, "throttle_reason": "None"})
    row["RAS Status"], row["RAS Error Delta"] = "CLEAN", "None"
    return row


def test_an_interconnect_finding_makes_a_healthy_card_a_watch():
    rows = [_row("baseline_metrics", 0.0), _row("memory_read", 868.0)]
    clean = pantheon.assess_gpu(rows, 0, "NVIDIA H100 80GB HBM3")
    assert clean["verdict"] == "HEALTHY" and clean["findings"]["interconnect"] == []

    reason = "NVLink: 17 of 18 links active; the other GPUs of this node have 18"
    verdict = pantheon.assess_gpu(rows, 0, "NVIDIA H100 80GB HBM3", interconnect=[reason])
    assert verdict["verdict"] == "WATCH"
    assert verdict["reasons"] == ["interconnect: " + reason]
    assert verdict["summary"] == "interconnect issue"
    assert verdict["findings"]["interconnect"] == [reason]


def test_an_interconnect_finding_never_makes_a_fault():
    rows = [_row("baseline_metrics", 0.0), _row("memory_read", 868.0)]
    verdict = pantheon.assess_gpu(rows, 0, "X", interconnect=["NVLink: none of its 18 links is active"])
    assert verdict["verdict"] == "WATCH"


def test_a_card_that_ran_nothing_stays_incomplete_whatever_the_links_say():
    verdict = pantheon.assess_gpu([_row("baseline_metrics", 0.0)], 0, "X", interconnect=["NVLink: 17 of 18 links active"])
    assert verdict["verdict"] == "WATCH" and verdict["workloads_completed"] == 0


# --- the error counters -------------------------------------------------------

def _counter_nvml(has_nvlink, values):
    """pynvml stand-in for nvidia_nvlink_counters: the NVLink fields answered from `values`."""
    names = {constant: 100 + i for i, constant in enumerate(pantheon.NVLINK_ERROR_FIELDS.values())}
    ids_by_value = {v: k for k, v in names.items()}

    def field_values(handle, field_ids):
        out = []
        for field_id in field_ids:
            ull = types.SimpleNamespace(ullVal=values.get(ids_by_value[field_id], 0))
            out.append(types.SimpleNamespace(nvmlReturn=0, value=ull))
        return out

    def link_state(handle, link):
        if not has_nvlink:
            raise RuntimeError("not supported")
        return ENABLED

    return types.SimpleNamespace(
        nvmlInit=lambda: None, nvmlDeviceGetHandleByIndex=lambda i: i, nvmlDeviceGetFieldValues=field_values,
        nvmlDeviceGetNvLinkState=link_state, NVML_SUCCESS=0, **names)


def test_nvlink_counters_are_recorded_only_on_a_gpu_that_has_nvlink(monkeypatch):
    monkeypatch.setattr(pantheon, "pynvml", _counter_nvml(True, {"NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL": 4}))
    with_links = pantheon.nvidia_nvlink_counters(0)
    assert with_links["status"] == "supported"
    assert with_links["metrics"]["nvlink.replay"]["value"] == 4 and with_links["metrics"]["nvlink.crc_flit"]["value"] == 0

    monkeypatch.setattr(pantheon, "pynvml", _counter_nvml(False, {}))
    without = pantheon.nvidia_nvlink_counters(0)
    assert without["status"] == "unsupported" and without["metrics"] == {}

    monkeypatch.setattr(pantheon, "pynvml", None)
    assert pantheon.nvidia_nvlink_counters(0)["status"] == "unavailable"


def test_a_field_the_gpu_does_not_report_is_unsupported_and_does_not_hide_the_others(monkeypatch):
    """On a V100, NVML answers NOT_SUPPORTED for every NVLink field of a call that includes ecc_data."""
    stand_in = _counter_nvml(True, {"NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL": 7})
    real = stand_in.nvmlDeviceGetFieldValues
    ecc_id = 100 + list(pantheon.NVLINK_ERROR_FIELDS).index("nvlink.ecc_data")

    def all_or_nothing(handle, field_ids):
        out = real(handle, field_ids)
        if ecc_id in field_ids:
            for field in out:
                field.nvmlReturn = 3          # NVML_ERROR_NOT_SUPPORTED
        return out

    stand_in.nvmlDeviceGetFieldValues = all_or_nothing
    monkeypatch.setattr(pantheon, "pynvml", stand_in)
    metrics = pantheon.nvidia_nvlink_counters(0)["metrics"]
    assert metrics["nvlink.ecc_data"]["status"] == "unsupported" and metrics["nvlink.ecc_data"]["value"] is None
    assert metrics["nvlink.replay"]["status"] == "supported" and metrics["nvlink.replay"]["value"] == 7


def test_the_nvlink_source_joins_the_ras_snapshot_on_cuda_only(monkeypatch):
    monkeypatch.setattr(pantheon, "nvidia_ras_snapshot", lambda gpu: pantheon.ras_source("unavailable", detail="x"))
    monkeypatch.setattr(pantheon, "nvidia_nvlink_counters", lambda gpu: pantheon.ras_source("supported", {}))
    monkeypatch.setattr(pantheon, "linux_aer_snapshot", lambda *a, **k: pantheon.ras_source("unsupported"))
    monkeypatch.setattr(pantheon, "linux_aer_events", lambda: ([], ""))
    monkeypatch.setattr(pantheon, "gpu_pci_bdf", lambda *a: "")
    assert "nvlink" in pantheon.collect_ras_snapshot("CUDA", 0)["sources"]
    assert "nvlink" not in pantheon.collect_ras_snapshot("MOCK", 0)["sources"]


def test_nvlink_counters_are_in_the_delta_but_do_not_change_the_ras_status():
    def snap(value):
        return {"sources": {"nvlink": {"status": "supported", "metrics": {
            "nvlink.replay": {"status": "supported", "value": value, "unit": "count"}}}}}
    before, after = snap(1), snap(9)
    delta = pantheon.diff_ras_snapshots(before, after)
    assert [(row["metric"], row["delta"]) for row in delta["metrics"]] == [("nvlink.replay", 8)]
    # The counters moved, and the ECC and PCIe counters could not be read at all:
    # that is UNAVAILABLE, not a clean bill from a source that is only recorded.
    assert pantheon.summarize_ras_delta(delta, before, after)["status"] == "UNAVAILABLE"
    with_ecc = {"sources": {**snap(9)["sources"], "vendor_ras": {"status": "supported", "metrics": {}}}}
    assert pantheon.summarize_ras_delta(delta, with_ecc, with_ecc)["status"] == "CLEAN"
