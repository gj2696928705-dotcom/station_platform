from copy import deepcopy
from datetime import datetime, timedelta, timezone
import asyncio

import pytest

from station_platform.domain import capability, check_local, evaluation, resources, schedule
from capacity_alignment.forecast_capacity import BASE, PV, freeze, verify
from station_platform.forecast_api import attach
from station_platform.legacy.demand import MockStationAdapter
from station_platform.storage import Store
from station_platform.util import PlatformError, digest, parse_time


def sample(at="2025-01-01T00:00:00+08:00", station="mock-station-001"):
    start = parse_time(at)
    end = start + timedelta(hours=1)
    meta = {"station_id": station, "meter_boundary": "AC-BUS", "simulation": True,
            "issued_at": (start-timedelta(hours=1)).isoformat(),
            "available_at": (start-timedelta(hours=1)).isoformat(), "expires_at": end.isoformat()}
    record = {**meta, "record_id": "load-1", "target_semantics": BASE,
              "unit": "kW", "value_semantics": "interval_average_power", "status": "available",
              "model_version": "test-1", "method_hash": digest("test-method"),
              "input_hash": digest("test-input"), "source_ref": "hand-calculated-fixture",
              "values": [{"start": (start+timedelta(minutes=15*i)).isoformat(),
                          "end": (start+timedelta(minutes=15*(i+1))).isoformat(), "value": 100} for i in range(4)]}
    return {"station_id": station, "meter_boundary": "AC-BUS", "task_id": "test-task", "task_version": 1,
            "execution_mode": "offline_replay", "as_of": at, "start": at, "end": end.isoformat(), "interval_minutes": 15,
            "binding": {**meta, "evidence_ref": "test-simulation-binding"},
            "state": {**meta, "quality": "A", "protection_active": False, "manual_takeover": False,
                      "limits_known": True, "grid_import_limit_kw": 150,
                      "devices": [{"asset_id": f"charger-{i}", "max_power_kw": 200, "availability": "available",
                                   "frozen": False, "occupied_intervals": []} for i in range(1, 4)]},
            "load_forecast": record,
            "pv_forecast": {**deepcopy(record), "record_id": "pv-1", "target_semantics": PV,
                            "values": [{"start": at, "end": end.isoformat(), "value": 20}]},
            "commitments": {**meta, "status": "known_empty", "source_ref": "explicit-empty-list",
                            "coverage_start": at, "coverage_end": end.isoformat(), "sessions": []}}


def session(inputs, power=30, sid="old-1", asset="charger-1"):
    return {**deepcopy(inputs["load_forecast"]), "session_id": sid, "asset_id": asset,
            "target_semantics": "committed_charge_ac",
            "values": [{**row, "value": power} for row in inputs["load_forecast"]["values"]]}


def test_manual_power_balance_and_exclusive_existing_session():
    inputs = sample()
    inputs["commitments"].update(status="known", sessions=[session(inputs)])
    context = freeze(inputs)
    assert context["status"] == "available"
    for slot in context["slots"]:
        assert slot["base_load_kw"] == 100
        assert slot["committed_charge_kw"] == 30
        assert slot["existing_load_kw"] == 130
        assert slot["grid_new_limit_kw"] == 40  # 150 + 20 - (100 + 30)
        assert slot["device_new_limit_kw"] == 400  # occupied port contributes zero
        assert slot["capacity_kw"] == 40
        assert 130 + 40 - 20 == 150
    assert verify(context) == context


def test_manual_non_quarter_start_overlap():
    inputs = sample()
    inputs["start"] = inputs["as_of"] = "2025-01-01T00:07:30+08:00"
    inputs["end"] = "2025-01-01T01:07:30+08:00"
    for record in (inputs["state"], inputs["binding"], inputs["load_forecast"], inputs["pv_forecast"], inputs["commitments"]):
        record["expires_at"] = inputs["end"]
    rows = inputs["load_forecast"]["values"]
    for row, value in zip(rows, [40, 60, 80, 100]):
        row["value"] = value
    rows.append({"start": "2025-01-01T01:00:00+08:00", "end": "2025-01-01T01:15:00+08:00", "value": 120})
    inputs["pv_forecast"]["values"][0]["end"] = inputs["end"]
    inputs["commitments"]["coverage_end"] = inputs["end"]
    context = freeze(inputs)
    assert context["status"] == "available"
    assert [s["base_load_kw"] for s in context["slots"]] == [50, 70, 90, 110]
    assert [s["capacity_kw"] for s in context["slots"]] == [120, 100, 80, 60]
    assert [r["overlap_seconds"] for r in context["slots"][0]["sources"]["load"]] == [450, 450]


def test_hourly_overlap_and_explicit_energy_conversion():
    inputs = sample("2025-01-01T09:50:00+08:00")
    inputs["pv_forecast"]["values"] = [
        {"start": "2025-01-01T09:00:00+08:00", "end": "2025-01-01T10:00:00+08:00", "value": 200},
        {"start": "2025-01-01T10:00:00+08:00", "end": "2025-01-01T11:00:00+08:00", "value": 400}]
    inputs["load_forecast"].update(unit="kWh", value_semantics="interval_energy")
    for row in inputs["load_forecast"]["values"]:
        row["value"] = 25  # 25 kWh / .25 h = 100 kW
    context = freeze(inputs)
    assert context["status"] == "available"
    assert context["slots"][0]["pv_available_kw"] == pytest.approx(800/3)
    assert context["slots"][0]["base_load_kw"] == 100
    assert [s["pv_available_kw"] for s in context["slots"]][1:] == [400]*3


@pytest.mark.parametrize("case,code", [
    ("station", "STATION_MISMATCH"), ("boundary", "METER_BOUNDARY_MISMATCH"),
    ("unit", "UNIT_SEMANTICS_MISMATCH"), ("mixed_unit", "UNIT_SEMANTICS_MISMATCH"),
    ("total_load", "TARGET_SEMANTICS_MISMATCH"), ("late", "LATE_INPUT"),
    ("expired", "INPUT_EXPIRED"), ("unknown", "COMMITMENT_UNKNOWN"),
    ("duplicate", "COMMITMENT_DUPLICATE"), ("duplicate_asset", "DEVICE_DUPLICATE"),
    ("gap", "COVERAGE_INCOMPLETE"), ("overlap", "NATIVE_OVERLAP"),
    ("grid_conflict", "EXISTING_GRID_CONFLICT"), ("new_duplicate", "COMMITMENT_NEW_DUPLICATE"),
    ("future_observation", "FUTURE_OBSERVATION"), ("late_weather", "LATE_WEATHER"),
    ("bad_number", "NUMBER_INVALID"), ("device_conflict", "EXISTING_DEVICE_CONFLICT"),
    ("session_conflict", "EXISTING_SESSION_CONFLICT"), ("missing_field", "INPUT_INVALID"),
    ("unknown_occupancy", "INPUT_INVALID"), ("protection", "LOCAL_RESTRICTED"),
    ("missing_commit_coverage", "COMMITMENT_COVERAGE_INCOMPLETE"),
])
def test_invalid_inputs_do_not_advertise_capacity(case, code):
    inputs = sample()
    if case == "station": inputs["pv_forecast"]["station_id"] = "different"
    elif case == "boundary": inputs["pv_forecast"]["meter_boundary"] = "net-grid"
    elif case == "unit": inputs["load_forecast"]["unit"] = "kWh"
    elif case == "mixed_unit": inputs["load_forecast"]["values"][0]["unit"] = "kWh"
    elif case == "total_load": inputs["load_forecast"]["target_semantics"] = "total_load"
    elif case == "late": inputs["pv_forecast"]["available_at"] = inputs["end"]
    elif case == "expired": inputs["pv_forecast"]["expires_at"] = inputs["as_of"]
    elif case == "unknown": inputs["commitments"]["status"] = "energy_only"
    elif case == "duplicate": inputs["commitments"].update(status="known", sessions=[session(inputs), session(inputs)])
    elif case == "duplicate_asset": inputs["state"]["devices"].append(deepcopy(inputs["state"]["devices"][0]))
    elif case == "gap": inputs["pv_forecast"]["values"][0]["start"] = inputs["load_forecast"]["values"][1]["start"]
    elif case == "overlap": inputs["load_forecast"]["values"].append(inputs["load_forecast"]["values"][0])
    elif case == "grid_conflict": inputs["state"]["grid_import_limit_kw"] = 20
    elif case == "new_duplicate":
        inputs["new_session_ids"] = ["old-1"]
        inputs["commitments"].update(status="known", sessions=[session(inputs)])
    elif case == "future_observation": inputs["load_forecast"]["values"][0]["source_observation_end"] = inputs["end"]
    elif case == "late_weather": inputs["pv_forecast"].update(sampling_layout="day13_07_19", weather_issued_at=inputs["end"])
    elif case == "bad_number": inputs["load_forecast"]["values"][0]["value"] = True
    elif case == "device_conflict": inputs["commitments"].update(status="known", sessions=[session(inputs, power=300)])
    elif case == "session_conflict": inputs["commitments"].update(status="known", sessions=[session(inputs), session(inputs, sid="old-2")])
    elif case == "missing_field": del inputs["load_forecast"]["input_hash"]
    elif case == "unknown_occupancy": del inputs["state"]["devices"][0]["occupied_intervals"]
    elif case == "protection": inputs["state"]["protection_active"] = True
    elif case == "missing_commit_coverage": inputs["commitments"]["coverage_end"] = inputs["start"]
    context = freeze(inputs)
    assert context["status"] == "unavailable"
    assert code in {i["code"] for i in context["issues"]}
    assert all(s["capacity_kw"] is None for s in context["slots"])


def request_and_config(mode="simulation_use", at="2025-01-01T00:00:00+08:00"):
    inputs = sample(at)
    adapter = MockStationAdapter()
    adapter.site["energy"]["grid_import_limit_kw"] = 150
    demand = {"vehicles": [{"vehicle_id": "new-1", "requested_energy_kwh": 10,
                            "arrival_minutes": 0, "departure_minutes": 60, "priority": "ordinary"}]}
    request = adapter.capture(demand, at=parse_time(at))
    request["station_state_snapshot"]["source_state_id"] = "simulation-state-1"
    return request, {"mode": mode, "inputs": inputs}


def test_resource_search_evaluation_share_frozen_balance_and_restart(tmp_path):
    original, config = request_and_config()
    request = attach(original, config, "task-1", 1)
    r, s = resources(request), schedule(request)
    e = evaluation(request, r, s)
    assert r["frozen_context_hash"] == s["frozen_context_hash"] == e["frozen_context_hash"]
    assert capability(None, request) == r
    assert r["charging_headroom_kw"] == [70]*4
    # Existing base 100, PV 20, new energy 10: import = 80*1h + 10 = 90.
    assert all(row["grid_import_kwh"] == 90 for row in e["candidate_evaluations"])
    assert original["station_state_snapshot"]["energy"]["base_load_kw"] == [50]*4
    store_path = tmp_path/"tasks.sqlite3"
    Store(store_path).put("tasks", "task-1", {"request": request})
    config["inputs"]["load_forecast"]["values"][0]["value"] = 9999
    restored = Store(store_path).get("tasks", "task-1")["request"]
    assert restored == request
    assert evaluation(restored, resources(restored), schedule(restored))["candidate_evaluations"] == e["candidate_evaluations"]


def test_shadow_does_not_change_decisions_even_when_unavailable():
    original, config = request_and_config("shadow", datetime.now(timezone.utc).replace(microsecond=0).isoformat())
    baseline = evaluation(original, resources(original), schedule(original))
    for broken in (False, True):
        if broken: config["inputs"]["commitments"]["status"] = "unknown"
        request = attach(original, config, "shadow-1", 1)
        assert request["station_state_snapshot"] == original["station_state_snapshot"]
        result = evaluation(request, resources(request), schedule(request))
        assert result["candidate_evaluations"] == baseline["candidate_evaluations"]
        assert result["selected_candidate"] == baseline["selected_candidate"]
        assert result["forecast_status"] == "not_used"


def test_unavailable_simulation_blocks_search_and_context_mixing():
    original, config = request_and_config()
    config["inputs"]["commitments"]["status"] = "unknown"
    request = attach(original, config, "task-1", 1)
    assert resources(request)["charging_headroom_kw"] is None
    with pytest.raises(PlatformError, match="仿真预测上下文不可用"):
        schedule(request)
    config["inputs"]["commitments"]["status"] = "known_empty"
    request = attach(original, config, "task-1", 1)
    other = attach(original, config, "task-1", 2)
    with pytest.raises(PlatformError) as error:
        evaluation(request, resources(other), schedule(request))
    assert error.value.code == "CONTEXT_REFERENCE_MISMATCH"


def test_live_consumption_expiry_preserves_archive_and_withholds_capacity(monkeypatch):
    at = datetime.now(timezone.utc).replace(microsecond=0)
    original, config = request_and_config(at=at.isoformat())
    config["inputs"]["execution_mode"] = "simulation"
    original["station_state_snapshot"]["source_expires_at"] = (at+timedelta(seconds=20)).isoformat()
    request = attach(original, config, "task-expiry", 1)
    frozen_hash = request["forecast_context"]["context"]["context_hash"]
    assert resources(request)["usable"]
    monkeypatch.setattr("capacity_alignment.forecast_runtime.now", lambda: at+timedelta(seconds=21))
    result = resources(request)
    assert result["forecast_status"] == "expired" and not result["usable"]
    assert result["charging_headroom_kw"] is None
    assert all(s["capacity_kw"] is None for s in result["slots"])
    assert request["forecast_context"]["context"]["context_hash"] == frozen_hash
    with pytest.raises(PlatformError) as error: schedule(request)
    assert error.value.code == "FORECAST_CONTEXT_EXPIRED"


@pytest.mark.parametrize("field", ["input", "result", "snapshot", "mode"])
def test_frozen_context_tampering_is_detected(field):
    original, config = request_and_config()
    request = attach(original, config, "task-1", 1)
    if field == "input": request["forecast_context"]["context"]["inputs"]["task_version"] = 2
    elif field == "result": request["forecast_context"]["context"]["slots"][0]["capacity_kw"] = 999
    elif field == "snapshot": request["station_state_snapshot"]["energy"]["base_load_kw"][0] = 0
    elif field == "mode": request["forecast_context"]["mode"] = "shadow"
    with pytest.raises(PlatformError) as error: schedule(request)
    assert error.value.code == "CONTEXT_HASH_MISMATCH"


def test_disabled_and_latest_local_gate():
    at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    original, config = request_and_config(at=at)
    assert attach(original, {"mode": "disabled", "inputs": {"garbage": True}}, "task", 1) == original
    request = attach(original, config, "task-1", 1)
    candidate = evaluation(request, resources(request), schedule(request))["selected_candidate"]
    adapter = MockStationAdapter()
    adapter.site["energy"]["grid_import_limit_kw"] = 1  # fresh site vetoes forecast-derived plan
    adapter.site["energy"]["pv_available_kw"] = [0]*4
    state = {"station_id": "mock-station-001", "id": "live-2", "revision": 2, "quality": "A",
             "expires_at": (parse_time(at)+timedelta(hours=1)).isoformat(), "site": adapter.site}
    plan = {"id": "plan", "task_id": "task-1", "version": 1, "station_id": "mock-station-001",
            "request": request, "selected_candidate": candidate, "budget_kw": None, "simulation": True}
    plan["fingerprint"] = digest({k: plan[k] for k in ["task_id", "version", "station_id", "request", "selected_candidate", "budget_kw"]})
    assert "GRID_CAPACITY_LIMIT" in check_local(plan, state)["violations"]


@pytest.mark.parametrize("mode", ["shadow", "simulation_use"])
def test_http_task_freezes_and_consumes_config(stack, mode):
    at = datetime.now(timezone.utc).replace(microsecond=0)
    inputs = sample((at-timedelta(hours=1)).isoformat())
    # Broad explicit coverage allows the server to choose its exact capture time.
    end = (at+timedelta(hours=2)).isoformat()
    for key in ("load_forecast", "pv_forecast"):
        inputs[key]["values"] = [{"start": inputs["start"], "end": end, "value": 100 if key == "load_forecast" else 20}]
    for key in ("binding", "load_forecast", "pv_forecast", "commitments"):
        inputs[key]["expires_at"] = end
    inputs["commitments"]["coverage_end"] = end
    inputs["execution_mode"] = "simulation"
    payload = {"type": "tariff", "demand": {"vehicles": [{"vehicle_id": "new-http", "requested_energy_kwh": 10,
               "arrival_minutes": 0, "departure_minutes": 60, "priority": "ordinary"}]}, "forecast": {"mode": mode, "inputs": inputs}}
    response = stack.client.post("/api/tasks", json={**payload, "submission_key": f"forecast-{mode}"})
    assert response.status_code == 200, response.text
    task = stack.wait_plan(response.json()["id"])
    artifacts = task["artifacts"]
    hashes = {artifacts[key]["frozen_context_hash"] for key in ("resource", "service", "evaluation")}
    assert len(hashes) == 1
    assert (None in hashes) == (mode == "shadow")
    assert task["request"]["forecast_context"]["context"]["inputs"]["task_id"] == task["id"]
    context = task["request"]["forecast_context"]["context"]
    assert context["status"] == "available", context["issues"]
    cap = asyncio.run(stack.center.state.service.tool(task["configs"]["O2"], "station_capability", {}, task["id"], task["id"], "resources", 1))
    assert cap == artifacts["resource"]
    if mode == "simulation_use":
        from station_platform.center import create_center
        from station_platform.tests.conftest import Server
        stack.servers[1].close()
        stack.center = create_center(stack.path, stack.urls["node"], stack.token, stack.ports["center"], interval=.1)
        stack.servers[1] = Server(stack.center, stack.ports["center"])
        stack.client.headers["X-Platform-Token"] = stack.client.get("/api/bootstrap").json()["token"]
        restored = stack.wait_plan(task["id"])
        assert restored["request"] == task["request"]
        assert restored["plan_id"] == task["plan_id"]
        assert len(stack.node.state.service.store.all("plans")) == 1
    stack.client.post(f"/api/tasks/{task['id']}/advance", json={"all": True}).raise_for_status()
    stack.wait(lambda: stack.task(task["id"])["status"] == "COMPLETED")


def test_http_unavailable_forecast_is_blocked_with_no_plan(stack):
    inputs = sample(datetime.now(timezone.utc).replace(microsecond=0).isoformat())
    inputs["commitments"]["status"] = "unknown"
    original, _ = request_and_config()
    response = stack.client.post("/api/tasks", json={"type": "tariff", "demand": original["user_demand"],
        "submission_key": "unavailable-forecast", "forecast": {"mode": "simulation_use", "inputs": inputs}})
    assert response.status_code == 200, response.text
    task_id = response.json()["id"]
    stack.wait(lambda: stack.task(task_id)["status"] in {"BLOCKED", "FAILED"})
    task = stack.task(task_id)
    assert task["status"] == "BLOCKED", task
    assert task["code"] == "FORECAST_CONTEXT_UNAVAILABLE"
    assert not task.get("plan_id")
    assert "service" not in task["artifacts"] and "evaluation" not in task["artifacts"]
    assert task["artifacts"]["resource"]["charging_headroom_kw"] is None
