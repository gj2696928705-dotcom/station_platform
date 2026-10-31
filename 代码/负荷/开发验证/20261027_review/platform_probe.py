"""Stdlib-only offline feasibility probe. Does not install a platform integration."""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

LOAD_ROOT = Path(__file__).resolve().parents[2]
ROOT = LOAD_ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from station_platform.domain import capability, check_local, evaluation, resources, schedule
from station_platform.legacy.demand import MockStationAdapter
from station_platform.util import PlatformError, digest

STEP = timedelta(minutes=15)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None, "timezone is required")
    return result


def number(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "invalid power")
    return value


def validate_native(record):
    require(record["schema_version"] == "station-load-forecast-1" and record["forecast_type"] == "station_load", "schema/type")
    require(record["status"] == "available" and record["decision_eligible"] is False, "unavailable or invalid candidate")
    require(record["publication_status"] == "candidate" and record["execution_mode"] == "offline_replay", "probe requires candidate replay")
    require(record["target_semantics"] == "base_load_excluding_charging_pv_storage", "charging double count risk")
    require(type(record["interval_minutes"]) is int and record["interval_minutes"] == 15, "interval")
    values, bounds = record["forecast_values"], record["prediction_interval"]["values"]
    require(len(values) == len(bounds) == 96 and record["prediction_interval"]["available"] is True, "length/interval")
    start = timestamp(record["forecast_start_at"])
    require(start.minute % 15 == 0 and start.second == start.microsecond == 0, "grid")
    require(timestamp(record["forecast_end_at"]) == start + 96 * STEP, "exclusive end")
    require(record["input_data_quality_summary"]["used_history_end_exclusive"] == record["forecast_start_at"], "history cutoff")
    for i, (point, bound) in enumerate(zip(values, bounds)):
        require(timestamp(point["timestamp"]) == timestamp(bound["timestamp"]) == start + i * STEP, "timeline")
        require(point["unit"] == bound["unit"] == "kW", "unit")
        require(number(bound["lower"]) <= number(point["value"]) <= number(bound["upper"]), "bounds")
    return record


def project(record, start, as_of, station_id, binding=None):
    validate_native(record)
    require(binding == {"source": record["asset_scope"], "target": station_id, "simulation_only": True}, "explicit simulation binding required")
    clock = timestamp(record["as_of"])
    require(clock == as_of and as_of <= start, "replay clock/window mismatch")
    require(timestamp(record["forecast_start_at"]) == as_of.replace(minute=as_of.minute // 15 * 15, second=0, microsecond=0), "stale origin")
    result = {"base_kw": [], "lower_kw": [], "upper_kw": []}
    for i in range(4):
        left, right = start + i * STEP, start + (i + 1) * STEP
        covered, totals = 0., [0., 0., 0.]
        for point, bound in zip(record["forecast_values"], record["prediction_interval"]["values"]):
            t = timestamp(point["timestamp"])
            overlap = max(0., (min(right, t + STEP) - max(left, t)).total_seconds())
            covered += overlap
            for j, value in enumerate((point["value"], bound["lower"], bound["upper"])):
                totals[j] += value * overlap
        require(abs(covered - 900) < 1e-9, "incomplete window coverage")
        for key, total in zip(result, totals):
            result[key].append(total / 900)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    bundle = LOAD_ROOT / "运行结果/20260928_232203_101506"
    source = LOAD_ROOT.parent / "数据/负荷数据/黄花站_2025年_15分钟负荷模拟数据.csv"
    at = datetime(2026, 1, 1, 0, 7, 30, tzinfo=timezone(timedelta(hours=8)))
    checks, calls = {}, []

    def invoke(name, replay=at.isoformat(), model_bundle=bundle, timeout=30):
        target = out / f"{name}.json"
        cmd = [str(args.python), "-X", "utf8", "-m", "load_forecast.cli", "predict", "--input", str(source),
               "--bundle-dir", str(model_bundle), "--as-of", replay, "--output", str(target)]
        begin = time.perf_counter()
        proc = subprocess.run(cmd, cwd=LOAD_ROOT, capture_output=True, encoding="utf-8", timeout=timeout)
        (out / f"{name}.stderr.log").write_text(proc.stderr, encoding="utf-8")
        record = json.loads(proc.stdout)
        require(json.loads(target.read_text(encoding="utf-8")) == record, "stdout/output mismatch")
        calls.append({"case": name, "exit_code": proc.returncode, "status": record["status"],
                      "reason": record["reason"], "elapsed_seconds": time.perf_counter() - begin})
        return proc.returncode, record

    code, record = invoke("cli_replay")
    require(code == 0 and record["status"] == "available", "real CLI replay failed")
    binding = {"source": "HH-Huanghua-CSS", "target": "mock-station-001", "simulation_only": True}
    projected = project(record, at, at, "mock-station-001", binding)
    aligned = project(record, at + timedelta(minutes=7, seconds=30), at, "mock-station-001", binding)
    expected = record["forecast_values"][0]["value"] * .5 + record["forecast_values"][1]["value"] * .5
    require(abs(projected["base_kw"][0] - expected) < 1e-9, "off-grid overlap error")
    require(aligned["base_kw"] == [p["value"] for p in record["forecast_values"][1:5]], "aligned slice error")
    checks["aligned_and_second_resolution_overlap"] = True
    for name in ("binding", "unit", "semantics", "missing_point", "coverage", "clock"):
        bad = deepcopy(record)
        test_start, test_clock, test_binding = at, at, binding
        if name == "binding": test_binding = None
        if name == "unit": bad["forecast_values"][0]["unit"] = "W"
        if name == "semantics": bad["target_semantics"] = "gross_load_including_charging_excluding_pv_storage"
        if name == "missing_point": bad["forecast_values"].pop()
        if name == "coverage": test_start = at + timedelta(hours=23)
        if name == "clock": test_clock = at + timedelta(minutes=15)
        try:
            project(bad, test_start, test_clock, "mock-station-001", test_binding)
        except ValueError:
            checks[f"reject_{name}"] = True
        else:
            raise AssertionError(f"accepted invalid {name}")
    # One-hour task on the unchanged mock grid boundary; demand is a small test order.
    demand = {"vehicles": [{"vehicle_id": "负荷适配验证车", "requested_energy_kwh": 10,
                            "arrival_minutes": 0, "departure_minutes": 15, "priority": "ordinary"}]}
    original = MockStationAdapter().capture(demand, at=at)
    original["station_state_snapshot"]["source_state_id"] = "offline-load-probe-state"
    tagged = deepcopy(original)
    tagged["station_state_snapshot"]["forecasts"] = [record]
    require(resources(original)["charging_headroom_kw"] == resources(tagged)["charging_headroom_kw"] and
            resources(tagged)["forecast_status"] == "not_used", "unexpected forecast consumer")
    checks["adding_forecasts_does_not_change_platform_resources"] = True
    existing_charging_kw = [20., 30., 40., 50.]  # Test assumption, independent of historical CSV charging.
    work = deepcopy(original)
    snap = work["station_state_snapshot"]
    snap["source_state_id"] = "offline-load-probe-state"
    snap["energy"]["base_load_kw"] = [b + c for b, c in zip(projected["base_kw"], existing_charging_kw)]
    # Production binding/persistence is absent. This freezes only an isolated local test request.
    frozen = {"mode": "offline_probe", "forecast_id": record["forecast_id"], "binding": binding,
              "as_of": at.isoformat(), "projection": projected, "existing_charging_kw": existing_charging_kw,
              "platform_base_load_kw": snap["energy"]["base_load_kw"],
              "model_sha256": hashlib.sha256((bundle / "xgboost.json").read_bytes()).hexdigest()}
    snap["load_probe_context"] = frozen
    resource = resources(work)
    state = {"station_id": "mock-station-001", "id": "offline-load-probe-state", "revision": 1,
             "quality": "B", "simulation": True, "measured_at": at.isoformat(),
             "expires_at": (at + timedelta(hours=1)).isoformat(),
             "site": {k: deepcopy(snap[k]) for k in ("energy", "equipment", "safety")}}
    with patch("station_platform.domain.now", return_value=at), patch("station_platform.legacy.computation.now", return_value=at):
        service = schedule(work)
        evaluated = evaluation(work, resource, service)
        cap = capability(state)
        require(cap["charging_headroom_kw"] == resource["charging_headroom_kw"], "capability/resource context mismatch")
        plan = {"id": "offline-probe-plan", "task_id": "offline-probe-task", "version": 1,
                "simulation": True, "station_id": "mock-station-001", "request": work,
                "selected_candidate": evaluated["selected_candidate"], "budget_kw": None}
        plan["fingerprint"] = digest({k: plan[k] for k in ("task_id", "version", "station_id", "request", "selected_candidate", "budget_kw")})
        local = check_local(plan, state)
        require(local["passed"], f"local check failed: {local}")
        state["site"]["energy"]["grid_import_limit_kw"] = 1
        require("GRID_CAPACITY_LIMIT" in check_local(plan, state)["violations"], "live boundary bypassed")
        blocked = deepcopy(work)
        blocked["station_state_snapshot"]["energy"]["grid_import_limit_kw"] = 1
        try:
            schedule(blocked)
        except PlatformError as exc:
            require(exc.code == "CAPACITY_LIMIT", "unexpected planning rejection")
        else:
            raise AssertionError("infeasible grid boundary accepted")
    checks["resource_schedule_evaluation_and_local_check"] = True
    checks["live_and_planning_grid_limits_still_block"] = True
    for name, replay, model_bundle in (("stale", "2026-10-27T12:00:00+08:00", bundle),
                                        ("missing_model", at.isoformat(), out / "absent_bundle")):
        code, bad = invoke(name, replay, model_bundle)
        require(code == 2 and bad["status"] == "unavailable" and bad["forecast_values"] == [], name)
        checks[name] = True
    try:
        invoke("timeout", timeout=.001)
    except subprocess.TimeoutExpired:
        checks["timeout_is_failure"] = True
    else:
        raise AssertionError("expected real process startup timeout")
    saved = out / "frozen_context.json"
    saved.write_text(json.dumps(frozen, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    require(json.loads(saved.read_text(encoding="utf-8")) == frozen, "frozen context roundtrip")
    checks["frozen_context_file_roundtrip"] = True
    report = {"scope": "offline isolated feasibility probe; no live API/UI/task integration",
              "numerical_packages_in_platform": {n: importlib.util.find_spec(n) is not None for n in ("numpy", "pandas", "xgboost")},
              "checks": checks, "cli_calls": calls, "projection": projected,
              "existing_charging_kw": existing_charging_kw, "platform_base_load_kw": snap["energy"]["base_load_kw"],
              "charging_headroom_kw": resource["charging_headroom_kw"], "local_check": local,
              "platform_forecast_status": resource["forecast_status"], "production_integration_implemented": False}
    (out / "platform_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
