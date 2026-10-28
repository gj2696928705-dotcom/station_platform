"""Independent audit of frozen baselines, causal events, technical response and recovery."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CFG = json.loads((HERE / "simulation_config.json").read_text(encoding="utf-8"))
Q = Decimal("0.000001")
TOL = Decimal("0.000003")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(name: str) -> list[dict[str, str]]:
    with (HERE / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (HERE / name).read_text(encoding="utf-8").splitlines() if line]


def dt(s: str) -> datetime:
    out = datetime.fromisoformat(s)
    assert out.utcoffset() == timedelta(hours=8)
    return out


def close(a: Decimal, b: Decimal) -> None:
    assert abs(a - b) <= TOL, f"{a} != {b}"


def main() -> None:
    report = json.loads((HERE / "validation_report.json").read_text(encoding="utf-8"))
    assert report["dataset_id"] == CFG["dataset_id"]
    for name, expected in CFG["source_hashes_sha256"].items():
        assert sha(ROOT / name) == expected
    for name, expected in report["output_hashes_sha256"].items():
        assert sha(HERE / name) == expected
    source_forecasts = {}
    with (ROOT / "代码/数据/场站联动模拟数据/forecast_values.csv").open(encoding="utf-8-sig", newline="") as stream:
        for r in csv.DictReader(stream):
            if r["scenario_id"] == CFG["source_scenario_id"] and r["resource"] == "load":
                source_forecasts[r["timestamp"]] = r
    source_truth = {}
    with (ROOT / "代码/数据/场站联动模拟数据/scenario_truth.csv").open(encoding="utf-8-sig", newline="") as stream:
        for r in csv.DictReader(stream):
            if r["scenario_id"] == CFG["source_scenario_id"]:
                source_truth[r["timestamp"]] = r
    d02_truth = [json.loads(line) for line in (ROOT / "代码/数据/滚动事件模拟数据/execution_truth.jsonl").read_text(encoding="utf-8").splitlines() if line]
    normal_d02 = {r["step_index"]: r for r in d02_truth if r["scenario_id"] == "R01_normal"}
    assert len(source_forecasts) == len(source_truth) == len(normal_d02) == 4
    manifest = json.loads((HERE / "scenario_manifest.json").read_text(encoding="utf-8"))
    manifest_by = {m["scenario_id"]: m for m in manifest}
    assert len(manifest_by) == 10
    baseline = read("baseline_plan.csv")
    by_baseline = defaultdict(list)
    for b in baseline:
        assert b["station_id"] == CFG["station_id"] and b["is_simulated"] == "true"
        assert dt(b["frozen_at"]) == dt(b["as_of"]) == dt(CFG["baseline_frozen_at"])
        assert dt(b["source_observation_end"]) <= dt(b["frozen_at"])
        assert dt(b["interval_end"]) == dt(b["timestamp"]) + timedelta(minutes=15)
        source = source_forecasts[b["timestamp"]]
        assert b["source_observation_end"] == source["source_observation_end"]
        assert b["base_load_forecast_kw"] == source["point_kw"]
        assert b["market_baseline_rule_version"] == "unknown"
        vehicles = json.loads(b["vehicle_charge_kw_json"])
        close(Decimal(b["net_import_kw"]) + Decimal(b["pv_used_kw"]) + Decimal(b["storage_discharge_kw"]),
              Decimal(b["base_load_forecast_kw"]) + Decimal(b["existing_committed_charge_kw"])
              + sum(map(Decimal, vehicles.values())) + Decimal(b["storage_charge_kw"]))
        assert Decimal(b["existing_committed_charge_kw"]) == 80
        assert b["existing_charge_status"] == "synthetic_known_D02_R01_service"
        assert Decimal(b["grid_import_limit_kw"]) == 10000 and Decimal(b["grid_export_limit_kw"]) == 0
        assert Decimal(b["storage_energy_after_kwh"]) == 2400
        by_baseline[b["scenario_id"]].append(b)
    assert len(by_baseline) == 10 and all(len(v) == 4 for v in by_baseline.values())
    for sid, arr in by_baseline.items():
        arr.sort(key=lambda r: int(r["step_index"]))
        frozen_content = []
        for k, b in enumerate(arr):
            assert int(b["step_index"]) == k
            frozen_content.append({"step_index": k, "timestamp": b["timestamp"],
                                   "vehicle_charge_kw": json.loads(b["vehicle_charge_kw_json"]),
                                   "existing_committed_charge_kw": b["existing_committed_charge_kw"],
                                   "storage_charge_kw": b["storage_charge_kw"],
                                   "storage_discharge_kw": b["storage_discharge_kw"],
                                   "storage_energy_after_kwh": b["storage_energy_after_kwh"],
                                   "pv_used_kw": b["pv_used_kw"], "net_import_kw": b["net_import_kw"],
                                   "committed_ac_energy_kwh_each": b["committed_ac_energy_kwh_each"],
                                   "source_load_vintage_id": source_forecasts[b["timestamp"]]["vintage_id"]})
            assert dt(source_truth[b["timestamp"]]["available_at"]) > dt(b["as_of"])
        fp = digest({"scenario_id": sid, "frozen_at": CFG["baseline_frozen_at"], "seed": CFG["seed"],
                     "rule": CFG["technical_fixture_rule_version"], "steps": frozen_content})
        assert len({r["plan_fingerprint"] for r in arr}) == 1 and arr[0]["plan_fingerprint"] == fp
        assert manifest_by[sid]["baseline_fingerprint"] == fp

    offers = read("response_offers.csv")
    offer_by = {o["scenario_id"]: o for o in offers}
    assert len(offer_by) == 10
    for sid, o in offer_by.items():
        assert o["station_id"] == CFG["station_id"] and o["is_simulated"] == "true"
        assert o["plan_fingerprint"] == by_baseline[sid][0]["plan_fingerprint"]
        assert dt(o["offered_at"]) == dt(o["available_at"]) == dt(CFG["baseline_frozen_at"])
        assert dt(o["response_start"]) < dt(o["response_end"]) <= dt(o["recovery_end"])
        assert dt(o["response_end"]) == dt(o["recovery_start"])
        assert o["direction"] in ("up", "down") and o["market_status"] == "not_applicable"
        assert o["baseline_rule_version"] == o["assessment_rule_version"] == o["notice_lead_minutes"] == "unknown"
        duration = Decimal(str((dt(o["response_end"]) - dt(o["response_start"])).total_seconds())) / 3600
        close(Decimal(o["requested_energy_kwh"]), Decimal(o["requested_power_kw"]) * duration)
        assert int(o["duration_minutes"]) == int(duration * 60)
        assert int(o["fixture_notice_lead_minutes"]) <= int((dt(o["response_start"]) - dt(o["offered_at"])).total_seconds() / 60)

    activation = jsonl("activation_events.jsonl")
    by_event = defaultdict(list)
    seen_ids = {}
    duplicates = 0
    d02_ids = {r["event_id"] for line in (ROOT / "代码/数据/滚动事件模拟数据/events.jsonl").read_text(encoding="utf-8").splitlines() if line for r in [json.loads(line)]}
    for e in activation:
        sid = e["scenario_id"]
        assert e["station_id"] == CFG["station_id"] and e["is_simulated"] is True
        assert e["offer_id"] == offer_by[sid]["offer_id"] and e["plan_version"] == 1
        assert dt(e["business_time"]) <= dt(e["available_at"]) <= dt(e["received_at"])
        assert e["available_at"] == e["received_at"]
        assert e["source_ref"].split(":")[-1] in d02_ids
        semantic = {k: v for k, v in e.items() if k not in ("received_at", "available_at")}
        if e["event_id"] in seen_ids:
            assert seen_ids[e["event_id"]] == digest(semantic)
            duplicates += 1
        else:
            seen_ids[e["event_id"]] = digest(semantic)
            by_event[sid].append(e)
    assert duplicates == 1
    assert len(by_event) == 10

    truth = read("execution_truth.csv")
    by_truth = defaultdict(list)
    missing_count = 0
    for r in truth:
        sid = r["scenario_id"]
        assert r["station_id"] == CFG["station_id"] and r["is_simulated"] == "true"
        assert dt(r["available_at"]) > dt(CFG["baseline_frozen_at"])
        assert dt(r["available_at"]) >= dt(r["interval_end"]) and dt(r["received_at"]) >= dt(r["available_at"])
        assert r["plan_fingerprint"] == by_baseline[sid][0]["plan_fingerprint"]
        device = json.loads(r["device_state_json"])
        assert device["storage_site_dispatch_ready"] is False
        assert Decimal(device["grid_import_limit_kw"]) == 10000
        assert device["meter_quality"] == r["measurement_quality"]
        assert r["d02_event_ref"] == manifest_by[sid]["source_d02_event_id"]
        assert r["d02_event_applicability"] == "workflow_pattern_only"
        if sid == "DR03_down_unavailable":
            assert device["actual_device_condition"] == "storage_frozen"
        source = source_truth[r["timestamp"]]
        close(Decimal(r["base_energy_kwh"]), Decimal(source["base_load_kw"]) / 4)
        close(Decimal(r["historical_charge_energy_kwh"]), Decimal(source["historical_charge_kw"]) / 4)
        assert r["pv_available_energy_kwh_approx"] == source["pv_interval_energy_kwh_approx"]
        k = int((dt(r["timestamp"]) - dt(CFG["baseline_frozen_at"])) / timedelta(minutes=15))
        assert Decimal(normal_d02[k]["actual"]["base_energy_kwh"]) == Decimal(r["base_energy_kwh"])
        powers = json.loads(r["vehicle_charge_kw_json"])
        energies = json.loads(r["vehicle_ac_energy_kwh_json"])
        assert set(powers) == set(energies) == set(CFG["technical_assumptions"]["vehicle_ids"])
        for vid, power in powers.items():
            assert 0 <= Decimal(power) <= 120
            close(Decimal(energies[vid]), Decimal(power) / 4)
        assert Decimal(r["pv_used_energy_kwh"]) <= Decimal(r["pv_available_energy_kwh_approx"])
        assert Decimal(r["storage_charge_energy_ac_kwh"]) == Decimal(r["storage_discharge_energy_ac_kwh"]) == 0
        assert Decimal(r["storage_energy_after_kwh"]) == 2400 and Decimal(r["soc_after_percent"]) == 60
        assert Decimal(r["existing_committed_charge_energy_kwh"]) == sum(map(Decimal, normal_d02[k]["actual"]["vehicle_energy_kwh"].values())) == 20
        expected_grid = Decimal(r["base_energy_kwh"]) + Decimal(r["historical_charge_energy_kwh"]) + Decimal(r["existing_committed_charge_energy_kwh"]) + sum(map(Decimal, energies.values()))
        if r["measurement_quality"] == "unknown":
            assert r["grid_import_energy_kwh"] == ""
            missing_count += 1
        else:
            assert r["measurement_quality"] == "good"
            close(Decimal(r["grid_import_energy_kwh"]) + Decimal(r["pv_used_energy_kwh"]) + Decimal(r["storage_discharge_energy_ac_kwh"]),
                  expected_grid + Decimal(r["storage_charge_energy_ac_kwh"]) + Decimal(r["grid_export_energy_kwh"]))
            assert 0 <= Decimal(r["grid_import_energy_kwh"]) <= 2500
        by_truth[sid].append(r)
    assert len(truth) == 40 and missing_count == 1
    assert all(len(v) == 4 for v in by_truth.values())

    mutations = jsonl("baseline_mutation_attempts.jsonl")
    assert len(mutations) == 1
    mutation = mutations[0]
    assert mutation["scenario_id"] == "DR08_baseline_tamper"
    assert mutation["baseline_fingerprint"] == by_baseline[mutation["scenario_id"]][0]["plan_fingerprint"]
    assert mutation["attempted_fingerprint"] != mutation["baseline_fingerprint"]
    checks = []
    for sid, m in manifest_by.items():
        ev = sorted(by_event[sid], key=lambda r: r["received_at"])
        o = offer_by[sid]
        b = by_baseline[sid]
        a = sorted(by_truth[sid], key=lambda r: r["timestamp"])
        kinds = {e["event_type"] for e in ev}
        ack = next((e for e in ev if e["event_type"] == "activation_acknowledged"), None)
        if "cancelled" in kinds:
            assert dt(next(e for e in ev if e["event_type"] == "cancelled")["received_at"]) < dt(o["response_start"])
            actual_status = "cancelled_before_start"
        elif ack is not None and ack["plan_fingerprint"] != b[0]["plan_fingerprint"]:
            actual_status = "reject_baseline_fingerprint"
        elif ack is not None and dt(ack["received_at"]) > dt(o["response_start"]):
            actual_status = "reject_late_ack"
        elif ack is not None and ack["payload"]["accepted"] is False:
            actual_status = ack["payload"]["reason"]
        elif "activation_started" in kinds:
            started = next(e for e in ev if e["event_type"] == "activation_started")
            actual_status = "evidence_incomplete" if any(r["measurement_quality"] != "good" for r in a) else "partial_technical" if dt(started["business_time"]) > dt(o["response_start"]) else "activated_technical"
        else:
            raise AssertionError(sid)
        assert actual_status == m["expected_status"], (sid, actual_status, m["expected_status"])
        base_energy = defaultdict(Decimal)
        actual_energy = defaultdict(Decimal)
        for x, y in zip(b, a):
            assert x["timestamp"] == y["timestamp"]
            for vid, power in json.loads(x["vehicle_charge_kw_json"]).items():
                base_energy[vid] += Decimal(power) / 4
            for vid, energy in json.loads(y["vehicle_ac_energy_kwh_json"]).items():
                actual_energy[vid] += Decimal(energy)
        for vid in base_energy:
            close(actual_energy[vid], base_energy[vid])
            close(base_energy[vid], Decimal(b[0]["committed_ac_energy_kwh_each"]))
        response_indices = [i for i, row in enumerate(b) if dt(o["response_start"]) <= dt(row["timestamp"]) < dt(o["response_end"])]
        baseline_response = sum((sum(map(Decimal, json.loads(b[i]["vehicle_charge_kw_json"]).values())) / 4 for i in response_indices), Decimal(0))
        actual_response = sum((sum(map(Decimal, json.loads(a[i]["vehicle_ac_energy_kwh_json"]).values())) for i in response_indices), Decimal(0))
        delta = baseline_response - actual_response
        if actual_status in ("activated_technical", "evidence_incomplete"):
            close(delta, Decimal(o["requested_energy_kwh"]))
        elif actual_status == "partial_technical":
            close(delta, Decimal(o["requested_energy_kwh"]) / 2)
        else:
            close(delta, Decimal(0))
        if sid == "DR02_up_unavailable":
            headroom = sum((Decimal(120) - Decimal(v) for v in json.loads(b[1]["vehicle_charge_kw_json"]).values()), Decimal(0))
            assert Decimal(o["requested_power_kw"]) > headroom
        if sid == "DR03_down_unavailable":
            assert sum(map(Decimal, json.loads(b[1]["vehicle_charge_kw_json"]).values())) == 0
            assert json.loads(o["authorized_resources_json"]) == []
        if sid == "DR04_sustained_service":
            assert dt(o["response_end"]) == dt(o["recovery_end"])
            max_delivered_if_reduced = sum(map(Decimal, json.loads(b[0]["vehicle_charge_kw_json"]).values())) / 4
            assert max_delivered_if_reduced < sum(base_energy.values())
        if sid == "DR05_recovery_rebound":
            needed = sum(map(Decimal, json.loads(b[3]["vehicle_charge_kw_json"]).values())) + Decimal(80)
            assert needed > Decimal(o["recovery_fleet_power_cap_kw"])
        checks.append({"scenario_id": sid, "expected_status": m["expected_status"],
                       "actual_status": actual_status, "difference": "",
                       "response_energy_reduction_kwh": str(delta.quantize(Q)),
                       "whole_window_vehicle_energy_conserved": True,
                       "measured_grid_complete": all(r["measurement_quality"] == "good" for r in a)})
    settlement = json.loads((HERE / "settlement_reference.json").read_text(encoding="utf-8"))
    assert settlement["status"] == "not_applicable" and settlement["payment_amount"] is None
    assert settlement["performance_passed"] is None and settlement["method"] == "unknown"
    rules = json.loads((HERE / "rule_sources.json").read_text(encoding="utf-8"))
    assert rules["market_rule_status"] == "unknown"
    report["status"] = "passed"
    report["independent_validation"] = {"frozen_baseline_fingerprints_recomputed": True,
                                        "past_only_forecasts": True, "future_truth_hidden_at_as_of": True,
                                        "response_and_recovery_energy_conserved": True,
                                        "missing_meter_not_zero": True, "device_states_and_D02_references_checked": True,
                                        "duplicate_activation_events": duplicates,
                                        "technical_scenario_checks": checks,
                                        "settlement_generated": False}
    (HERE / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "scenarios": len(manifest), "truth_intervals": len(truth),
                      "activation_events": len(activation), "missing_meter_rows": missing_count}))


if __name__ == "__main__":
    main()
