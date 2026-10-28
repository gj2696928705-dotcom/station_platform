"""Independent source, energy, schedule, aggregation and collaboration audit."""
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
STEP = timedelta(minutes=15)
TOL = Decimal("0.000003")


def read(name: str) -> list[dict[str, str]]:
    with (HERE / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def dt(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    assert result.utcoffset() == timedelta(hours=8)
    return result


def close(a: Decimal, b: Decimal) -> None:
    assert abs(a - b) <= TOL, f"{a} != {b}"


def main() -> None:
    report = json.loads((HERE / "validation_report.json").read_text(encoding="utf-8"))
    for name, expected in CFG["source_hashes_sha256"].items():
        assert sha(ROOT / name) == expected
    for name, expected in report["output_hashes_sha256"].items():
        assert sha(HERE / name) == expected
    root_truth = {}
    with (ROOT / "代码/数据/场站联动模拟数据/station_interval_truth.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            root_truth[row["timestamp"]] = row
    registry = set()
    with (ROOT / "代码/数据/场站配置与设备台账/设备台账.csv").open(encoding="utf-8-sig", newline="") as stream:
        registry = {r["port_id"] for r in csv.DictReader(stream) if r["port_id"]}
    mapping = read("identity_mapping.csv")
    assert len(registry) == 128 and len(mapping) == 4
    mapped = {r["station_port_id"] for r in mapping}
    assert len(mapped) == 4 and mapped <= registry
    assert all(r["station_id"] == CFG["station_id"] and r["fleet_station_id"] == CFG["station_id"]
               and r["excluded_example_station_id"] != CFG["station_id"] for r in mapping)
    manifest = json.loads((HERE / "scenario_manifest.json").read_text(encoding="utf-8"))
    expected_status = {m["scenario_id"]: m["expected_status"] for m in manifest}
    assert len(expected_status) == 5
    day = dt(CFG["business_date"] + "T00:00:00+08:00")
    station_capacity = read("station_capacity_15m.csv")
    capacity_by = defaultdict(list)
    for r in station_capacity:
        assert r["is_simulated"] == "true" and r["station_id"] == CFG["station_id"]
        assert r["scenario_id"] in expected_status and dt(r["available_at"]) <= dt(r["as_of"]) == day
        slot = int((dt(r["timestamp"]) - day) / STEP)
        assert 0 <= slot < 96 and dt(r["interval_end"]) == dt(r["timestamp"]) + STEP
        load_source = root_truth[(dt(r["timestamp"]) - timedelta(days=7)).isoformat(timespec="seconds")]
        pv_source = root_truth[(dt(r["timestamp"]) - timedelta(days=1)).isoformat(timespec="seconds")]
        assert r["load_source_end"] == load_source["available_at"]
        assert r["pv_source_end"] == pv_source["available_at"]
        assert dt(r["load_source_end"]) <= day and dt(r["pv_source_end"]) <= day
        close(Decimal(r["base_load_forecast_kw"]), Decimal(load_source["base_load_kw"]))
        close(Decimal(r["pv_available_forecast_kw"]), Decimal(pv_source["pv_interval_mean_kw_approx"]))
        assert Decimal(r["existing_committed_charge_kw"]) == (80 if 40 <= slot < 44 else 0)
        assert Decimal(r["grid_import_limit_kw"]) == 10000 and Decimal(r["grid_export_limit_kw"]) == 0
        assert int(r["station_total_ports"]) == 128 and int(r["fleet_mapped_ports"]) == 4 and int(r["unmapped_ports"]) == 124
        occupied = json.loads(r["occupied_port_ids_json"])
        expected_occupied = [CFG["fleet_assumptions"]["existing_service_port_id"]] if 40 <= slot < 44 else []
        assert occupied == expected_occupied and not set(occupied) & mapped
        assert all(port in registry for port in occupied)
        assert r["existing_service_port_id"] == (expected_occupied[0] if occupied else "")
        assert int(r["other_ports_free_count"]) == 124 - len(occupied)
        assert r["storage_site_dispatch_ready"] == "false"
        capacity_by[r["scenario_id"]].append(r)
    assert all(len(v) == 96 for v in capacity_by.values())
    for arr in capacity_by.values():
        arr.sort(key=lambda r: r["timestamp"])
        assert [dt(r["timestamp"]) for r in arr] == [day + i * STEP for i in range(96)]
        assert len({r["forecast_input_sha256"] for r in arr}) == 1
        observations = [(r["load_source_end"], r["pv_source_end"]) for r in arr]
        values = [(str(Decimal(r["base_load_forecast_kw"])), str(Decimal(r["pv_available_forecast_kw"]))) for r in arr]
        assert arr[0]["forecast_input_sha256"] == digest(observations + values)

    vehicles = read("vehicles.csv")
    vehicle_map = {(v["scenario_id"], v["vehicle_id"]): v for v in vehicles}
    assert len(vehicle_map) == len(vehicles)
    requests = read("charging_requests.csv")
    req_by = defaultdict(list)
    for r in requests:
        assert (r["scenario_id"], r["vehicle_id"]) in vehicle_map
        assert r["station_id"] == r["station_id_ref"] == CFG["station_id"]
        assert dt(r["available_at"]) <= day and dt(r["arrival_at"]) < dt(r["departure_deadline_at"])
        assert Decimal(r["min_ac_energy_kwh"]) <= Decimal(r["target_ac_energy_kwh"])
        req_by[r["scenario_id"]].append(r)
    tasks = read("transport_tasks.csv")
    for r in tasks:
        assert (r["scenario_id"], r["vehicle_id"]) in vehicle_map
        assert dt(r["load_start"]) < dt(r["load_end"]) == dt(r["travel_start"]) < dt(r["travel_end"]) < dt(r["unload_end"]) <= dt(r["deadline"])
        assert r["mandatory"] == "true" and dt(r["available_at"]) <= day
    allocations = read("allocation_15m.csv")
    alloc_by_slot = defaultdict(list)
    energy_by_vehicle = defaultdict(Decimal)
    last_charge_end = {}
    for a in allocations:
        assert a["port_id"] in mapped and (a["scenario_id"], a["vehicle_id"]) in vehicle_map
        assert dt(a["interval_end"]) == dt(a["timestamp"]) + STEP
        power = Decimal(a["charge_power_kw"])
        assert 0 < power <= 120 and power <= 240
        close(Decimal(a["ac_energy_kwh"]), power / 4)
        close(Decimal(a["battery_energy_gain_kwh"]), power * Decimal("0.95") / 4)
        key = (a["scenario_id"], a["timestamp"])
        alloc_by_slot[key].append(a)
        energy_by_vehicle[(a["scenario_id"], a["vehicle_id"])] += Decimal(a["ac_energy_kwh"])
        last_charge_end[(a["scenario_id"], a["vehicle_id"])] = max(last_charge_end.get((a["scenario_id"], a["vehicle_id"]), dt(a["interval_end"])), dt(a["interval_end"]))
        request = next(r for r in req_by[a["scenario_id"]] if r["vehicle_id"] == a["vehicle_id"])
        assert dt(request["arrival_at"]) <= dt(a["timestamp"]) and dt(a["interval_end"]) <= dt(request["departure_deadline_at"])
    for key, arr in alloc_by_slot.items():
        assert len({a["vehicle_id"] for a in arr}) == len(arr)
        assert len({a["port_id"] for a in arr}) == len(arr)
        assert len(arr) <= 4

    service = read("station_service_15m.csv")
    service_by = defaultdict(list)
    for s in service:
        sid = s["scenario_id"]
        cap = next(c for c in capacity_by[sid] if c["timestamp"] == s["timestamp"])
        alloc = alloc_by_slot[(sid, s["timestamp"])]
        by_vehicle = json.loads(s["vehicle_charge_kw_json"])
        assert set(by_vehicle) == {a["vehicle_id"] for a in alloc}
        for a in alloc:
            assert Decimal(by_vehicle[a["vehicle_id"]]) == Decimal(a["charge_power_kw"])
        new = sum(map(Decimal, by_vehicle.values()), Decimal(0))
        close(Decimal(s["fleet_charge_kw"]), new)
        assert new <= Decimal(cap["fleet_allocated_capacity_kw"])
        for key1, key2 in (("base_load_kw", "base_load_forecast_kw"),
                           ("pv_available_kw", "pv_available_forecast_kw"),
                           ("existing_charge_kw", "existing_committed_charge_kw")):
            assert s[key1] == cap[key2]
        assert Decimal(0) <= Decimal(s["pv_used_kw"]) <= Decimal(s["pv_available_kw"])
        close(Decimal(s["pv_used_kw"]) + Decimal(s["pv_curtailed_kw"]), Decimal(s["pv_available_kw"]))
        close(Decimal(s["grid_import_kw"]) + Decimal(s["pv_used_kw"]) + Decimal(s["storage_discharge_kw"]),
              Decimal(s["base_load_kw"]) + Decimal(s["existing_charge_kw"]) + new + Decimal(s["storage_charge_kw"]) + Decimal(s["grid_export_kw"]))
        assert 0 <= Decimal(s["grid_import_kw"]) <= 10000 and Decimal(s["grid_export_kw"]) == 0
        assert Decimal(s["storage_charge_kw"]) == Decimal(s["storage_discharge_kw"]) == 0
        assert Decimal(s["storage_energy_after_kwh"]) == 2400
        service_by[sid].append(s)
    assert set(service_by) == {"F01_feasible", "F04_cancel_reassign", "F05_pressure"}
    assert all(len(v) == 96 for v in service_by.values())
    for sid, arr in service_by.items():
        arr.sort(key=lambda r: r["timestamp"])
        assert [dt(r["timestamp"]) for r in arr] == [day + i * STEP for i in range(96)]
        for r in req_by[sid]:
            assert energy_by_vehicle[(sid, r["vehicle_id"])] >= Decimal(r["min_ac_energy_kwh"])
        for task in (t for t in tasks if t["scenario_id"] == sid):
            key = (sid, task["vehicle_id"])
            assert last_charge_end[key] <= dt(task["load_start"])
            v = vehicle_map[key]
            battery_after_transport = Decimal(v["initial_energy_kwh"]) + Decimal("0.95") * energy_by_vehicle[key] - Decimal(task["travel_energy_kwh"])
            assert Decimal(v["reserve_energy_kwh"]) <= battery_after_transport <= Decimal(v["battery_kwh"])
    trace = read("vehicle_energy_trace_15m.csv")
    trace_by_vehicle = defaultdict(list)
    task_lookup = {(t["scenario_id"], t["vehicle_id"]): t for t in tasks}
    allocation_lookup = {(a["scenario_id"], a["vehicle_id"], a["timestamp"]): a for a in allocations}
    for r in trace:
        key = (r["scenario_id"], r["vehicle_id"])
        assert key in vehicle_map and r["is_simulated"] == "true"
        assert dt(r["available_at"]) == dt(r["interval_end"]) == dt(r["timestamp"]) + STEP
        allocation = allocation_lookup.get((key[0], key[1], r["timestamp"]))
        task = task_lookup.get(key)
        charge = Decimal(allocation["battery_energy_gain_kwh"]) if allocation else Decimal(0)
        travel = Decimal(task["travel_energy_kwh"]) if task and task["travel_end"] == r["interval_end"] else Decimal(0)
        close(Decimal(r["charge_gain_kwh"]), charge)
        close(Decimal(r["transport_energy_kwh"]), travel)
        close(Decimal(r["energy_after_kwh"]), Decimal(r["energy_before_kwh"]) + charge - travel)
        close(Decimal(r["soc_after_percent"]), Decimal(r["energy_after_kwh"]) / 4)
        assert Decimal(vehicle_map[key]["reserve_energy_kwh"]) <= Decimal(r["energy_after_kwh"]) <= Decimal(vehicle_map[key]["battery_kwh"])
        assert r["charge_port_id"] == (allocation["port_id"] if allocation else "")
        assert r["transport_task_id"] == (task["task_id"] if travel else "")
        trace_by_vehicle[key].append(r)
    assert len(trace) == 96 * 12
    for key, arr in trace_by_vehicle.items():
        arr.sort(key=lambda r: r["timestamp"])
        assert len(arr) == 96 and dt(arr[0]["timestamp"]) == day
        close(Decimal(arr[0]["energy_before_kwh"]), Decimal(vehicle_map[key]["initial_energy_kwh"]))
        for previous, current in zip(arr, arr[1:]):
            assert previous["interval_end"] == current["timestamp"]
            close(Decimal(previous["energy_after_kwh"]), Decimal(current["energy_before_kwh"]))
    # Infeasible cases have explicit lower-bound proofs, not fabricated service plans.
    assert not service_by["F02_station_short"] and not service_by["F03_deadline_conflict"]
    assert sum(Decimal(r["min_ac_energy_kwh"]) for r in req_by["F02_station_short"]) > Decimal(60) * 1
    deadline_req = req_by["F03_deadline_conflict"][0]
    hours = Decimal(str((dt(deadline_req["departure_deadline_at"]) - dt(deadline_req["arrival_at"])).total_seconds())) / 3600
    assert Decimal(deadline_req["min_ac_energy_kwh"]) > Decimal(120) * hours

    aggregates = read("fleet_aggregates_30_60m.csv")
    assert len(aggregates) == 3 * (48 + 24)
    for a in aggregates:
        sid = a["scenario_id"]
        resolution = int(a["resolution_minutes"])
        assert resolution in (30, 60)
        block = [r for r in service_by[sid] if dt(a["timestamp"]) <= dt(r["timestamp"]) < dt(a["interval_end"])]
        assert len(block) == resolution // 15
        expected_grid = sum((Decimal(r["grid_import_kw"]) / 4 for r in block), Decimal(0))
        expected_pv = sum((Decimal(r["pv_used_kw"]) / 4 for r in block), Decimal(0))
        close(Decimal(a["grid_energy_kwh"]), expected_grid)
        close(Decimal(a["grid_mean_kw"]), expected_grid * 60 / resolution)
        close(Decimal(a["pv_used_energy_kwh"]), expected_pv)
        computed = defaultdict(Decimal)
        for r in block:
            for vid, power in json.loads(r["vehicle_charge_kw_json"]).items():
                computed[vid] += Decimal(power) / 4
        stated = json.loads(a["vehicle_ac_energy_kwh_json"])
        assert set(computed) == set(stated)
        for vid in computed:
            close(Decimal(stated[vid]), computed[vid])

    messages = [json.loads(line) for line in (HERE / "collaboration_messages.jsonl").read_text(encoding="utf-8").splitlines() if line]
    seen = {}
    duplicate_count = conflict_count = 0
    by_message_scenario = defaultdict(list)
    for m in messages:
        assert m["is_simulated"] is True and m["source_version"]
        assert m["kind"] in ("adjustment", "coordination_result", "grid", "reservation", "execution", "cancel_ack")
        assert m["source"] in ("coordinator", "grid", "S1") and m["evidence"]["simulation"] is True
        assert dt(m["issued_at"]) <= dt(m["received_at"]) <= dt(m["expires_at"])
        assert m["round_id"] == m["version"] and len(m["plan_fingerprint"]) == 64
        for contract_field in ("passed", "reason", "capacity_kw", "service", "equilibrium", "status", "confirmed"):
            if contract_field in m["payload"]:
                assert m[contract_field] == m["payload"][contract_field]
        semantic = {k: v for k, v in m.items() if k != "received_at"}
        key = (m["scenario_id"], m["message_id"])
        h = digest(semantic)
        if key in seen:
            if seen[key] == h:
                duplicate_count += 1
            else:
                conflict_count += 1
            continue
        seen[key] = h
        by_message_scenario[m["scenario_id"]].append(m)
        if m["kind"] == "coordination_result":
            assert "service" in m and "equilibrium" in m and "passed" in m
            assert m["payload"]["equilibrium"]["passed"] is False
            assert m["payload"]["passed"] is False
        if m["kind"] in ("grid", "reservation"):
            assert "passed" in m
            if m["kind"] == "grid":
                assert "reason" in m
            assert m["payload"]["passed"] is True
    assert duplicate_count == conflict_count == 1
    cancel = sorted(by_message_scenario["F04_cancel_reassign"], key=lambda m: m["issued_at"])
    old_res = next(m for m in cancel if m["kind"] == "reservation" and m["version"] == 1)
    ack = next(m for m in cancel if m["kind"] == "cancel_ack")
    new_res = next(m for m in cancel if m["kind"] == "reservation" and m["version"] == 2)
    assert dt(old_res["issued_at"]) < dt(ack["issued_at"]) < dt(new_res["issued_at"])
    assert ack["plan_fingerprint"] == old_res["plan_fingerprint"] != new_res["plan_fingerprint"]
    assert ack["payload"]["confirmed"] is True
    for sid in service_by:
        own = service_by[sid]
        if not own:
            continue
        vids = [v["vehicle_id"] for v in vehicles if v["scenario_id"] == sid]
        supplied = {"charging": {vid: [json.loads(r["vehicle_charge_kw_json"]).get(vid, "0.000000") for r in own] for vid in vids},
                    "capacity_kw": [r["fleet_capacity_kw"] for r in own],
                    "storage_charge_kw": [r["storage_charge_kw"] for r in own],
                    "storage_discharge_kw": [r["storage_discharge_kw"] for r in own],
                    "storage_energy_kwh": [r["storage_energy_after_kwh"] for r in own],
                    "net_import_kw": [r["grid_import_kw"] for r in own],
                    "pv_used_kw": [r["pv_used_kw"] for r in own], "feasible": True}
        version = 2 if sid == "F04_cancel_reassign" else 1
        expected_fp = digest({"scenario_id": sid, "version": version, "service": supplied,
                              "source_input_hash": capacity_by[sid][0]["forecast_input_sha256"]})
        for m in by_message_scenario[sid]:
            if m["version"] == version and m["kind"] != "cancel_ack":
                assert m["plan_fingerprint"] == expected_fp, (sid, m["kind"], m["plan_fingerprint"], expected_fp)

    history = read("history_uncontracted.csv")
    visible = read("history_uncontracted_visible.csv")
    assert len(history) == 61 * 4
    cutoff = dt(CFG["history_assumptions"]["visible_as_of"])
    hist_map = {(r["timestamp"], r["station_id"]): r for r in history}
    for r in history:
        assert r["station_id"] == CFG["station_id"] and r["is_simulated"] == "true"
        status = r["label_status"]
        if status in ("missing", "signed_excluded"):
            assert r["uncontracted_energy_kwh"] == ""
        else:
            assert status in ("observed_zero", "observed_uncontracted")
            source = root_truth[r["timestamp"]]
            close(Decimal(r["uncontracted_energy_kwh"]), Decimal(source["historical_charge_kw"]) * Decimal("0.1") / 4)
        if r["signed_at"] and dt(r["signed_at"]) <= dt(r["timestamp"]):
            assert status in ("signed_excluded", "missing")
        date = r["business_date"]
        assert r["split"] == ("train" if date < "2025-09-01" else "calibration" if date < "2025-09-15" else "holdout_replay")
    for r in visible:
        original = hist_map[(r["timestamp"], r["station_id"])]
        assert dt(r["label_available_at"]) <= cutoff and r["as_of"] == cutoff.isoformat(timespec="seconds")
        assert original["split"] != "holdout_replay"
        if original["signed_at"] and dt(original["signed_at"]) > cutoff:
            assert r["signed_at"] == "" and r["future_signing_redacted"] == "true"
    assert len(visible) == sum(dt(r["label_available_at"]) <= cutoff for r in history)
    # Fixed previous-slot historical-mean baseline; report a time-out replay metric only.
    means = {}
    for slot in range(40, 44):
        train = [Decimal(r["uncontracted_energy_kwh"]) for r in history if r["split"] == "train" and dt(r["timestamp"]).hour * 4 + dt(r["timestamp"]).minute // 15 == slot and r["label_status"].startswith("observed")]
        assert train
        means[slot] = sum(train) / len(train)
    holdout = [r for r in history if r["split"] == "holdout_replay" and r["label_status"].startswith("observed")]
    mae = sum((abs(Decimal(r["uncontracted_energy_kwh"]) - means[dt(r["timestamp"]).hour * 4 + dt(r["timestamp"]).minute // 15]) for r in holdout), Decimal(0)) / len(holdout)
    report["status"] = "passed"
    report["independent_validation"] = {"full_day_15min_grid": True, "source_forecasts_past_only": True,
                                        "vehicle_port_energy_and_transport": True,
                                        "vehicle_battery_15min_trace_recomputed": len(trace),
                                        "station_grid_pv_storage_balance": True,
                                        "aggregation_30_60_energy_conserved": True,
                                        "infeasible_cases_rejected": ["F02_station_short", "F03_deadline_conflict"],
                                        "message_duplicates": duplicate_count, "message_conflicts_rejected": conflict_count,
                                        "cancel_ack_before_new_reservation": True,
                                        "history_future_signing_redacted": True,
                                        "history_holdout_replay": {"train_slot_mean_only": True,
                                                                    "evaluated_intervals": len(holdout),
                                                                    "mae_kwh_per_15min": str(mae.quantize(Decimal("0.000001"))),
                                                                    "blind_test_claim": False}}
    (HERE / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "scenarios": len(manifest), "service_rows": len(service),
                      "allocations": len(allocations), "history_rows": len(history)}))


if __name__ == "__main__":
    main()
