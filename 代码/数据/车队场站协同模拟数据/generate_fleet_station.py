"""Generate a small auditable Huanghua fleet/station fixture and a pressure case."""
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
STATION = CFG["station_id"]
DATASET = CFG["dataset_id"]
DAY = datetime.fromisoformat(CFG["business_date"] + "T00:00:00+08:00")
STEP = timedelta(minutes=15)
Q = Decimal("0.000001")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_out(name: str, rows: list[dict], columns: list[str] | None = None) -> None:
    with (HERE / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]), lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_out(name: str, value: object) -> None:
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonl_out(name: str, rows: list[dict]) -> None:
    (HERE / name).write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def fmt(value: Decimal | int | float) -> str:
    return str(Decimal(str(value)).quantize(Q))


def time(slot: int) -> str:
    return (DAY + slot * STEP).isoformat(timespec="seconds")


def tariff(slot: int) -> Decimal:
    h = (slot * 15) // 60
    return Decimal("0.4") if h < 7 or h >= 23 else Decimal("0.8") if h < 17 else Decimal("1.2")


def main() -> None:
    assert all(sha(ROOT / name) == h for name, h in CFG["source_hashes_sha256"].items())
    truth = read(ROOT / "代码/数据/场站联动模拟数据/station_interval_truth.csv")
    truth_map = {r["timestamp"]: r for r in truth}
    assets = read(ROOT / "代码/数据/场站配置与设备台账/设备台账.csv")
    registry_ports = {r["port_id"] for r in assets if r["port_id"]}
    ports = CFG["fleet_assumptions"]["port_subset"]
    existing_port = CFG["fleet_assumptions"]["existing_service_port_id"]
    assert len(registry_ports) == 128 and set(ports) <= registry_ports
    assert existing_port in registry_ports and existing_port not in ports
    storage = read(ROOT / "代码/数据/储能参数与初始状态/储能参数.csv")[0]
    assert storage["site_dispatch_ready"] == "false" and Decimal(storage["usable_capacity_kwh"]) == 4000
    d01_tariff = next(r for r in read(ROOT / "代码/数据/场站联动模拟数据/tariff_intervals.csv") if r["scenario_id"] == "S06_calibration")
    assert Decimal(d01_tariff["import_tariff_cny_per_kwh"]) == tariff(40)

    capacities = {"F01_feasible": 240, "F02_station_short": 60, "F03_deadline_conflict": 120,
                  "F04_cancel_reassign": 120, "F05_pressure": 480}
    expected = {"F01_feasible": "feasible", "F02_station_short": "reject_station_allocation",
                "F03_deadline_conflict": "reject_vehicle_deadline", "F04_cancel_reassign": "feasible_after_cancel_ack",
                "F05_pressure": "feasible"}
    # Rows are a separately generated full-day forecast, using only observations ended by day start.
    base_curve = []
    pv_curve = []
    source_observation_ends = []
    for slot in range(96):
        target = DAY + slot * STEP
        load_source = truth_map[(target - timedelta(days=7)).isoformat(timespec="seconds")]
        pv_source = truth_map[(target - timedelta(days=1)).isoformat(timespec="seconds")]
        assert datetime.fromisoformat(load_source["available_at"]) <= DAY
        assert datetime.fromisoformat(pv_source["available_at"]) <= DAY
        base_curve.append(Decimal(load_source["base_load_kw"]))
        pv_curve.append(Decimal(pv_source["pv_interval_mean_kw_approx"]))
        source_observation_ends.append((load_source["available_at"], pv_source["available_at"]))
    base_input_hash = digest(source_observation_ends + [(str(x), str(y)) for x, y in zip(base_curve, pv_curve)])

    id_map = [{"station_id": STATION, "dataset_id": DATASET, "scenario_id": "ALL",
               "is_simulated": "true", "fleet_station_id": STATION,
               "excluded_example_station_id": "示例充电站", "station_port_id": p,
               "fleet_port_id": f"FLEET-PORT-{i+1:02d}", "station_port_rated_kw": "240.000000",
               "fleet_port_limit_kw": "120.000000", "connector": "DC",
               "meter_boundary": "AC charging branch", "source_version": "registry_v1+new_fleet_subset_v1"}
              for i, p in enumerate(ports)]
    csv_out("identity_mapping.csv", id_map)
    vehicles: list[dict] = []
    tasks: list[dict] = []
    requests: list[dict] = []
    capacity_rows: list[dict] = []
    allocations: list[dict] = []
    service_rows: list[dict] = []
    aggregates: list[dict] = []
    messages: list[dict] = []
    manifests: list[dict] = []
    for sid, cap in capacities.items():
        count = 8 if sid == "F05_pressure" else 1 if sid == "F03_deadline_conflict" else 2
        ids = [f"{sid}-EV-{i+1:02d}" for i in range(count)]
        for i, vid in enumerate(ids):
            initial = 150 if sid == "F05_pressure" else 200
            vehicles.append({"station_id": STATION, "scenario_id": sid, "dataset_id": DATASET,
                             "is_simulated": "true", "source_version": "new_fleet_subset_v1", "vehicle_id": vid,
                             "battery_kwh": "400.000000", "initial_energy_kwh": fmt(initial),
                             "initial_soc_percent": fmt(Decimal(initial) / 4), "reserve_energy_kwh": "80.000000",
                             "max_charge_kw": "120.000000", "max_discharge_kw": "0.000000",
                             "charge_efficiency": "0.950000", "connector": "DC", "v2g_authorized": "false",
                             "available_from": time(39), "available_until": time(52), "available_at": time(0)})
            if sid != "F04_cancel_reassign" or i == 1:
                task_start = 42 if sid == "F03_deadline_conflict" else 45
                travel_kwh = 60 if sid == "F05_pressure" else 50
                tasks.append({"station_id": STATION, "scenario_id": sid, "dataset_id": DATASET,
                              "is_simulated": "true", "source_version": "synthetic_transport_v1",
                              "task_id": f"{sid}-TASK-{i+1:02d}", "vehicle_id": vid,
                              "load_start": time(task_start), "load_end": time(task_start + 1),
                              "travel_start": time(task_start + 1), "travel_end": time(task_start + 4),
                              "unload_end": time(task_start + 5), "deadline": time(task_start + 6),
                              "travel_energy_kwh": fmt(travel_kwh), "mandatory": "true", "available_at": time(0)})
        if sid == "F03_deadline_conflict":
            req_spec = [(0, 40, 42, 80)]
        elif sid == "F04_cancel_reassign":
            req_spec = [(1, 42, 44, 60)]
        elif sid == "F05_pressure":
            req_spec = [(i, 40 if i < 4 else 42, 42 if i < 4 else 44, 60) for i in range(8)]
        else:
            req_spec = [(i, 40, 44, 40) for i in range(2)]
        for i, a, b, energy in req_spec:
            requests.append({"station_id": STATION, "scenario_id": sid, "dataset_id": DATASET,
                             "is_simulated": "true", "source_version": "synthetic_request_v1",
                             "request_id": f"{sid}-REQ-{i+1:02d}", "vehicle_id": ids[i],
                             "arrival_at": time(a), "departure_deadline_at": time(b),
                             "min_ac_energy_kwh": fmt(energy), "target_ac_energy_kwh": fmt(energy),
                             "priority": "mandatory", "available_at": time(0), "station_id_ref": STATION})
        # Explicit hand-auditable service allocation. Infeasible fixtures have no service rows.
        schedule = defaultdict(dict)
        if sid == "F01_feasible":
            for slot in range(40, 44):
                schedule[slot] = {ids[0]: (ports[0], Decimal(40)), ids[1]: (ports[1], Decimal(40))}
        elif sid == "F04_cancel_reassign":
            for slot in (42, 43):
                schedule[slot] = {ids[1]: (ports[0], Decimal(120))}
        elif sid == "F05_pressure":
            for slot in range(40, 44):
                offset = 0 if slot < 42 else 4
                schedule[slot] = {ids[offset + j]: (ports[j], Decimal(120)) for j in range(4)}
        for slot in range(96):
            existing = Decimal(80) if 40 <= slot < 44 else Decimal(0)
            occupied = [existing_port] if existing else []
            capacity_rows.append({"station_id": STATION, "scenario_id": sid, "dataset_id": DATASET,
                                  "is_simulated": "true", "source_versions": "D01_truth_past_only|registry_v1|storage_v1|synthetic_existing_service_v1",
                                  "business_date": CFG["business_date"], "timestamp": time(slot), "interval_end": time(slot + 1),
                                  "as_of": time(0), "available_at": time(0),
                                  "load_source_end": source_observation_ends[slot][0],
                                  "pv_source_end": source_observation_ends[slot][1],
                                  "forecast_input_sha256": base_input_hash,
                                  "base_load_forecast_kw": fmt(base_curve[slot]),
                                  "existing_committed_charge_kw": fmt(existing),
                                  "existing_commitment_status": "synthetic_known_schedule",
                                  "pv_available_forecast_kw": fmt(pv_curve[slot]),
                                  "grid_import_limit_kw": "10000.000000", "grid_export_limit_kw": "0.000000",
                                  "fleet_allocated_capacity_kw": fmt(cap), "station_total_ports": 128,
                                  "fleet_mapped_ports": 4, "unmapped_ports": 124,
                                  "existing_service_port_id": existing_port if existing else "",
                                  "occupied_port_ids_json": json.dumps(occupied),
                                  "other_ports_free_count": 124 - len(occupied),
                                  "port_rated_kw": "240.000000", "fleet_port_limit_kw": "120.000000",
                                  "storage_initial_energy_kwh": "2400.000000", "storage_site_dispatch_ready": "false",
                                  "import_tariff_cny_per_kwh": fmt(tariff(slot))})
            if sid in ("F02_station_short", "F03_deadline_conflict"):
                continue
            vehicle_kw = {vid: power for vid, (_, power) in schedule.get(slot, {}).items()}
            new_kw = sum(vehicle_kw.values(), Decimal(0))
            demand = base_curve[slot] + existing + new_kw
            pv_used = min(pv_curve[slot], demand)
            grid = demand - pv_used
            assert Decimal(0) <= grid <= Decimal(10000) and new_kw <= Decimal(cap)
            for vid, (port, power) in schedule.get(slot, {}).items():
                allocations.append({"station_id": STATION, "scenario_id": sid, "dataset_id": DATASET,
                                    "is_simulated": "true", "source_version": "synthetic_feasible_fixture_v1",
                                    "timestamp": time(slot), "interval_end": time(slot + 1),
                                    "vehicle_id": vid, "port_id": port, "charge_power_kw": fmt(power),
                                    "ac_energy_kwh": fmt(power / 4),
                                    "battery_energy_gain_kwh": fmt(power * Decimal("0.95") / 4)})
            service_rows.append({"station_id": STATION, "scenario_id": sid, "dataset_id": DATASET,
                                 "is_simulated": "true", "source_versions": "D01_past_only|synthetic_feasible_fixture_v1",
                                 "timestamp": time(slot), "interval_end": time(slot + 1),
                                 "vehicle_charge_kw_json": json.dumps({k: fmt(v) for k, v in vehicle_kw.items()}, sort_keys=True),
                                 "fleet_charge_kw": fmt(new_kw), "existing_charge_kw": fmt(existing),
                                 "base_load_kw": fmt(base_curve[slot]), "pv_available_kw": fmt(pv_curve[slot]),
                                 "pv_used_kw": fmt(pv_used), "pv_curtailed_kw": fmt(pv_curve[slot] - pv_used),
                                 "storage_charge_kw": "0.000000", "storage_discharge_kw": "0.000000",
                                 "storage_energy_after_kwh": "2400.000000", "grid_import_kw": fmt(grid),
                                 "grid_export_kw": "0.000000", "fleet_capacity_kw": fmt(cap),
                                 "feasible_reference_fixture": "true"})
        if sid not in ("F02_station_short", "F03_deadline_conflict"):
            own = [r for r in service_rows if r["scenario_id"] == sid]
            for resolution in (30, 60):
                n = resolution // 15
                for offset in range(0, 96, n):
                    block = own[offset:offset + n]
                    per_vehicle = defaultdict(Decimal)
                    for row in block:
                        for vid, power in json.loads(row["vehicle_charge_kw_json"]).items():
                            per_vehicle[vid] += Decimal(power) / 4
                    grid_energy = sum((Decimal(r["grid_import_kw"]) / 4 for r in block), Decimal(0))
                    pv_energy = sum((Decimal(r["pv_used_kw"]) / 4 for r in block), Decimal(0))
                    storage_energy = sum((Decimal(r["storage_charge_kw"]) / 4 - Decimal(r["storage_discharge_kw"]) / 4 for r in block), Decimal(0))
                    aggregates.append({"station_id": STATION, "scenario_id": sid, "dataset_id": DATASET,
                                       "is_simulated": "true", "source_version": "aggregate_from_15min_service_v1",
                                       "resolution_minutes": resolution, "timestamp": time(offset),
                                       "interval_end": time(offset + n),
                                       "vehicle_ac_energy_kwh_json": json.dumps({k: fmt(v) for k, v in sorted(per_vehicle.items())}, sort_keys=True),
                                       "grid_energy_kwh": fmt(grid_energy), "grid_mean_kw": fmt(grid_energy * 60 / resolution),
                                       "pv_used_energy_kwh": fmt(pv_energy),
                                       "storage_net_charge_energy_kwh": fmt(storage_energy),
                                       "aggregation_rule": "sum_15min_energy_then_divide_by_duration"})
        manifests.append({"scenario_id": sid, "station_id": STATION, "business_date": CFG["business_date"],
                          "as_of": time(0), "expected_status": expected[sid],
                          "small_hand_case": sid != "F05_pressure", "fleet_vehicle_count": count,
                          "mapped_port_count": 4, "unmapped_station_ports": 124,
                          "fleet_capacity_kw": cap, "source_input_hash": base_input_hash,
                          "is_simulated": True, "source_version": DATASET,
                          "selection_note": "synthetic fixture; 2025 observed truth is not a decision feature"})

    # Message payloads use the fleet-platform contract shape, with explicit simulation evidence.
    services = {}
    fingerprints = {}
    for sid in capacities:
        own = [r for r in service_rows if r["scenario_id"] == sid]
        if own:
            vids = [r["vehicle_id"] for r in vehicles if r["scenario_id"] == sid]
            service = {"charging": {vid: [json.loads(r["vehicle_charge_kw_json"]).get(vid, "0.000000") for r in own] for vid in vids},
                       "capacity_kw": [r["fleet_capacity_kw"] for r in own],
                       "storage_charge_kw": [r["storage_charge_kw"] for r in own],
                       "storage_discharge_kw": [r["storage_discharge_kw"] for r in own],
                       "storage_energy_kwh": [r["storage_energy_after_kwh"] for r in own],
                       "net_import_kw": [r["grid_import_kw"] for r in own],
                       "pv_used_kw": [r["pv_used_kw"] for r in own], "feasible": True}
            services[sid] = service
            fingerprints[sid] = digest({"scenario_id": sid, "version": 2 if sid == "F04_cancel_reassign" else 1,
                                        "service": service, "source_input_hash": base_input_hash})
        else:
            fingerprints[sid] = digest({"scenario_id": sid, "requests": [r for r in requests if r["scenario_id"] == sid],
                                        "capacity_kw": capacities[sid], "seed": CFG["seed"]})

    def message(sid: str, kind: str, version: int, minute: int, payload: dict, *, source: str,
                message_id: str | None = None, fp: str | None = None, received: int | None = None) -> dict:
        row = {"scenario_id": sid, "message_id": message_id or f"{sid}-{kind}-v{version}-{minute}",
               "task_id": f"FLEET-TASK-{sid}", "kind": kind, "round_id": version,
               "version": version, "plan_fingerprint": fp or fingerprints[sid],
               "issued_at": (DAY + timedelta(minutes=minute)).isoformat(timespec="seconds"),
               "expires_at": time(96),
               "received_at": (DAY + timedelta(minutes=minute if received is None else received)).isoformat(timespec="seconds"),
               "source": source, "evidence": {"reference": f"{sid}-{kind}-synthetic-evidence", "simulation": True},
               "payload": payload, "is_simulated": True, "source_version": "fleet_collaboration_fixture_v1"}
        for contract_field in ("passed", "reason", "capacity_kw", "service", "equilibrium", "status", "confirmed"):
            if contract_field in payload:
                row[contract_field] = payload[contract_field]
        messages.append(row)
        return row

    for sid in capacities:
        if sid == "F04_cancel_reassign":
            old_fp = digest({"scenario_id": sid, "version": 1, "reservation": "old_vehicle_on_port_1", "seed": CFG["seed"]})
            message(sid, "reservation", 1, 1, {"passed": True, "port_id": ports[0], "vehicle_id": f"{sid}-EV-01"}, source="S1", fp=old_fp)
            message(sid, "adjustment", 1, 2, {"action": "cancel_old_reservation", "capacity_kw": ["0.000000"] * 96}, source="coordinator", fp=old_fp)
            message(sid, "cancel_ack", 1, 3, {"confirmed": True, "released_port_id": ports[0]}, source="S1", fp=old_fp)
            version, first_minute = 2, 4
        else:
            version, first_minute = 1, 1
        message(sid, "adjustment", version, first_minute,
                {"capacity_kw": [fmt(capacities[sid])] * 96}, source="coordinator")
        if sid in ("F02_station_short", "F03_deadline_conflict"):
            rejection = message(sid, "coordination_result", version, first_minute + 1,
                                {"passed": False, "reason": expected[sid], "service": None,
                                 "equilibrium": {"passed": False, "method": "not_applicable", "evidence": "infeasible_fixture"}},
                                source="coordinator")
            if sid == "F02_station_short":
                message(sid, "coordination_result", version, first_minute + 1,
                        {"passed": True, "reason": "conflicting_same_id", "service": None,
                         "equilibrium": {"passed": False, "method": "not_applicable", "evidence": "conflict"}},
                        source="coordinator", message_id=rejection["message_id"], received=first_minute + 2)
            continue
        if sid == "F01_feasible":
            message(sid, "coordination_result", version, first_minute + 1,
                    {"passed": False, "reason": "economic_equilibrium_not_proven",
                     "service": services[sid],
                     "equilibrium": {"passed": False, "method": "not_proven", "evidence": "only_feasibility_handcheck"}},
                    source="coordinator")
        grid = message(sid, "grid", version, first_minute + 2,
                       {"passed": True, "net_import_kw": services[sid]["net_import_kw"],
                        "reason": "synthetic_grid_limit_check"}, source="grid")
        if sid == "F04_cancel_reassign":
            messages.append({**grid, "received_at": (DAY + timedelta(minutes=first_minute + 3)).isoformat(timespec="seconds")})
        message(sid, "reservation", version, first_minute + 4,
                {"passed": True, "port_ids": ports[:4], "reason": "synthetic_S1_confirmation"}, source="S1")
        message(sid, "execution", version, 12 * 60,
                {"status": "COMPLETED", "executed_service_fingerprint": fingerprints[sid]}, source="S1")

    # Separate retrospective labels and an as_of-redacted decision view.
    history: list[dict] = []
    visible_history: list[dict] = []
    hcfg = CFG["history_assumptions"]
    cutoff = datetime.fromisoformat(hcfg["visible_as_of"])
    day = datetime.fromisoformat("2025-08-01T00:00:00+08:00")
    stop = datetime.fromisoformat("2025-10-01T00:00:00+08:00")
    while day < stop:
        ordinal = day.toordinal()
        split = "train" if day.date().isoformat() < hcfg["train_end_exclusive"] else "calibration" if day.date().isoformat() < hcfg["calibration_end_exclusive"] else "holdout_replay"
        for slot in hcfg["daily_slots"]:
            t = day + slot * STEP
            source = truth_map[t.isoformat(timespec="seconds")]
            signed_before = ordinal % hcfg["signed_before_target_day_ordinal_mod"] == 0
            missing = ordinal % hcfg["missing_day_ordinal_mod"] == 0
            later_signed = ordinal % hcfg["later_signed_day_ordinal_mod"] == 0
            sign_time = (t - timedelta(hours=1)) if signed_before else (t + timedelta(days=2)) if later_signed else None
            if missing:
                label, energy = "missing", ""
            elif signed_before:
                label, energy = "signed_excluded", ""
            else:
                power = Decimal(source["historical_charge_kw"]) * Decimal("0.1")
                energy = fmt(power / 4)
                label = "observed_zero" if power == 0 else "observed_uncontracted"
            record = {"station_id": STATION, "scenario_id": "HISTORY_UNCONTRACTED", "dataset_id": DATASET,
                      "is_simulated": "true", "source_version": "D01_historical_charge+synthetic_contract_status_v1",
                      "business_date": day.date().isoformat(), "timestamp": t.isoformat(timespec="seconds"),
                      "interval_end": (t + STEP).isoformat(timespec="seconds"),
                      "label_available_at": (t + STEP + timedelta(minutes=5)).isoformat(timespec="seconds"),
                      "split": split, "label_status": label, "uncontracted_energy_kwh": energy,
                      "signed_at": "" if sign_time is None else sign_time.isoformat(timespec="seconds"),
                      "quality": "missing" if missing else "synthetic_observed"}
            history.append(record)
            if datetime.fromisoformat(record["label_available_at"]) <= cutoff:
                visible_history.append({**record, "as_of": hcfg["visible_as_of"],
                                        "signed_at": record["signed_at"] if sign_time is not None and sign_time <= cutoff else "",
                                        "future_signing_redacted": "true" if sign_time is not None and sign_time > cutoff else "false"})
        day += timedelta(days=1)

    battery_trace: list[dict] = []
    allocation_lookup = {(a["scenario_id"], a["vehicle_id"], a["timestamp"]): a for a in allocations}
    task_lookup = {(t["scenario_id"], t["vehicle_id"]): t for t in tasks}
    for v in vehicles:
        sid, vid = v["scenario_id"], v["vehicle_id"]
        if sid in ("F02_station_short", "F03_deadline_conflict"):
            continue
        energy = Decimal(v["initial_energy_kwh"])
        task = task_lookup.get((sid, vid))
        for slot in range(96):
            allocation = allocation_lookup.get((sid, vid, time(slot)))
            gain = Decimal(allocation["battery_energy_gain_kwh"]) if allocation else Decimal(0)
            travel = Decimal(task["travel_energy_kwh"]) if task and task["travel_end"] == time(slot + 1) else Decimal(0)
            before = energy
            energy = before + gain - travel
            assert Decimal(v["reserve_energy_kwh"]) <= energy <= Decimal(v["battery_kwh"])
            battery_trace.append({"station_id": STATION, "scenario_id": sid, "dataset_id": DATASET,
                                  "is_simulated": "true", "source_version": "synthetic_post_execution_battery_trace_v1",
                                  "vehicle_id": vid, "timestamp": time(slot), "interval_end": time(slot + 1),
                                  "available_at": time(slot + 1), "view": "post_execution_reference",
                                  "energy_before_kwh": fmt(before), "charge_gain_kwh": fmt(gain),
                                  "transport_energy_kwh": fmt(travel), "energy_after_kwh": fmt(energy),
                                  "soc_after_percent": fmt(energy / 4),
                                  "charge_port_id": allocation["port_id"] if allocation else "",
                                  "transport_task_id": task["task_id"] if travel else ""})

    csv_out("vehicles.csv", vehicles)
    csv_out("transport_tasks.csv", tasks)
    csv_out("charging_requests.csv", requests)
    csv_out("station_capacity_15m.csv", capacity_rows)
    csv_out("allocation_15m.csv", allocations)
    csv_out("station_service_15m.csv", service_rows)
    csv_out("fleet_aggregates_30_60m.csv", aggregates)
    csv_out("vehicle_energy_trace_15m.csv", battery_trace)
    csv_out("history_uncontracted.csv", history)
    csv_out("history_uncontracted_visible.csv", visible_history)
    jsonl_out("collaboration_messages.jsonl", messages)
    json_out("scenario_manifest.json", manifests)
    json_out("source_manifest.json", [{"path": name, "sha256": h, "read_only": True,
                                       "station_id": STATION, "scenario_id": "PACKAGE", "is_simulated": True}
                                      for name, h in sorted(CFG["source_hashes_sha256"].items())])
    outputs = ["identity_mapping.csv", "vehicles.csv", "transport_tasks.csv", "charging_requests.csv",
               "station_capacity_15m.csv", "allocation_15m.csv", "station_service_15m.csv",
               "fleet_aggregates_30_60m.csv", "vehicle_energy_trace_15m.csv",
               "history_uncontracted.csv", "history_uncontracted_visible.csv",
               "collaboration_messages.jsonl", "scenario_manifest.json", "source_manifest.json"]
    assert all(sha(ROOT / name) == h for name, h in CFG["source_hashes_sha256"].items())
    json_out("validation_report.json", {"dataset_id": DATASET, "status": "generated_pending_independent_validation",
             "is_simulated": True, "source_hashes_sha256": CFG["source_hashes_sha256"],
             "output_hashes_sha256": {name: sha(HERE / name) for name in outputs},
             "counts": {"scenarios": len(manifests), "vehicles": len(vehicles), "tasks": len(tasks),
                        "requests": len(requests), "capacity_intervals": len(capacity_rows),
                        "service_intervals": len(service_rows), "allocations": len(allocations),
                        "history_intervals": len(history), "messages": len(messages),
                        "vehicle_energy_trace_intervals": len(battery_trace)},
             "limitations": "Synthetic small/full-day fixture; no live grid or S1 confirmation, no economic equilibrium proof"})


if __name__ == "__main__":
    main()
