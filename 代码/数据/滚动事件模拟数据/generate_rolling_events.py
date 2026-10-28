"""Create deterministic synthetic rolling-event fixtures from the D01 offline replay."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
CFG = json.loads((HERE / "simulation_config.json").read_text(encoding="utf-8"))
STATION = CFG["station_id"]
DATASET = CFG["dataset_id"]
Q = Decimal("0.000001")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(obj: object) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def fmt(value: Decimal | int | float) -> str:
    return str(Decimal(str(value)).quantize(Q))


def dump_json(name: str, obj: object) -> None:
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def dump_jsonl(name: str, records: list[dict]) -> None:
    (HERE / name).write_text("".join(canonical(r) + "\n" for r in records), encoding="utf-8")


def replay(stream: list[dict]) -> dict:
    """Small reference consumer: it checks workflow invariants, not optimizer quality."""
    state = {"status": "new", "current_version": 0, "current_fingerprint": "", "accepted": False,
             "cancel_pending": False, "connected": True, "blocked": False, "measurement_good": False,
             "forecast_good": False, "executed": {}, "replan_count": 0, "duplicates": 0,
             "conflicts": 0, "stale_receipts": 0, "double_occupancy_blocks": 0,
             "rejections": [], "seen": {}, "plan_steps": {}}
    for e in sorted(stream, key=lambda r: (r["received_at"], r["ordinal"])):
        semantic = {k: v for k, v in e.items() if k not in ("received_at", "available_at", "ordinal")}
        sig = fingerprint(semantic)
        old = state["seen"].get(e["event_id"])
        if old is not None:
            if old == sig:
                state["duplicates"] += 1
            else:
                state["conflicts"] += 1
                state["rejections"].append("EVENT_ID_CONTENT_CONFLICT")
            continue
        state["seen"][e["event_id"]] = sig
        kind = e["event_type"]
        p = e["payload"]
        if kind == "forecast_issued":
            state["forecast_good"] = (p["quality"] == "good" and p["coverage_count"] == 4
                                      and datetime.fromisoformat(p["valid_until"]) >= datetime.fromisoformat(e["received_at"]))
            if not state["forecast_good"]:
                state["blocked"] = True
                state["status"] = "blocked_forecast"
        elif kind == "measurement":
            state["measurement_good"] = (p["quality"] == "good" and p["base_kw"] is not None
                                         and p["soc_percent"] is not None
                                         and datetime.fromisoformat(p["valid_until"]) >= datetime.fromisoformat(e["received_at"]))
            if not state["measurement_good"]:
                state["blocked"] = True
                state["status"] = "blocked_measurement"
            elif state["accepted"] and p.get("expected_soc_percent") is not None:
                if abs(Decimal(str(p["soc_percent"])) - Decimal(str(p["expected_soc_percent"]))) > Decimal("0.5"):
                    state["status"] = "replan_required"
            if state["accepted"] and p.get("forecast_pv_kw") is not None:
                if Decimal(str(p["forecast_pv_kw"])) - Decimal(str(p["pv_kw"])) > Decimal(CFG["plan_fixture"]["pv_drop_trigger_kw"]):
                    state["status"] = "replan_required"
        elif kind == "booking_changed":
            if p.get("remaining_energy_kwh") is not None:
                hours = (datetime.fromisoformat(p["deadline_at"]) - datetime.fromisoformat(e["business_time"])).total_seconds() / 3600
                if Decimal(str(p["remaining_energy_kwh"])) > Decimal(str(max(0, hours))) * Decimal(str(p["port_max_kw"])):
                    state["blocked"] = True
                    state["status"] = "remaining_service_infeasible"
        elif kind in ("derating", "protection", "manual_takeover", "device_frozen"):
            if kind == "derating" and state["accepted"]:
                state["status"] = "replan_required"
            elif kind != "derating":
                state["blocked"] = True
                state["status"] = "blocked_safety"
        elif kind == "connection_lost":
            state["connected"] = False
            state["status"] = "blocked_connection"
        elif kind == "connection_restored":
            state["connected"] = True
            state["status"] = "replan_required"
        elif kind == "replan_requested":
            state["replan_count"] += 1
            if state["replan_count"] > CFG["plan_fixture"]["max_replans"]:
                state["blocked"] = True
                state["status"] = "replan_limit_exhausted"
            else:
                state["status"] = "replan_required"
        elif kind == "plan_sent":
            if state["cancel_pending"]:
                state["double_occupancy_blocks"] += 1
                state["rejections"].append("CANCEL_NOT_ACKNOWLEDGED")
                continue
            if state["blocked"] or not state["connected"] or not state["measurement_good"] or not state["forecast_good"]:
                state["rejections"].append("PUBLISH_BLOCKED")
                continue
            if e["plan_version"] > 1 and p["plan"]["frozen_prefix_fingerprint"] != fingerprint(state["executed"]):
                state["rejections"].append("EXECUTED_PREFIX_MISMATCH")
                continue
            state["current_version"] = e["plan_version"]
            state["current_fingerprint"] = e["plan_fingerprint"]
            state["plan_steps"] = {str(s["step_index"]): s for s in p["plan"]["steps"]}
            state["accepted"] = False
            state["status"] = "sent"
        elif kind == "plan_accepted":
            if e["plan_version"] == state["current_version"] and e["plan_fingerprint"] == state["current_fingerprint"] and not state["blocked"]:
                state["accepted"] = True
                state["status"] = "active"
        elif kind == "plan_rejected":
            if e["plan_version"] == state["current_version"] and e["plan_fingerprint"] == state["current_fingerprint"]:
                state["accepted"] = False
                state["status"] = "rejected_zero_execution"
        elif kind == "step_completed":
            if e["plan_version"] != state["current_version"] or e["plan_fingerprint"] != state["current_fingerprint"]:
                state["stale_receipts"] += 1
                continue
            if not state["accepted"]:
                state["rejections"].append("STEP_WITHOUT_ACCEPTED_PLAN")
                continue
            key = str(p["step_index"])
            if key in state["executed"]:
                state["duplicates"] += 1
                continue
            if key not in state["plan_steps"]:
                state["rejections"].append("STEP_OUTSIDE_PLAN")
                continue
            state["executed"][key] = p["actual"]
            if abs(Decimal(p["actual"]["storage_energy_after_kwh"]) - Decimal(state["plan_steps"][key]["planned_storage_energy_after_kwh"])) > Decimal(CFG["plan_fixture"]["soc_drift_trigger_kwh"]):
                state["status"] = "replan_required"
            elif len(state["executed"]) == 4:
                state["status"] = "completed"
        elif kind == "cancel_requested":
            if e["plan_version"] == state["current_version"]:
                state["cancel_pending"] = True
                state["status"] = "cancel_pending"
        elif kind == "cancel_acknowledged":
            if e["plan_version"] == state["current_version"] and state["cancel_pending"]:
                state["cancel_pending"] = False
                state["accepted"] = False
                state["status"] = "cancelled"
    return {k: v for k, v in state.items() if k != "seen"}


def main() -> None:
    for name, expected in CFG["source_hashes_sha256"].items():
        assert sha(DATA / name) == expected, name
    base_id = CFG["base_scenario_id"]
    scenario = next(r for r in read(DATA / "场站联动模拟数据/scenario_inputs.csv") if r["scenario_id"] == base_id)
    assert scenario["station_id"] == STATION and scenario["forecast_status"] == "usable"
    start = datetime.fromisoformat(scenario["window_start"])
    source_truth = sorted((r for r in read(DATA / "场站联动模拟数据/scenario_truth.csv") if r["scenario_id"] == base_id), key=lambda r: r["timestamp"])
    source_forecasts = [r for r in read(DATA / "场站联动模拟数据/forecast_values.csv") if r["scenario_id"] == base_id and r["resource"] == "load"]
    source_pv_forecasts = [r for r in read(DATA / "场站联动模拟数据/forecast_values.csv") if r["scenario_id"] == base_id and r["resource"] == "pv"]
    source_bookings = [r for r in read(DATA / "场站联动模拟数据/booking_events.csv") if r["scenario_id"] == base_id and r["event_type"] == "booking"]
    assert len(source_truth) == len(source_forecasts) == len(source_pv_forecasts) == 4
    assert all(datetime.fromisoformat(r["available_at"]) > start for r in source_truth)
    assert all(datetime.fromisoformat(r["source_observation_end"]) <= start for r in source_forecasts)
    source_forecasts.sort(key=lambda r: r["timestamp"])
    source_pv_forecasts.sort(key=lambda r: r["timestamp"])
    assert all(datetime.fromisoformat(r["source_observation_end"]) <= start for r in source_pv_forecasts)
    previous = next(r for r in read(DATA / "场站联动模拟数据/station_interval_truth.csv") if r["timestamp"] == (start - timedelta(minutes=15)).isoformat(timespec="seconds"))
    assert previous["available_at"] == start.isoformat(timespec="seconds")
    storage = read(DATA / "储能参数与初始状态/储能参数.csv")[0]
    eta = Decimal(storage["discharge_efficiency"])

    def at(seconds: int) -> str:
        return (start + timedelta(seconds=seconds)).isoformat(timespec="seconds")

    def make_plan(sid: str, version: int, frozen: dict | None = None) -> tuple[dict, str]:
        steps = []
        for k in (range(4) if version == 1 else range(2, 4)):
            forecast_load = Decimal(source_forecasts[k]["point_kw"])
            steps.append({"step_index": k, "start": at(k * 900), "end": at((k + 1) * 900),
                          "vehicle_power_kw": CFG["plan_fixture"]["vehicles"],
                          "planned_storage_charge_kw": "0.000000", "planned_storage_discharge_kw": "0.000000",
                          "planned_storage_energy_after_kwh": "2400.000000", "planned_pv_used_kw": "0.000000",
                          "planned_grid_import_kw": fmt(forecast_load + 80)})
        plan = {"station_id": STATION, "task_id": f"TASK-{sid}", "version": version,
                "seed": CFG["seed"],
                "source_scenario_id": base_id, "window_start": at(0 if version == 1 else 1800), "window_end": at(3600),
                "frozen_prefix_fingerprint": fingerprint({} if frozen is None else frozen),
                "steps": steps, "fixture_kind": "nonoptimized_workflow_stub"}
        return plan, fingerprint(plan)

    def actual_step(k: int, discharge_kw: Decimal = Decimal(0)) -> dict:
        row = source_truth[k]
        total = Decimal(row["total_load_kw"])
        vehicle_energy = {v: fmt(Decimal(p) / 4) for v, p in CFG["plan_fixture"]["vehicles"].items()}
        after = Decimal(2400) - discharge_kw / (4 * eta)
        grid_kwh = (total + Decimal(80) - discharge_kw) / 4
        assert 0 <= grid_kwh <= Decimal(2500)
        return {"step_index": k, "interval_start": at(k * 900), "interval_end": at((k + 1) * 900),
                "vehicle_energy_kwh": vehicle_energy, "base_energy_kwh": fmt(Decimal(row["base_load_kw"]) / 4),
                "historical_charge_energy_kwh": fmt(Decimal(row["historical_charge_kw"]) / 4),
                "pv_used_energy_kwh": "0.000000", "storage_charge_kw": "0.000000",
                "storage_discharge_kw": fmt(discharge_kw), "storage_charge_energy_ac_kwh": "0.000000",
                "storage_discharge_energy_ac_kwh": fmt(discharge_kw / 4),
                "grid_import_energy_kwh": fmt(grid_kwh), "storage_energy_before_kwh": "2400.000000",
                "storage_energy_after_kwh": fmt(after), "soc_after_percent": fmt(after / 40),
                "quality": "good", "valid_until": at((k + 2) * 900)}

    specs = [
        ("R01_normal", "completed", "all four intervals execute"),
        ("R02_duplicate", "completed", "duplicate receipt idempotent"),
        ("R03_rejected", "rejected_zero_execution", "rejected plan executes zero steps"),
        ("R04_soc_drift", "replan_required", "actual SOC differs after first step"),
        ("R05_pv_drop", "replan_required", "PV measurement falls below forecast"),
        ("R06_grid_derate", "replan_required", "PCC capacity shrinks after first step"),
        ("R07_cancel_ack_late", "active", "no second plan before cancellation ack"),
        ("R08_old_receipt", "active", "old version receipt cannot change current confirmation"),
        ("R09_measurement_before_receipt", "active", "new measurement precedes old receipt"),
        ("R10_event_id_conflict", "active", "same ID with changed content rejected"),
        ("R11_restart_replay", "completed", "restart replay yields identical state"),
        ("R12_protection", "blocked_safety", "protection blocks publishing"),
        ("R13_manual_takeover", "blocked_safety", "manual takeover blocks publishing"),
        ("R14_device_frozen", "blocked_safety", "frozen device blocks publishing"),
        ("R15_poor_measurement", "blocked_measurement", "unknown measurement blocks publishing"),
        ("R16_stale_measurement", "blocked_measurement", "expired measurement blocks publishing"),
        ("R17_forecast_expired", "blocked_forecast", "expired forecast blocks publishing"),
        ("R18_forecast_gap", "blocked_forecast", "incomplete forecast blocks publishing"),
        ("R19_connection_lost", "blocked_connection", "disconnection blocks publishing"),
        ("R20_connection_restored", "replan_required", "restoration requires new check"),
        ("R21_replan_exhausted", "replan_limit_exhausted", "third attempt exceeds limit"),
        ("R22_remaining_infeasible", "remaining_service_infeasible", "new service cannot fit remaining window"),
    ]
    all_events: list[dict] = []
    all_truth: list[dict] = []
    event_truth: list[dict] = []
    manifests: list[dict] = []
    results: list[dict] = []
    for sid, expected, invariant in specs:
        task_id = f"TASK-{sid}"
        plan1, fp1 = make_plan(sid, 1)
        stream: list[dict] = []

        def emit(kind: str, seconds: int, payload: dict, *, version: int = 0, fp: str = "",
                 received: int | None = None, available: int | None = None, business: int | None = None,
                 measured: int | None = None, asset: str = "STATION", event_id: str | None = None) -> dict:
            ordinal = len(stream)
            receive_seconds = seconds if received is None else received
            visible_seconds = receive_seconds if available is None else available
            row = {"scenario_id": sid, "event_id": event_id or f"{sid}-E{ordinal:03d}",
                   "event_type": kind, "station_id": STATION, "asset_id": asset,
                   "business_time": at(seconds if business is None else business),
                   "measured_at": "" if measured is None else at(measured),
                   "available_at": at(visible_seconds),
                   "received_at": at(receive_seconds),
                   "task_id": task_id, "plan_version": version, "plan_fingerprint": fp,
                   "payload": payload, "is_simulated": True,
                   "source_ref": "D01:S06_calibration+rolling_fixture_v1", "ordinal": ordinal}
            assert datetime.fromisoformat(row["available_at"]) <= datetime.fromisoformat(row["received_at"])
            stream.append(row)
            return row

        forecast_quality = "good"
        forecast_count = 4
        forecast_valid = at(3600)
        if sid == "R17_forecast_expired":
            forecast_valid = at(-1)
        if sid == "R18_forecast_gap":
            forecast_count = 2
        emit("forecast_issued", 0, {"vintage_ids": {"load": scenario["load_vintage_id"], "pv": scenario["pv_vintage_id"]},
                                    "load_point_kw": [r["point_kw"] for r in source_forecasts[:forecast_count]],
                                    "pv_point_kw": [r["point_kw"] for r in source_pv_forecasts[:forecast_count]],
                                    "issued_at": scenario["as_of"],
                                    "native_grid": "15min_interval_average", "coverage_count": forecast_count,
                                    "quality": forecast_quality, "valid_until": forecast_valid}, asset="FORECAST")
        measure_quality = "unknown" if sid == "R15_poor_measurement" else "good"
        measure_valid = at(-1) if sid == "R16_stale_measurement" else at(1800)
        emit("measurement", 0, {"base_kw": None if measure_quality == "unknown" else previous["base_load_kw"],
                                "pv_kw": None if measure_quality == "unknown" else previous["pv_interval_mean_kw_approx"],
                                "grid_kw": None if measure_quality == "unknown" else fmt(Decimal(previous["total_load_kw"]) - min(Decimal(previous["total_load_kw"]), Decimal(previous["pv_interval_mean_kw_approx"]))),
                                "soc_percent": None if measure_quality == "unknown" else "60.000000",
                                "equipment_quality": measure_quality, "quality": measure_quality,
                                "valid_until": measure_valid}, business=-900, measured=0, asset="M-PCC")
        emit("booking_changed", 0, {"visible_bookings": [
                                    {"session_id": r["session_id"], "port_id": r["port_id"],
                                     "committed_energy_kwh": r["committed_energy_kwh"], "available_at": r["available_at"]}
                                    for r in source_bookings[:2]],
                                    "quality": "synthetic", "valid_until": at(3600)}, asset="BR-CHARGE")
        preblock = {"R12_protection": "protection", "R13_manual_takeover": "manual_takeover",
                    "R14_device_frozen": "device_frozen", "R19_connection_lost": "connection_lost",
                    "R20_connection_restored": "connection_lost"}.get(sid)
        if preblock:
            emit(preblock, 0, {"active": True, "quality": "good", "valid_until": at(3600)}, asset="PCS-001")
        emit("plan_sent", 0, {"plan": plan1, "quality": "synthetic_fixture", "valid_until": at(3600)}, version=1, fp=fp1)
        if sid in {"R12_protection", "R13_manual_takeover", "R14_device_frozen", "R15_poor_measurement",
                   "R16_stale_measurement", "R17_forecast_expired", "R18_forecast_gap", "R19_connection_lost", "R20_connection_restored"}:
            if sid == "R20_connection_restored":
                emit("connection_restored", 10, {"active": False, "quality": "good", "valid_until": at(3600)}, asset="PCC")
        elif sid == "R03_rejected":
            emit("plan_rejected", 1, {"reason": "simulated_station_rejection", "executed_steps": 0,
                                       "quality": "good", "valid_until": at(900)}, version=1, fp=fp1)
        else:
            emit("plan_accepted", 0, {"accepted": True, "quality": "good", "valid_until": at(3600)}, version=1, fp=fp1)
            n_steps = 4 if sid in {"R01_normal", "R02_duplicate", "R11_restart_replay"} else 1
            executed = {}
            for k in range(n_steps):
                discharge = Decimal(200) if sid == "R04_soc_drift" and k == 0 else Decimal(0)
                actual = actual_step(k, discharge)
                event = emit("step_completed", (k + 1) * 900, {"step_index": k, "actual": actual,
                                      "quality": "good", "valid_until": at((k + 2) * 900)},
                             version=1, fp=fp1, measured=(k + 1) * 900, received=(k + 1) * 900 + 5,
                             business=k * 900, asset="M-PCC")
                executed[str(k)] = actual
                all_truth.append({"scenario_id": sid, "station_id": STATION, "task_id": task_id,
                                  "plan_version": 1, "step_index": k, "business_time": at(k * 900),
                                  "available_at": at((k + 1) * 900), "actual": actual,
                                  "is_simulated": True, "source_ref": "D01:scenario_truth+synthetic_execution_intervention"})
                if sid == "R02_duplicate" and k == 0:
                    emit("step_completed", 900, event["payload"], version=1, fp=fp1,
                         measured=900, received=910, business=0, asset="M-PCC", event_id=event["event_id"])
            if sid == "R05_pv_drop":
                emit("measurement", 920, {"base_kw": source_truth[0]["base_load_kw"], "pv_kw": "0.000000",
                                          "grid_kw": source_truth[0]["total_load_kw"], "soc_percent": "60.000000",
                                          "expected_soc_percent": "60.000000", "forecast_pv_kw": source_pv_forecasts[1]["point_kw"],
                                          "quality": "good", "valid_until": at(1800)}, measured=920, asset="M-PV")
            if sid == "R06_grid_derate":
                emit("derating", 920, {"new_import_limit_kw": "500.000000", "quality": "good",
                                        "valid_until": at(3600)}, asset="PCC")
            if sid in {"R07_cancel_ack_late", "R08_old_receipt", "R09_measurement_before_receipt"}:
                emit("cancel_requested", 960, {"reason": "replan", "quality": "good", "valid_until": at(3600)}, version=1, fp=fp1)
                plan2, fp2 = make_plan(sid, 2, executed)
                if sid == "R07_cancel_ack_late":
                    emit("plan_sent", 1020, {"plan": plan2, "quality": "synthetic_fixture", "valid_until": at(3600)}, version=2, fp=fp2)
                emit("cancel_acknowledged", 1800, {"confirmed": True, "stopped_at": at(900),
                                                   "executed_steps": 1, "quality": "good", "valid_until": at(3600)}, version=1, fp=fp1)
                emit("plan_sent", 1800, {"plan": plan2, "quality": "synthetic_fixture", "valid_until": at(3600)}, version=2, fp=fp2)
                emit("plan_accepted", 1800, {"accepted": True, "quality": "good", "valid_until": at(3600)}, version=2, fp=fp2)
                if sid == "R09_measurement_before_receipt":
                    emit("measurement", 2040, {"base_kw": source_truth[1]["base_load_kw"], "pv_kw": source_truth[1]["pv_interval_mean_kw_approx"],
                                               "grid_kw": source_truth[1]["total_load_kw"], "soc_percent": "60.000000",
                                               "quality": "good", "valid_until": at(3600)}, measured=2040, asset="M-PCC")
                if sid in {"R08_old_receipt", "R09_measurement_before_receipt"}:
                    emit("step_completed", 900, {"step_index": 0, "actual": executed["0"], "quality": "good",
                                                  "valid_until": at(3600)}, version=1, fp=fp1, measured=900,
                         received=2100, business=0, asset="M-PCC")
            if sid == "R10_event_id_conflict":
                first = next(e for e in stream if e["event_type"] == "step_completed")
                tampered = json.loads(json.dumps(first["payload"]))
                tampered["actual"]["grid_import_energy_kwh"] = "999.000000"
                emit("step_completed", 900, tampered, version=1, fp=fp1,
                     measured=900, received=915, business=0, asset="M-PCC", event_id=first["event_id"])
            if sid == "R21_replan_exhausted":
                for j in range(3):
                    emit("replan_requested", 920 + j * 10, {"attempt": j + 1, "quality": "good",
                                                        "valid_until": at(3600)})
            if sid == "R22_remaining_infeasible":
                emit("booking_changed", 920, {"remaining_energy_kwh": "500.000000",
                                              "port_max_kw": "240.000000", "deadline_at": at(1800),
                                              "quality": "synthetic", "valid_until": at(1800)}, asset="BR-CHARGE")
        actual_state = replay(stream)
        result = {"scenario_id": sid, "expected_status": expected, "actual_status": actual_state["status"],
                  "match": actual_state["status"] == expected,
                  "executed_prefix_steps": len(actual_state["executed"]),
                  "duplicates": actual_state["duplicates"], "conflicts": actual_state["conflicts"],
                  "stale_receipts": actual_state["stale_receipts"],
                  "double_occupancy_blocks": actual_state["double_occupancy_blocks"],
                  "rejections": actual_state["rejections"]}
        results.append(result)
        manifests.append({"scenario_id": sid, "station_id": STATION, "task_id": task_id,
                          "business_window": [at(0), at(3600)], "event_order": [e["event_id"] for e in sorted(stream, key=lambda r: (r["received_at"], r["ordinal"]))],
                          "expected_status": expected, "expected_invariant": invariant,
                          "truth_ref": "execution_truth.jsonl|event_truth.jsonl", "source_scenario_id": base_id,
                          "is_simulated": True, "source_version": DATASET})
        for e in stream:
            if e["event_type"] in {"measurement", "forecast_issued", "booking_changed", "derating",
                                   "protection", "manual_takeover", "device_frozen", "connection_lost",
                                   "connection_restored", "replan_requested"}:
                event_truth.append({"scenario_id": sid, "station_id": STATION, "asset_id": e["asset_id"],
                                    "source_event_id": e["event_id"], "state_kind": e["event_type"],
                                    "state_effective_at": e["business_time"],
                                    "truth_available_at": e["received_at"], "raw_value": e["payload"],
                                    "quality": e["payload"]["quality"], "is_simulated": True,
                                    "source_version": "rolling_event_fixture_state_truth_v1"})
        all_events.extend(stream)
    assert all(r["match"] for r in results), [r for r in results if not r["match"]]
    dump_jsonl("events.jsonl", all_events)
    dump_jsonl("execution_truth.jsonl", all_truth)
    dump_jsonl("event_truth.jsonl", event_truth)
    dump_json("scenario_manifest.json", manifests)
    outputs = ["events.jsonl", "execution_truth.jsonl", "event_truth.jsonl", "scenario_manifest.json"]
    assert {name: sha(DATA / name) for name in CFG["source_hashes_sha256"]} == CFG["source_hashes_sha256"]
    dump_json("validation_report.json", {"status": "generated_pending_independent_validation", "dataset_id": DATASET,
            "is_simulated": True, "scenario_count": len(specs), "event_count": len(all_events),
            "execution_truth_count": len(all_truth), "event_truth_count": len(event_truth),
            "source_hashes_sha256": CFG["source_hashes_sha256"],
            "output_hashes_sha256": {name: sha(HERE / name) for name in outputs},
            "scenario_results": results,
            "limitations": "Synthetic workflow fixture; results do not certify optimizer, physical station, or real alarms"})


if __name__ == "__main__":
    main()
