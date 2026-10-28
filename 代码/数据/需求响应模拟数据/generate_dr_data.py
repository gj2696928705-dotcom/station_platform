"""Technical demand-response fixtures with frozen no-response plans and no settlement."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CFG = json.loads((HERE / "simulation_config.json").read_text(encoding="utf-8"))
STATION = CFG["station_id"]
DATASET = CFG["dataset_id"]
START = datetime.fromisoformat(CFG["baseline_frozen_at"])
STEP = timedelta(minutes=15)
Q = Decimal("0.000001")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_csv(name: str, rows: list[dict]) -> None:
    with (HERE / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(name: str, obj: object) -> None:
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(name: str, rows: list[dict]) -> None:
    (HERE / name).write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for r in rows), encoding="utf-8")


def fmt(n: Decimal | int | float) -> str:
    return str(Decimal(str(n)).quantize(Q))


def at(minutes: int) -> str:
    return (START + timedelta(minutes=minutes)).isoformat(timespec="seconds")


def main() -> None:
    assert all(sha(ROOT / name) == h for name, h in CFG["source_hashes_sha256"].items())
    source_id = CFG["source_scenario_id"]
    scenario = next(r for r in read(ROOT / "代码/数据/场站联动模拟数据/scenario_inputs.csv") if r["scenario_id"] == source_id)
    truth = sorted((r for r in read(ROOT / "代码/数据/场站联动模拟数据/scenario_truth.csv") if r["scenario_id"] == source_id), key=lambda r: r["timestamp"])
    forecast = sorted((r for r in read(ROOT / "代码/数据/场站联动模拟数据/forecast_values.csv") if r["scenario_id"] == source_id and r["resource"] == "load"), key=lambda r: r["timestamp"])
    assert len(truth) == len(forecast) == 4 and scenario["window_start"] == at(0)
    assert all(datetime.fromisoformat(r["source_observation_end"]) <= START for r in forecast)
    assert all(datetime.fromisoformat(r["available_at"]) > START for r in truth)
    d02_truth = [r for r in load_jsonl(ROOT / "代码/数据/滚动事件模拟数据/execution_truth.jsonl") if r["scenario_id"] == "R01_normal"]
    assert len(d02_truth) == 4
    for k, r in enumerate(d02_truth):
        assert Decimal(r["actual"]["base_energy_kwh"]) == (Decimal(truth[k]["base_load_kw"]) / 4).quantize(Q)
        assert sum(map(Decimal, r["actual"]["vehicle_energy_kwh"].values())) == Decimal(20)
    storage = read(ROOT / "代码/数据/储能参数与初始状态/储能参数.csv")[0]
    assert storage["site_dispatch_ready"] == "false"

    specs = [
        ("DR01_normal_down", "down", 40, (1, 3), "activated_technical", "R01_normal"),
        ("DR02_up_unavailable", "up", 1000, (1, 3), "reject_up_capacity", "R06_grid_derate"),
        ("DR03_down_unavailable", "down", 100, (1, 3), "reject_down_capacity", "R14_device_frozen"),
        ("DR04_sustained_service", "down", 80, (1, 4), "reject_service_deadline", "R04_soc_drift"),
        ("DR05_recovery_rebound", "down", 40, (1, 3), "reject_recovery_capacity", "R06_grid_derate"),
        ("DR06_partial_activation", "down", 40, (1, 3), "partial_technical", "R01_normal"),
        ("DR07_cancelled", "down", 40, (1, 3), "cancelled_before_start", "R07_cancel_ack_late"),
        ("DR08_baseline_tamper", "down", 40, (1, 3), "reject_baseline_fingerprint", "R10_event_id_conflict"),
        ("DR09_missing_meter", "down", 40, (1, 3), "evidence_incomplete", "R15_poor_measurement"),
        ("DR10_late_receipt", "down", 40, (1, 3), "reject_late_ack", "R09_measurement_before_receipt"),
    ]
    d02_event_ids = {}
    d02_events = {}
    for event in load_jsonl(ROOT / "代码/数据/滚动事件模拟数据/events.jsonl"):
        d02_event_ids.setdefault(event["scenario_id"], []).append(event["event_id"])
        d02_events.setdefault(event["scenario_id"], []).append(event)
    baseline_rows: list[dict] = []
    offers: list[dict] = []
    events: list[dict] = []
    actual_rows: list[dict] = []
    manifests: list[dict] = []
    mutations: list[dict] = []
    for sid, direction, requested_kw, (first, end), expected, d02_sid in specs:
        preferred_event = {
            "DR02_up_unavailable": "derating", "DR03_down_unavailable": "device_frozen",
            "DR04_sustained_service": "step_completed", "DR05_recovery_rebound": "derating",
            "DR09_missing_meter": "measurement", "DR10_late_receipt": "step_completed",
        }.get(sid)
        d02_reference = next((e for e in d02_events[d02_sid] if e["event_type"] == preferred_event), d02_events[d02_sid][0])
        no_flex = sid == "DR03_down_unavailable"
        baseline_vehicle = {v: (0 if no_flex else 40) for v in CFG["technical_assumptions"]["vehicle_ids"]}
        commitment = 0 if no_flex else 40
        frozen_content = []
        for k in range(4):
            vehicle_kw = {v: fmt(p) for v, p in baseline_vehicle.items()}
            existing_kw = Decimal(CFG["technical_assumptions"]["existing_committed_charge_from_D02_R01_kw"])
            net_kw = Decimal(forecast[k]["point_kw"]) + existing_kw + sum(map(Decimal, vehicle_kw.values()))
            frozen_content.append({"step_index": k, "timestamp": at(k * 15), "vehicle_charge_kw": vehicle_kw,
                                   "existing_committed_charge_kw": fmt(existing_kw),
                                   "storage_charge_kw": "0.000000", "storage_discharge_kw": "0.000000",
                                   "storage_energy_after_kwh": "2400.000000", "pv_used_kw": "0.000000",
                                   "net_import_kw": fmt(net_kw), "committed_ac_energy_kwh_each": fmt(commitment),
                                   "source_load_vintage_id": scenario["load_vintage_id"]})
        frozen_fp = digest({"scenario_id": sid, "frozen_at": at(0), "seed": CFG["seed"],
                            "rule": CFG["technical_fixture_rule_version"], "steps": frozen_content})
        for k, content in enumerate(frozen_content):
            baseline_rows.append({"station_id": STATION, "scenario_id": sid, "dataset_id": DATASET,
                                  "is_simulated": "true", "source_versions": "D01_past_only|D02_workflow_stub|technical_fixture_v1",
                                  "frozen_at": at(0), "as_of": at(0), "baseline_rule_version": CFG["technical_fixture_rule_version"],
                                  "market_baseline_rule_version": "unknown", "plan_version": 1,
                                  "plan_fingerprint": frozen_fp, "step_index": k,
                                  "timestamp": content["timestamp"], "interval_end": at((k + 1) * 15),
                                  "source_observation_end": forecast[k]["source_observation_end"],
                                  "base_load_forecast_kw": forecast[k]["point_kw"],
                                  "existing_committed_charge_kw": content["existing_committed_charge_kw"],
                                  "existing_charge_status": "synthetic_known_D02_R01_service",
                                  "vehicle_charge_kw_json": json.dumps(content["vehicle_charge_kw"], sort_keys=True),
                                  "committed_ac_energy_kwh_each": fmt(commitment),
                                  "storage_charge_kw": content["storage_charge_kw"],
                                  "storage_discharge_kw": content["storage_discharge_kw"],
                                  "storage_energy_after_kwh": content["storage_energy_after_kwh"],
                                  "pv_used_kw": content["pv_used_kw"],
                                  "net_import_kw": content["net_import_kw"],
                                  "grid_import_limit_kw": "10000.000000", "grid_export_limit_kw": "0.000000"})
        response_end = at(end * 15)
        offer_id = f"OFFER-{sid}"
        requested_energy = Decimal(requested_kw) * Decimal((end - first) * 15) / 60
        recovery_cap = 120 if sid == "DR05_recovery_rebound" else 240
        offers.append({"station_id": STATION, "scenario_id": sid, "dataset_id": DATASET,
                       "is_simulated": "true", "source_version": "technical_offer_v1",
                       "offer_id": offer_id, "plan_version": 1, "plan_fingerprint": frozen_fp,
                       "direction": direction, "response_start": at(first * 15), "response_end": response_end,
                       "requested_power_kw": fmt(requested_kw), "requested_energy_kwh": fmt(requested_energy),
                       "notice_lead_minutes": "unknown", "fixture_notice_lead_minutes": 15,
                       "duration_minutes": (end - first) * 15, "recovery_start": response_end,
                       "recovery_end": at(60), "recovery_fleet_power_cap_kw": fmt(recovery_cap),
                       "authorized_resources_json": json.dumps(CFG["technical_assumptions"]["vehicle_ids"] if not no_flex else []),
                       "baseline_rule_version": "unknown", "assessment_rule_version": "unknown",
                       "technical_fixture_rule_version": CFG["technical_fixture_rule_version"],
                       "offered_at": at(0), "available_at": at(0), "market_status": "not_applicable"})

        def event(kind: str, minute: int, payload: dict, *, received_minute: int | None = None,
                  eid: str | None = None, fp: str | None = None) -> dict:
            visibility_minute = minute if received_minute is None else received_minute
            row = {"station_id": STATION, "scenario_id": sid, "dataset_id": DATASET,
                   "is_simulated": True, "source_version": "technical_activation_v1",
                   "event_id": eid or f"{sid}-A{len([e for e in events if e['scenario_id'] == sid]):02d}",
                   "event_type": kind, "offer_id": offer_id, "plan_version": 1,
                   "plan_fingerprint": frozen_fp if fp is None else fp,
                   "business_time": at(minute), "available_at": at(visibility_minute),
                   "received_at": at(visibility_minute),
                   "source": "simulated_aggregator" if kind in ("offer_received", "activation_started", "activation_ended", "cancelled") else "simulated_station",
                   "source_ref": f"D02:{d02_sid}:{d02_reference['event_id']}",
                   "payload": payload}
            events.append(row)
            return row

        event("offer_received", 0, {"quality": "synthetic", "valid_until": at(60)})
        if sid == "DR07_cancelled":
            event("cancelled", 10, {"reason": "before_start", "quality": "good", "valid_until": at(60)})
        elif sid == "DR08_baseline_tamper":
            changed = json.loads(json.dumps(frozen_content))
            changed[1]["vehicle_charge_kw"]["EV-DR-01"] = "80.000000"
            bad_fp = digest({"scenario_id": sid, "frozen_at": at(0), "seed": CFG["seed"],
                             "rule": CFG["technical_fixture_rule_version"], "steps": changed})
            mutations.append({"station_id": STATION, "scenario_id": sid, "is_simulated": True,
                              "source_version": "tamper_probe_v1", "baseline_fingerprint": frozen_fp,
                              "attempted_fingerprint": bad_fp, "attempted_at": at(5),
                              "mutation": "EV-DR-01 slot1 40kW -> 80kW", "expected": "reject"})
            event("activation_acknowledged", 5, {"accepted": False, "reason": "baseline_fingerprint_mismatch",
                                                 "quality": "good", "valid_until": at(60)}, fp=bad_fp)
        elif sid == "DR10_late_receipt":
            event("activation_acknowledged", 5, {"accepted": False, "reason": "late_receipt",
                                                 "quality": "good", "valid_until": at(60)}, received_minute=20)
        elif expected.startswith("reject"):
            event("activation_acknowledged", 5, {"accepted": False, "reason": expected,
                                                 "quality": "good", "valid_until": at(60)})
        else:
            ack = event("activation_acknowledged", 5, {"accepted": True, "quality": "good", "valid_until": at(60)})
            if sid == "DR01_normal_down":
                event("activation_acknowledged", 5, ack["payload"], received_minute=6, eid=ack["event_id"])
            active_start = 30 if sid == "DR06_partial_activation" else first * 15
            event("activation_started", active_start, {"active_fraction": "0.5" if sid == "DR06_partial_activation" else "1.0",
                                                       "quality": "good", "valid_until": response_end})
            event("activation_ended", end * 15, {"quality": "good", "valid_until": at(60)})

        # Actual station truth is post-interval D01/D02 observation plus an explicit vehicle intervention.
        for k in range(4):
            vehicle_kw = dict(baseline_vehicle)
            if sid in ("DR01_normal_down", "DR09_missing_meter"):
                if k in (1, 2):
                    vehicle_kw["EV-DR-02"] = 0
                if k == 3:
                    vehicle_kw["EV-DR-02"] = 120
            elif sid == "DR06_partial_activation":
                if k == 2:
                    vehicle_kw["EV-DR-02"] = 0
                if k == 3:
                    vehicle_kw["EV-DR-02"] = 80
            base = Decimal(truth[k]["base_load_kw"])
            historic_charge = Decimal(truth[k]["historical_charge_kw"])
            vehicle_total = sum(map(Decimal, vehicle_kw.values()), Decimal(0))
            existing_kw = Decimal(CFG["technical_assumptions"]["existing_committed_charge_from_D02_R01_kw"])
            grid_kw = base + historic_charge + existing_kw + vehicle_total
            assert 0 <= grid_kw <= 10000
            quality = "unknown" if sid == "DR09_missing_meter" and k == 1 else "good"
            device_state = {"storage_site_dispatch_ready": False, "grid_import_limit_kw": "10000.000000",
                            "fleet_recovery_cap_kw": fmt(recovery_cap), "meter_quality": quality,
                            "actual_device_condition": "storage_frozen" if sid == "DR03_down_unavailable" else "normal"}
            actual_rows.append({"station_id": STATION, "scenario_id": sid, "dataset_id": DATASET,
                                "is_simulated": "true", "source_versions": "D01_truth|D02_execution_truth|synthetic_dr_intervention_v1",
                                "timestamp": at(k * 15), "interval_end": at((k + 1) * 15),
                                "available_at": at((k + 1) * 15 + 5), "plan_fingerprint": frozen_fp,
                                "activation_status": expected,
                                "base_energy_kwh": fmt(base / 4),
                                "historical_charge_energy_kwh": fmt(historic_charge / 4),
                                "existing_committed_charge_energy_kwh": fmt(existing_kw / 4),
                                "vehicle_charge_kw_json": json.dumps({v: fmt(p) for v, p in vehicle_kw.items()}, sort_keys=True),
                                "vehicle_ac_energy_kwh_json": json.dumps({v: fmt(Decimal(p) / 4) for v, p in vehicle_kw.items()}, sort_keys=True),
                                "pv_available_energy_kwh_approx": truth[k]["pv_interval_energy_kwh_approx"],
                                "pv_used_energy_kwh": "0.000000", "storage_charge_energy_ac_kwh": "0.000000",
                                "storage_discharge_energy_ac_kwh": "0.000000",
                                "storage_energy_after_kwh": "2400.000000", "soc_after_percent": "60.000000",
                                "grid_import_energy_kwh": "" if quality == "unknown" else fmt(grid_kw / 4),
                                "grid_export_energy_kwh": "0.000000", "measurement_quality": quality,
                                "device_state_json": json.dumps(device_state, sort_keys=True),
                                "d02_event_ref": d02_reference["event_id"],
                                "d02_event_applicability": "workflow_pattern_only",
                                "received_at": at((k + 1) * 15 + (20 if sid == "DR10_late_receipt" else 5))})
        manifests.append({"station_id": STATION, "scenario_id": sid, "dataset_id": DATASET,
                          "is_simulated": True, "source_version": DATASET, "offer_id": offer_id,
                          "expected_status": expected, "baseline_fingerprint": frozen_fp,
                          "source_d02_scenario_id": d02_sid, "source_d02_event_id": d02_reference["event_id"],
                          "expected_constraint": {
                              "DR01_normal_down": "40kW reduction for two slots, 80kW rebound in final slot, full service preserved",
                              "DR02_up_unavailable": "1000kW increase exceeds vehicle headroom and storage is not dispatch-ready",
                              "DR03_down_unavailable": "no flexible vehicle/PV/storage resource",
                              "DR04_sustained_service": "three-slot reduction leaves committed energy before deadline unmet",
                              "DR05_recovery_rebound": "recovery fleet cap 120kW is below 160kW needed",
                              "DR06_partial_activation": "one active slot and 40kW recovery increment",
                              "DR07_cancelled": "no activation after pre-start cancellation",
                              "DR08_baseline_tamper": "mutated fingerprint rejected; frozen baseline remains",
                              "DR09_missing_meter": "missing grid energy remains unknown; no performance score",
                              "DR10_late_receipt": "ack received after response start; no activation",
                          }[sid],
                          "settlement_status": "not_applicable", "market_rule_status": "unknown"})

    write_csv("baseline_plan.csv", baseline_rows)
    write_csv("response_offers.csv", offers)
    write_jsonl("activation_events.jsonl", events)
    write_csv("execution_truth.csv", actual_rows)
    write_jsonl("baseline_mutation_attempts.jsonl", mutations)
    write_json("scenario_manifest.json", manifests)
    write_json("source_manifest.json", [{"path": p, "sha256": h, "read_only": True,
                                         "station_id": STATION, "scenario_id": "PACKAGE", "is_simulated": True}
                                        for p, h in sorted(CFG["source_hashes_sha256"].items())])
    write_json("rule_sources.json", {"station_id": STATION, "scenario_id": "PACKAGE", "is_simulated": True,
             "market_rule_status": "unknown", "technical_fixture_rule": CFG["technical_fixture_rule_version"],
             "reviewed_sources": ["代码/后续算法会话交接/D04_需求响应模拟数据.md",
                                  "代码/后续算法会话交接/A09_需求响应可调能力与履约.md"],
             "finding": "No verifiable market baseline, assessment tolerance, settlement or authorization rule supplied; technical test only"})
    write_json("settlement_reference.json", {"station_id": STATION, "scenario_id": "PACKAGE", "is_simulated": True,
             "status": "not_applicable", "method": "unknown", "allowed_deviation": "unknown",
             "payment_amount": None, "performance_passed": None,
             "reason": "No business baseline and assessment rule; technical energy checks are not settlement"})
    outputs = ["baseline_plan.csv", "response_offers.csv", "activation_events.jsonl", "execution_truth.csv",
               "baseline_mutation_attempts.jsonl", "scenario_manifest.json", "source_manifest.json",
               "rule_sources.json", "settlement_reference.json"]
    assert all(sha(ROOT / name) == h for name, h in CFG["source_hashes_sha256"].items())
    write_json("validation_report.json", {"dataset_id": DATASET, "status": "generated_pending_independent_validation",
               "is_simulated": True, "execution_mode": "offline_technical_range_test",
               "source_hashes_sha256": CFG["source_hashes_sha256"],
               "output_hashes_sha256": {n: sha(HERE / n) for n in outputs},
               "counts": {"scenarios": len(manifests), "baseline_intervals": len(baseline_rows),
                          "offers": len(offers), "activation_events": len(events),
                          "execution_intervals": len(actual_rows), "tamper_attempts": len(mutations)},
               "settlement_status": "not_applicable", "market_rule_status": "unknown"})


if __name__ == "__main__":
    main()
