"""Independent arithmetic, provenance, visibility, and replay audit of D02 fixtures."""
from __future__ import annotations

import hashlib
import json
import csv
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from generate_rolling_events import replay

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
CFG = json.loads((HERE / "simulation_config.json").read_text(encoding="utf-8"))
TOL = Decimal("0.000003")
REQUIRED = ("scenario_id", "event_id", "event_type", "station_id", "asset_id", "business_time",
            "measured_at", "available_at", "received_at", "task_id", "plan_version",
            "plan_fingerprint", "payload", "is_simulated", "source_ref")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dt(value: str) -> datetime:
    out = datetime.fromisoformat(value)
    assert out.utcoffset() == timedelta(hours=8)
    return out


def close(a: Decimal, b: Decimal) -> None:
    assert abs(a - b) <= TOL, f"energy mismatch {a} / {b}"


def digest(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    report = json.loads((HERE / "validation_report.json").read_text(encoding="utf-8"))
    assert report["dataset_id"] == CFG["dataset_id"]
    for name, expected in CFG["source_hashes_sha256"].items():
        assert sha(DATA / name) == expected
    for name, expected in report["output_hashes_sha256"].items():
        assert sha(HERE / name) == expected
    events = records(HERE / "events.jsonl")
    truths = records(HERE / "execution_truth.jsonl")
    event_truth = records(HERE / "event_truth.jsonl")
    manifest = json.loads((HERE / "scenario_manifest.json").read_text(encoding="utf-8"))
    by_sid = defaultdict(list)
    with (DATA / "场站联动模拟数据/forecast_values.csv").open(encoding="utf-8-sig", newline="") as stream:
        source_forecasts = [r for r in csv.DictReader(stream) if r["scenario_id"] == CFG["base_scenario_id"]]
    expected_points = {resource: [r["point_kw"] for r in sorted(source_forecasts, key=lambda r: r["timestamp"]) if r["resource"] == resource]
                       for resource in ("load", "pv")}
    with (DATA / "场站联动模拟数据/booking_events.csv").open(encoding="utf-8-sig", newline="") as stream:
        source_bookings = [r for r in csv.DictReader(stream) if r["scenario_id"] == CFG["base_scenario_id"] and r["event_type"] == "booking"][:2]
    truth_map = {}
    seen_content = {}
    for t in truths:
        assert t["is_simulated"] is True and t["station_id"] == CFG["station_id"]
        assert dt(t["available_at"]) >= dt(t["actual"]["interval_end"])
        key = (t["scenario_id"], t["plan_version"], t["step_index"])
        assert key not in truth_map
        truth_map[key] = t
    for e in events:
        assert all(k in e for k in REQUIRED)
        assert e["is_simulated"] is True and e["station_id"] == CFG["station_id"]
        assert e["event_id"] and e["available_at"] and e["received_at"] and e["source_ref"]
        assert dt(e["business_time"]) <= dt(e["available_at"]) <= dt(e["received_at"])
        assert e["available_at"] == e["received_at"]
        if e["measured_at"]:
            assert dt(e["business_time"]) <= dt(e["measured_at"]) <= dt(e["available_at"])
        assert "quality" in e["payload"] and "valid_until" in e["payload"]
        semantic = {k: v for k, v in e.items() if k not in ("received_at", "available_at", "ordinal")}
        old_content = seen_content.setdefault((e["scenario_id"], e["event_id"]), digest(semantic))
        is_conflicting_retransmission = old_content != digest(semantic)
        if e["event_type"] == "measurement" and e["payload"]["quality"] == "unknown":
            assert e["payload"]["base_kw"] is None and e["payload"]["soc_percent"] is None
        if e["event_type"] == "forecast_issued":
            n = e["payload"]["coverage_count"]
            assert len(e["payload"]["load_point_kw"]) == len(e["payload"]["pv_point_kw"]) == n
            assert e["payload"]["load_point_kw"] == expected_points["load"][:n]
            assert e["payload"]["pv_point_kw"] == expected_points["pv"][:n]
        if e["event_type"] == "booking_changed" and "visible_bookings" in e["payload"]:
            expected_bookings = [{"session_id": r["session_id"], "port_id": r["port_id"],
                                  "committed_energy_kwh": r["committed_energy_kwh"], "available_at": r["available_at"]}
                                 for r in source_bookings]
            assert e["payload"]["visible_bookings"] == expected_bookings
            assert all(dt(r["available_at"]) <= dt(e["received_at"]) for r in e["payload"]["visible_bookings"])
        if e["event_type"] == "plan_sent":
            assert e["plan_fingerprint"] == digest(e["payload"]["plan"])
            assert e["plan_version"] == e["payload"]["plan"]["version"]
            assert e["task_id"] == e["payload"]["plan"]["task_id"]
        if e["event_type"] == "step_completed" and not is_conflicting_retransmission:
            a = e["payload"]["actual"]
            left = Decimal(a["grid_import_energy_kwh"]) + Decimal(a["pv_used_energy_kwh"]) + Decimal(a["storage_discharge_energy_ac_kwh"])
            right = Decimal(a["base_energy_kwh"]) + Decimal(a["historical_charge_energy_kwh"]) + sum(map(Decimal, a["vehicle_energy_kwh"].values())) + Decimal(a["storage_charge_energy_ac_kwh"])
            close(left, right)
            expected_energy = Decimal(a["storage_energy_before_kwh"]) + Decimal("0.95") * Decimal(a["storage_charge_energy_ac_kwh"]) - Decimal(a["storage_discharge_energy_ac_kwh"]) / Decimal("0.95")
            close(Decimal(a["storage_energy_after_kwh"]), expected_energy)
            close(Decimal(a["soc_after_percent"]), Decimal(a["storage_energy_after_kwh"]) / 40)
            assert Decimal(a["storage_charge_kw"]) == 0 or Decimal(a["storage_discharge_kw"]) == 0
            assert Decimal(a["grid_import_energy_kwh"]) >= 0 and Decimal(a["grid_import_energy_kwh"]) <= 2500
            key = (e["scenario_id"], e["plan_version"], a["step_index"])
            assert key in truth_map and truth_map[key]["actual"] != {}  # actual record is independently stored
        by_sid[e["scenario_id"]].append(e)
    indexed_events = {(e["scenario_id"], e["event_id"]): e for e in events}
    for t in event_truth:
        e = indexed_events[(t["scenario_id"], t["source_event_id"])]
        assert t["is_simulated"] is True and t["asset_id"] == e["asset_id"]
        assert t["state_kind"] == e["event_type"] and t["raw_value"] == e["payload"]
        assert dt(t["state_effective_at"]) <= dt(t["truth_available_at"])
        assert t["truth_available_at"] == e["received_at"]
    assert len(by_sid) == len(manifest) == 22
    checks = []
    for sc in manifest:
        sid = sc["scenario_id"]
        stream = sorted(by_sid[sid], key=lambda e: (e["received_at"], e["ordinal"]))
        assert sc["event_order"] == [e["event_id"] for e in stream]
        as_of = dt(sc["business_window"][0])
        visible = [e for e in stream if dt(e["available_at"]) <= as_of]
        assert all(e["event_type"] != "step_completed" for e in visible)
        assert all(dt(t["available_at"]) > as_of for t in truths if t["scenario_id"] == sid)
        state1 = replay(stream)
        state2 = replay(json.loads(json.dumps(stream)))
        assert state1 == state2
        assert state1["status"] == sc["expected_status"]
        assert state1["status"] == next(r["actual_status"] for r in report["scenario_results"] if r["scenario_id"] == sid)
        if sid == "R02_duplicate":
            assert state1["duplicates"] >= 1 and len(state1["executed"]) == 4
        if sid == "R03_rejected":
            assert not state1["executed"]
        if sid == "R07_cancel_ack_late":
            assert state1["double_occupancy_blocks"] == 1
        if sid in ("R08_old_receipt", "R09_measurement_before_receipt"):
            assert state1["stale_receipts"] == 1 and state1["current_version"] == 2
        if sid == "R10_event_id_conflict":
            assert state1["conflicts"] == 1 and len(state1["executed"]) == 1
        if sid in ("R12_protection", "R13_manual_takeover", "R14_device_frozen", "R19_connection_lost"):
            assert state1["current_version"] == 0 and state1["status"].startswith("blocked")
        if sid == "R15_poor_measurement":
            assert state1["current_version"] == 0
        if sid == "R18_forecast_gap":
            assert state1["current_version"] == 0
        if sid == "R22_remaining_infeasible":
            assert state1["blocked"]
        for e in stream:
            if e["event_type"] == "plan_sent" and e["plan_version"] == 2:
                frozen = e["payload"]["plan"]["frozen_prefix_fingerprint"]
                assert frozen == digest({"0": truth_map[(sid, 1, 0)]["actual"]})
        checks.append({"scenario_id": sid, "expected_status": sc["expected_status"],
                       "actual_status": state1["status"], "difference": "", "replay_equal": True,
                       "visible_at_as_of": len(visible), "executed_prefix_steps": len(state1["executed"])})
    report["status"] = "passed"
    report["independent_validation"] = {"all_event_fields_and_time_order": True,
                                        "all_step_energy_balances": True, "all_storage_updates": True,
                                        "future_truth_hidden_at_as_of": True, "replayed_twice_identically": True,
                                        "separate_event_state_truth_checked": len(event_truth),
                                        "scenario_checks": checks}
    (HERE / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "scenarios": len(manifest), "events": len(events), "truth_steps": len(truths)}))


if __name__ == "__main__":
    main()
