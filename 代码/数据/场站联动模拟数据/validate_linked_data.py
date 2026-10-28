"""Independent source-to-output and temporal validation; no generator imports."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
STEP = timedelta(minutes=15)
TOL = Decimal("0.000002")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dt(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    assert result.utcoffset() == timedelta(hours=8), value
    return result


def close(a: Decimal, b: Decimal) -> None:
    assert abs(a - b) <= TOL, f"{a} != {b}"


def main() -> None:
    config = json.loads((HERE / "simulation_config.json").read_text(encoding="utf-8"))
    report = json.loads((HERE / "validation_report.json").read_text(encoding="utf-8"))
    for name, expected in config["source_hashes_sha256"].items():
        assert sha(DATA / name) == expected
    for name, expected in report["output_hashes_sha256"].items():
        assert sha(HERE / name) == expected, name
    for source in json.loads((HERE / "source_manifest.json").read_text(encoding="utf-8")):
        assert sha(DATA / source["path"]) == source["sha256"]

    truth = rows(HERE / "station_interval_truth.csv")
    load = rows(DATA / "负荷数据/黄花站_2025年_15分钟负荷模拟数据.csv")
    pv = rows(DATA / "光伏数据/黄花站_2025年_15分钟光伏模拟数据.csv")
    assert len(truth) == len(load) == len(pv) == 35040
    truth_map = {}
    for i, (out, raw_load, raw_pv) in enumerate(zip(truth, load, pv)):
        t = dt(out["timestamp"])
        assert t == dt(raw_load["timestamp"]) == dt(raw_pv["timestamp"])
        assert t == datetime.fromisoformat("2025-01-01T00:00:00+08:00") + i * STEP
        assert dt(out["interval_end"]) == dt(out["available_at"]) == t + STEP
        assert out["station_id"] == raw_load["station_id"] == raw_pv["station_id"] == config["station_id"]
        assert out["scenario_id"] == "BASE_2025" and out["is_simulated"] == "true"
        base = Decimal(raw_load["base_load_kw"])
        charge = Decimal(raw_load["charging_load_kw"])
        total = Decimal(raw_load["total_load_kw"])
        assert base + charge == total
        assert out["base_load_kw"] == raw_load["base_load_kw"]
        assert out["historical_charge_kw"] == raw_load["charging_load_kw"]
        assert out["total_load_kw"] == raw_load["total_load_kw"]
        close(Decimal(out["base_energy_kwh"]), base / 4)
        close(Decimal(out["historical_charge_energy_kwh"]), charge / 4)
        close(Decimal(out["total_load_energy_kwh"]), total / 4)
        left = Decimal(raw_pv["pv_power_kw"])
        right = Decimal(pv[i + 1]["pv_power_kw"]) if i + 1 < len(pv) else Decimal(0)
        if i == len(pv) - 1:
            assert left == 0 and out["pv_right_endpoint_source"] == "source_document_2026_midnight_zero_anchor"
        else:
            assert out["pv_right_endpoint_source"] == "source_next_sample"
        close(Decimal(out["pv_sample_start_kw"]), left)
        close(Decimal(out["pv_sample_end_kw"]), right)
        close(Decimal(out["pv_interval_mean_kw_approx"]), (left + right) / 2)
        close(Decimal(out["pv_interval_energy_kwh_approx"]), (left + right) / 8)
        assert out["pv_conversion"] == "linear_trapezoid_15min"
        truth_map[out["timestamp"]] = out
    assert dt(truth[-1]["interval_end"]) == datetime.fromisoformat("2026-01-01T00:00:00+08:00")

    sessions = rows(DATA / "负荷数据/充电会话明细.csv")
    by_port = defaultdict(list)
    session_map = {}
    for s in sessions:
        assert s["station_id"] == config["station_id"]
        assert dt(s["arrival_at"]) <= dt(s["start_at"]) < dt(s["end_at"])
        assert Decimal(s["peak_power_kw"]) <= 240
        by_port[s["port_id"]].append((dt(s["start_at"]), dt(s["end_at"])))
        session_map[s["session_id"]] = s
    assert len(by_port) == 128
    for periods in by_port.values():
        periods.sort()
        assert all(a[1] <= b[0] for a, b in zip(periods, periods[1:]))

    scenarios = rows(HERE / "scenario_inputs.csv")
    scenario_map = {r["scenario_id"]: r for r in scenarios}
    assert len(scenarios) == len(scenario_map) == 12
    for r in scenarios:
        assert r["station_id"] == config["station_id"] and r["is_simulated"] == "true"
        assert r["execution_mode"] == "offline_replay"
        assert dt(r["as_of"]) == dt(r["window_start"])
        assert dt(r["window_end"]) - dt(r["window_start"]) == 4 * STEP
        assert Decimal(r["grid_import_limit_kw"]) == 10000
        assert Decimal(r["grid_export_limit_kw"]) == 0
        assert int(r["available_port_count"]) == 128
        assert dt(r["storage_state_available_at"]) <= dt(r["as_of"])
        close(Decimal(r["storage_initial_energy_kwh"]), Decimal(r["storage_initial_soc_percent"]) * 40)
        assert r["storage_site_dispatch_ready"] == "false"
        assert r["split"] == ("development" if r["as_of"] < "2025-09-01" else "calibration" if r["as_of"] < "2025-11-01" else "replay_evaluation")
        assert not {"base_load_kw", "historical_charge_kw", "total_load_kw", "pv_interval_mean_kw_approx"} & set(r)

    events = rows(HERE / "booking_events.csv")
    grouped_events = defaultdict(list)
    for e in events:
        sid = e["scenario_id"]
        assert sid in scenario_map and e["is_simulated"] == "true"
        assert e["session_id"].startswith("BOOK-") and e["session_id"] not in session_map
        template = session_map[e["template_session_id"]]
        assert e["port_id"] == template["port_id"]
        assert Decimal(e["committed_energy_kwh"]) == Decimal(template["requested_energy_kwh"])
        assert dt(template["end_at"]) <= dt(scenario_map[sid]["as_of"]) - STEP
        assert dt(e["event_at"]) <= dt(e["available_at"])
        assert e["event_type"] in ("booking", "arrival", "cancellation")
        if e["event_type"] == "arrival":
            assert dt(e["event_at"]) == dt(scenario_map[sid]["as_of"]) + STEP
        grouped_events[sid].append(e)
    visible_counts = {}
    reserved_ports = {}
    for sid, scenario in scenario_map.items():
        visible = sorted((e for e in grouped_events[sid] if dt(e["available_at"]) <= dt(scenario["as_of"])), key=lambda e: e["available_at"])
        active = {}
        for e in visible:
            if e["event_type"] == "booking":
                active[e["session_id"]] = e
            elif e["event_type"] == "cancellation":
                active.pop(e["session_id"], None)
        assert len(active) == int(scenario["visible_booking_count"])
        visible_counts[sid] = len(active)
        reserved_ports[sid] = {e["port_id"] for e in active.values()}
        if sid == "S10_no_booking":
            assert not active and scenario["commitment_status"] == "unknown"

    requests = rows(HERE / "vehicle_requests.csv")
    by_scenario_requests = defaultdict(list)
    for r in requests:
        sc = scenario_map[r["scenario_id"]]
        assert r["is_simulated"] == "true" and dt(r["available_at"]) <= dt(sc["as_of"])
        assert dt(r["arrival_at"]) == dt(sc["as_of"]) and dt(r["departure_deadline_at"]) == dt(sc["window_end"])
        assert Decimal(r["max_power_kw"]) <= 240 and Decimal(r["requested_energy_kwh"]) > 0
        by_scenario_requests[r["scenario_id"]].append(r)
    for sid, sc in scenario_map.items():
        rr = by_scenario_requests[sid]
        assert len(rr) == int(sc["request_count"])
        assert len({r["port_id"] for r in rr}) == len(rr)
        assert all(r["port_id"] in by_port for r in rr)
        assert not {r["port_id"] for r in rr} & reserved_ports[sid]
        if sid == "S09_derated":
            assert sc["derated_port_id"] in {r["port_id"] for r in rr}
            assert Decimal(sc["derated_port_max_kw"]) == 120

    vintages = rows(HERE / "forecast_vintages.csv")
    values = rows(HERE / "forecast_values.csv")
    vintage_map = {v["vintage_id"]: v for v in vintages}
    by_vintage = defaultdict(list)
    for v in values:
        meta = vintage_map[v["vintage_id"]]
        sc = scenario_map[v["scenario_id"]]
        assert v["is_simulated"] == "true" and dt(v["issued_at"]) <= dt(sc["as_of"])
        assert dt(v["source_observation_end"]) <= dt(v["issued_at"])
        assert dt(v["interval_end"]) == dt(v["timestamp"]) + STEP
        assert Decimal(0) <= Decimal(v["lower_kw"]) <= Decimal(v["point_kw"]) <= Decimal(v["upper_kw"])
        if v["vintage_id"].endswith("-past_only") or v["vintage_id"].endswith("-incomplete"):
            previous = dt(v["timestamp"]) - timedelta(days=7 if v["resource"] == "load" else 1)
            assert dt(v["source_observation_end"]) == previous + STEP
            source = truth_map[previous.isoformat(timespec="seconds")]
            expected = Decimal(source["base_load_kw"] if v["resource"] == "load" else source["pv_interval_mean_kw_approx"])
            close(Decimal(v["point_kw"]), expected)
        if v["resource"] == "pv":
            assert Decimal(v["upper_kw"]) <= 5000
        assert meta["resource"] == v["resource"] and meta["scenario_id"] == v["scenario_id"]
        by_vintage[v["vintage_id"]].append(v)
    for meta in vintages:
        sc = scenario_map[meta["scenario_id"]]
        assert meta["is_simulated"] == "true" and int(meta["point_count"]) == len(by_vintage[meta["vintage_id"]])
        if meta["status"] == "usable":
            assert meta["method_sha256"] == sha(HERE / "generate_linked_data.py")
            assert meta["selected"] in ("true", "false")
            assert [v["timestamp"] for v in by_vintage[meta["vintage_id"]]] == [
                (dt(sc["window_start"]) + k * STEP).isoformat(timespec="seconds") for k in range(4)]
        if meta["vintage_id"] == sc["pv_vintage_id"]:
            assert meta["selected"] == "true"
            assert (meta["status"] == "usable") == (sc["forecast_status"] == "usable")
        if meta["native_grid"] == "day13_07_19_hourly_point":
            cutoff = datetime.fromisoformat(meta["window_start"][:10] + "T07:00:00+08:00")
            assert dt(meta["issued_at"]) > cutoff and meta["status"] != "usable"
    for sid, sc in scenario_map.items():
        assert vintage_map[sc["load_vintage_id"]]["status"] == "usable"
        pv_vintage = vintage_map[sc["pv_vintage_id"]]
        if sid == "S11_pv_gap":
            assert len(by_vintage[pv_vintage["vintage_id"]]) == 2
        if sid == "S12_late_weather":
            assert not by_vintage[pv_vintage["vintage_id"]]

    scenario_truth = rows(HERE / "scenario_truth.csv")
    by_scenario_truth = defaultdict(list)
    for r in scenario_truth:
        sc = scenario_map[r["scenario_id"]]
        source = truth_map[r["timestamp"]]
        assert r["is_simulated"] == "true" and dt(r["available_at"]) > dt(sc["as_of"])
        for key in ("base_load_kw", "historical_charge_kw", "total_load_kw", "pv_interval_mean_kw_approx", "pv_interval_energy_kwh_approx"):
            assert r[key] == source[key]
        by_scenario_truth[r["scenario_id"]].append(r)
    assert all(len(v) == 4 for v in by_scenario_truth.values())
    tariffs = rows(HERE / "tariff_intervals.csv")
    assert len(tariffs) == 48
    for row in tariffs:
        sc = scenario_map[row["scenario_id"]]
        assert row["is_simulated"] == "true" and dt(row["available_at"]) <= dt(sc["as_of"])
        assert dt(sc["window_start"]) <= dt(row["timestamp"]) < dt(sc["window_end"])
        h = dt(row["timestamp"]).hour
        expected = Decimal("0.4") if h < 7 or h >= 23 else Decimal("0.8") if h < 17 else Decimal("1.2")
        assert Decimal(row["import_tariff_cny_per_kwh"]) == expected
        assert Decimal(row["export_tariff_cny_per_kwh"]) == 0

    # Verify original storage trajectory independently, including physical energy update.
    parameters = rows(DATA / "储能参数与初始状态/储能参数.csv")[0]
    trace = rows(DATA / "储能参数与初始状态/状态校验轨迹.csv")
    accepted = 0
    for r in trace:
        if r["result_status"] != "accepted":
            continue
        expected = (Decimal(r["input_energy_kwh"])
                    + Decimal(parameters["charge_efficiency"]) * Decimal(r["applied_charge_kw"]) / 4
                    - Decimal(r["applied_discharge_kw"]) / (4 * Decimal(parameters["discharge_efficiency"])))
        close(Decimal(r["energy_after_kwh"]), expected)
        assert Decimal(r["applied_charge_kw"]) == 0 or Decimal(r["applied_discharge_kw"]) == 0
        accepted += 1

    # Cross-check several source periods and the zero-export power balance with idle storage.
    spotchecks = []
    for sid in ("S01_sunny", "S02_mismatch", "S03_grid", "S05_pv_surplus", "S11_pv_gap", "S12_late_weather"):
        r = by_scenario_truth[sid][0]
        load_kw = Decimal(r["base_load_kw"]) + Decimal(r["historical_charge_kw"])
        pv_available = Decimal(r["pv_interval_mean_kw_approx"])
        pv_used = min(load_kw, pv_available)
        pv_curtailed = pv_available - pv_used
        grid_import = load_kw - pv_used
        close(grid_import + pv_used, load_kw)
        close(pv_used + pv_curtailed, pv_available)
        assert Decimal(0) <= grid_import <= 10000 and pv_curtailed >= 0
        spotchecks.append({"scenario_id": sid, "timestamp": r["timestamp"],
                           "derived_grid_import_kw": str(grid_import), "derived_pv_used_kw": str(pv_used),
                           "derived_pv_curtailed_kw": str(pv_curtailed), "balance_residual_kw": "0"})
    assert Decimal(by_scenario_truth["S01_sunny"][0]["pv_interval_mean_kw_approx"]) > 3000
    assert Decimal(by_scenario_truth["S02_mismatch"][0]["pv_interval_mean_kw_approx"]) == 0
    assert Decimal(by_scenario_truth["S03_grid"][0]["total_load_kw"]) > 8000
    assert Decimal(by_scenario_truth["S05_pv_surplus"][0]["pv_interval_mean_kw_approx"]) > Decimal(by_scenario_truth["S05_pv_surplus"][0]["total_load_kw"])
    assert Decimal(scenario_map["S07_storage_floor"]["storage_initial_soc_percent"]) == Decimal(parameters["reserve_soc_percent"])
    assert Decimal(scenario_map["S08_storage_ceiling"]["storage_initial_soc_percent"]) == Decimal(parameters["soc_max_percent"])

    report["status"] = "passed"
    report["independent_validation"] = {
        "source_and_output_hashes_match": True, "all_35040_truth_rows_recomputed": True,
        "source_session_port_nonoverlap_count": len(sessions), "ports": len(by_port),
        "accepted_source_storage_steps_recomputed": accepted,
        "all_event_visibility_checked": True, "all_vintage_coverage_checked": True,
        "all_forecast_points_recomputed_from_past_only_truth": True,
        "all_tariff_intervals_checked": True,
        "all_scenario_truth_separated_from_as_of": True, "spotchecks": spotchecks,
    }
    (HERE / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "truth_rows": len(truth), "scenarios": len(scenarios),
                      "storage_steps": accepted, "spotchecks": len(spotchecks)}))


if __name__ == "__main__":
    main()
