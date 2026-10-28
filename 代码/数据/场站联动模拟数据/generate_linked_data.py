"""Deterministic, read-only derivation of the Huanghua 2025 offline replay package."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
CONFIG = json.loads((HERE / "simulation_config.json").read_text(encoding="utf-8"))
STATION = CONFIG["station_id"]
DATASET = CONFIG["dataset_id"]
STEP = timedelta(minutes=15)
Q = Decimal("0.000001")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name: str, rows: list[dict], columns: list[str]) -> None:
    with (HERE / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(name: str, value: object) -> None:
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def num(value: Decimal | float | int) -> str:
    return str(Decimal(str(value)).quantize(Q))


def iso(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def day_split(date: str) -> str:
    if date < "2025-09-01":
        return "development"
    if date < "2025-11-01":
        return "calibration"
    return "replay_evaluation"


def tariff(t: datetime) -> Decimal:
    h = t.hour
    return Decimal("0.4") if h < 7 or h >= 23 else Decimal("0.8") if h < 17 else Decimal("1.2")


def main() -> None:
    before = {name: sha(DATA / name) for name in CONFIG["source_hashes_sha256"]}
    assert before == CONFIG["source_hashes_sha256"], "Pinned input SHA-256 changed"
    load_rows = read_csv(DATA / "负荷数据/黄花站_2025年_15分钟负荷模拟数据.csv")
    pv_rows = read_csv(DATA / "光伏数据/黄花站_2025年_15分钟光伏模拟数据.csv")
    sessions = read_csv(DATA / "负荷数据/充电会话明细.csv")
    assets = read_csv(DATA / "场站配置与设备台账/设备台账.csv")
    limits = read_csv(DATA / "场站配置与设备台账/功率约束.csv")
    storage = read_csv(DATA / "储能参数与初始状态/储能参数.csv")[0]
    assert len(load_rows) == len(pv_rows) == 35040
    assert storage["storage_id"] == "BESS-001" and storage["site_dispatch_ready"] == "false"
    ports = sorted({r["port_id"] for r in assets if r["port_id"]})
    assert len(ports) == 128 and all(s["port_id"] in ports for s in sessions)
    grid_limit = next(Decimal(r["limit_value"]) for r in limits if r["constraint_id"] == "C-PCC-IMPORT")
    export_limit = next(Decimal(r["limit_value"]) for r in limits if r["constraint_id"] == "C-PCC-EXPORT")
    assert grid_limit == 10000 and export_limit == 0

    truth: list[dict] = []
    truth_by_time: dict[str, dict] = {}
    for i, (load, pv) in enumerate(zip(load_rows, pv_rows)):
        t = datetime.fromisoformat(load["timestamp"])
        assert t == datetime.fromisoformat(pv["timestamp"]) and t.utcoffset() == timedelta(hours=8)
        assert load["station_id"] == pv["station_id"] == STATION
        assert load["is_simulated"] == pv["is_simulated"] == "true"
        assert i == 0 or t == datetime.fromisoformat(load_rows[i - 1]["timestamp"]) + STEP
        base = Decimal(load["base_load_kw"])
        charge = Decimal(load["charging_load_kw"])
        total = Decimal(load["total_load_kw"])
        assert base >= 0 and charge >= 0 and base + charge == total
        pv_left = Decimal(pv["pv_power_kw"])
        if i + 1 < len(pv_rows):
            pv_right = Decimal(pv_rows[i + 1]["pv_power_kw"])
            endpoint = "source_next_sample"
        else:
            # The PV source documentation explicitly defines this synthetic midnight anchor.
            assert t == datetime.fromisoformat("2025-12-31T23:45:00+08:00") and pv_left == 0
            pv_right = Decimal(0)
            endpoint = "source_document_2026_midnight_zero_anchor"
        assert 0 <= pv_left <= 5000 and 0 <= pv_right <= 5000
        pv_mean = (pv_left + pv_right) / 2
        row = {
            "station_id": STATION, "scenario_id": "BASE_2025", "dataset_id": DATASET,
            "is_simulated": "true", "source_versions": "load_v1|pv_v1|registry_v1",
            "timestamp": iso(t), "interval_end": iso(t + STEP), "available_at": iso(t + STEP),
            "base_load_kw": load["base_load_kw"], "historical_charge_kw": load["charging_load_kw"],
            "total_load_kw": load["total_load_kw"], "base_energy_kwh": num(base / 4),
            "historical_charge_energy_kwh": num(charge / 4), "total_load_energy_kwh": num(total / 4),
            "pv_sample_start_kw": pv["pv_power_kw"], "pv_sample_end_kw": num(pv_right),
            "pv_interval_mean_kw_approx": num(pv_mean), "pv_interval_energy_kwh_approx": num(pv_mean / 4),
            "pv_conversion": "linear_trapezoid_15min", "pv_right_endpoint_source": endpoint,
            "pv_interval_quality": "approximation_not_metered", "truth_view": "post_interval_only",
        }
        truth.append(row)
        truth_by_time[row["timestamp"]] = row
    assert len(truth_by_time) == 35040
    write_csv("station_interval_truth.csv", truth, list(truth[0]))

    by_day: dict[str, list[dict]] = defaultdict(list)
    for row in truth:
        by_day[row["timestamp"][:10]].append(row)

    def pick(day: str, metric: str) -> datetime:
        candidates = by_day[day]
        if metric == "sun":
            candidates = [r for r in candidates if 10 <= datetime.fromisoformat(r["timestamp"]).hour < 15]
            key = lambda r: Decimal(r["pv_interval_mean_kw_approx"])
        elif metric == "mismatch":
            candidates = [r for r in candidates if 17 <= datetime.fromisoformat(r["timestamp"]).hour < 22]
            key = lambda r: Decimal(r["historical_charge_kw"]) - Decimal(r["pv_interval_mean_kw_approx"])
        elif metric == "surplus":
            key = lambda r: Decimal(r["pv_interval_mean_kw_approx"]) - Decimal(r["total_load_kw"])
        elif metric == "charge":
            key = lambda r: Decimal(r["historical_charge_kw"])
        elif metric == "total":
            key = lambda r: Decimal(r["total_load_kw"])
        else:
            return datetime.fromisoformat(day + "T10:00:00+08:00")
        return datetime.fromisoformat(max(candidates, key=key)["timestamp"])

    specs = [
        ("S01_sunny", "2025-04-30", "sun", 2, 60, "normal sunny; PV use <= available"),
        ("S02_mismatch", "2025-06-17", "mismatch", 8, 60, "PV/load peak mismatch; enforce every interval"),
        ("S03_grid", "2025-04-02", "total", 40, 60, "10MW import ceiling may bind under simultaneous requests"),
        ("S04_concurrency", "2025-03-07", "charge", 70, 60, "128 independent ports and 240kW per port; grid ceiling"),
        ("S05_pv_surplus", "2025-03-01", "surplus", 0, 60, "zero export; absorb or curtail surplus PV"),
        ("S06_calibration", "2025-09-15", "fixed", 4, 60, "ordinary chronological calibration replay"),
        ("S07_storage_floor", "2025-11-15", "fixed", 4, 30, "SOC reserve floor; no ordinary discharge below 30%"),
        ("S08_storage_ceiling", "2025-11-16", "fixed", 4, 90, "SOC ceiling; no charging above 90%"),
        ("S09_derated", "2025-11-17", "fixed", 4, 60, "50% PCS and one port derating"),
        ("S10_no_booking", "2025-12-01", "fixed", 4, 60, "future commitment unknown without visible booking"),
        ("S11_pv_gap", "2025-12-02", "fixed", 4, 60, "selected PV vintage covers only 2 of 4 intervals; reject"),
        ("S12_late_weather", "2025-12-03", "fixed", 4, 60, "weather batch issued after D07:00; reject day13 candidate"),
    ]
    events: list[dict] = []
    requests: list[dict] = []
    scenario_inputs: list[dict] = []
    scenario_truth: list[dict] = []
    tariff_intervals: list[dict] = []
    vintages: list[dict] = []
    forecast_values: list[dict] = []
    for sid, day, metric, n_requests, soc, expected in specs:
        start = pick(day, metric)
        end = start + 4 * STEP
        as_of = start
        templates = [s for s in sessions if as_of - timedelta(days=7) <= datetime.fromisoformat(s["end_at"]) <= as_of - STEP]
        templates.sort(key=lambda s: s["session_id"])
        random.Random(CONFIG["seed"] + start.toordinal()).shuffle(templates)
        selected = []
        seen_ports = set()
        for s in templates:
            if s["port_id"] not in seen_ports:
                selected.append(s)
                seen_ports.add(s["port_id"])
            if len(selected) == 3:
                break
        assert len(selected) == 3
        if sid == "S10_no_booking":
            selected = []
        for j, s in enumerate(selected):
            book_at = as_of - timedelta(minutes=60)
            book_available = as_of - timedelta(minutes=30)
            common = {"station_id": STATION, "scenario_id": sid, "dataset_id": DATASET,
                      "is_simulated": "true", "source_version": "load_v1+synthetic_booking_rule_v1",
                      "session_id": f"BOOK-{sid}-{j}", "template_session_id": s["session_id"],
                      "port_id": s["port_id"],
                      "committed_energy_kwh": s["requested_energy_kwh"], "unit": "kWh"}
            events.append({**common, "event_id": f"{sid}-B{j}", "event_type": "booking",
                           "event_at": iso(book_at), "available_at": iso(book_available)})
            if j == 2:
                events.append({**common, "event_id": f"{sid}-C{j}", "event_type": "cancellation",
                               "event_at": iso(as_of - timedelta(minutes=5)),
                               "available_at": iso(as_of - timedelta(minutes=5))})
            if j < 2:
                arrival = as_of + STEP
                events.append({**common, "event_id": f"{sid}-A{j}", "event_type": "arrival",
                               "event_at": iso(arrival), "available_at": iso(arrival)})
        visible = [e for e in events if e["scenario_id"] == sid and datetime.fromisoformat(e["available_at"]) <= as_of]
        active_bookings = {e["session_id"]: e for e in visible if e["event_type"] == "booking"}
        for e in visible:
            if e["event_type"] == "cancellation":
                active_bookings.pop(e["session_id"], None)
        reserved = {e["port_id"] for e in active_bookings.values()}
        free = [p for p in ports if p not in reserved]
        for j in range(n_requests):
            requests.append({"station_id": STATION, "scenario_id": sid, "dataset_id": DATASET,
                             "is_simulated": "true", "source_version": "synthetic_request_rule_v1",
                             "request_id": f"{sid}-R{j + 1:03d}", "port_id": free[j],
                             "available_at": iso(as_of), "arrival_at": iso(as_of), "departure_deadline_at": iso(end),
                             "requested_energy_kwh": "50.000000", "max_power_kw": "240.000000"})
        for resource in ("load", "pv"):
            vid = f"{sid}-{resource}-past_only"
            observations = []
            for k in range(4):
                target = start + k * STEP
                source_t = target - timedelta(days=7 if resource == "load" else 1)
                source = truth_by_time[iso(source_t)]
                assert source_t + STEP <= as_of
                point = Decimal(source["base_load_kw"] if resource == "load" else source["pv_interval_mean_kw_approx"])
                width = Decimal(100) + point * Decimal("0.1" if resource == "load" else "0.25")
                upper_cap = Decimal("1e9") if resource == "load" else Decimal(5000)
                observations.append((iso(source_t), str(point)))
                forecast_values.append({"station_id": STATION, "scenario_id": sid, "dataset_id": DATASET,
                                        "is_simulated": "true", "source_version": "linked_baseline_v1",
                                        "vintage_id": vid, "resource": resource, "timestamp": iso(target),
                                        "interval_end": iso(target + STEP), "issued_at": iso(as_of),
                                        "as_of": iso(as_of), "source_observation_end": iso(source_t + STEP),
                                        "native_grid": "15min_interval_average", "point_kw": num(point),
                                        "lower_kw": num(max(Decimal(0), point - width)),
                                        "upper_kw": num(min(upper_cap, point + width)),
                                        "interval_kind": "heuristic_uncalibrated", "status": "usable"})
            vintages.append({"station_id": STATION, "scenario_id": sid, "dataset_id": DATASET,
                             "is_simulated": "true", "source_version": "linked_baseline_v1",
                             "vintage_id": vid, "resource": resource, "issued_at": iso(as_of), "as_of": iso(as_of),
                             "native_grid": "15min_interval_average", "window_start": iso(start), "window_end": iso(end),
                             "point_count": 4, "status": "usable", "reason": "past_only_baseline",
                             "method": CONFIG["baseline"][f"{resource}_method"], "bundle_sha256": "",
                             "method_sha256": sha(Path(__file__)),
                             "input_sha256": digest(observations), "selected": "true" if sid not in ("S11_pv_gap", "S12_late_weather") or resource == "load" else "false"})
        chosen_pv = f"{sid}-pv-past_only"
        forecast_status = "usable"
        if sid == "S11_pv_gap":
            chosen_pv = f"{sid}-pv-incomplete"
            forecast_status = "unavailable_incomplete_coverage"
            subset = [r for r in forecast_values if r["vintage_id"] == f"{sid}-pv-past_only"][:2]
            for r in subset:
                forecast_values.append({**r, "vintage_id": chosen_pv, "status": "unavailable_incomplete_coverage"})
            vintages.append({**vintages[-1], "vintage_id": chosen_pv, "point_count": 2,
                             "status": forecast_status, "reason": "2_of_4_intervals", "selected": "true",
                             "input_sha256": digest([(r["timestamp"], r["point_kw"]) for r in subset])})
        elif sid == "S12_late_weather":
            chosen_pv = f"{sid}-pv-day13-late"
            forecast_status = "unavailable_weather_after_cutoff"
            issued = iso(datetime.fromisoformat(day + "T07:10:00+08:00"))
            vintages.append({**vintages[-1], "vintage_id": chosen_pv, "issued_at": issued,
                             "native_grid": "day13_07_19_hourly_point", "point_count": 0,
                             "status": forecast_status, "reason": "no_complete_same_batch_issued_by_D07:00",
                             "method": "2026_candidate_day13_not_run_for_2025_as_of", "selected": "true",
                             "input_sha256": "", "bundle_sha256": "", "method_sha256": ""})
        derate = Decimal("0.5") if sid == "S09_derated" else Decimal(1)
        storage_kwh = Decimal(soc) * Decimal(storage["soc_reference_capacity_kwh"]) / 100
        scenario_inputs.append({"station_id": STATION, "scenario_id": sid, "dataset_id": DATASET,
                                "is_simulated": "true", "source_versions": "load_v1|pv_v1|registry_v1|storage_v1|linked_assumptions_v1",
                                "execution_mode": "offline_replay", "split": day_split(day),
                                "as_of": iso(as_of), "window_start": iso(start), "window_end": iso(end),
                                "load_vintage_id": f"{sid}-load-past_only", "pv_vintage_id": chosen_pv,
                                "forecast_status": forecast_status,
                                "commitment_status": "unknown" if not active_bookings else "visible_synthetic_bookings_only",
                                "visible_booking_count": len(active_bookings), "request_count": n_requests,
                                "grid_import_limit_kw": num(grid_limit), "grid_export_limit_kw": num(export_limit),
                                "available_port_count": 128, "port_rated_kw": "240.000000",
                                "derated_port_id": free[0] if sid == "S09_derated" else "",
                                "derated_port_max_kw": "120.000000" if sid == "S09_derated" else "",
                                "storage_id": storage["storage_id"], "storage_state_available_at": iso(as_of),
                                "storage_initial_soc_percent": num(soc), "storage_initial_energy_kwh": num(storage_kwh),
                                "storage_max_charge_kw": num(Decimal(storage["max_charge_kw"]) * derate),
                                "storage_max_discharge_kw": num(Decimal(storage["max_discharge_kw"]) * derate),
                                "storage_charge_efficiency": storage["charge_efficiency"],
                                "storage_discharge_efficiency": storage["discharge_efficiency"],
                                "storage_reserve_soc_percent": storage["reserve_soc_percent"],
                                "storage_soc_max_percent": storage["soc_max_percent"],
                                "storage_terminal_target_percent": storage["terminal_soc_target_percent"],
                                "storage_terminal_at": iso(datetime.fromisoformat(day + "T00:00:00+08:00") + timedelta(days=1)),
                                "storage_site_dispatch_ready": "false", "tariff_rule_id": "synthetic_tou_2025_v1",
                                "expected_constraint": expected, "selection_basis": "posthoc_synthetic_truth_stress_case"})
        for k in range(4):
            target = start + k * STEP
            actual = truth_by_time[iso(target)]
            tariff_intervals.append({"station_id": STATION, "scenario_id": sid, "dataset_id": DATASET,
                                     "is_simulated": "true", "source_version": "synthetic_tou_2025_v1",
                                     "timestamp": iso(target), "interval_end": iso(target + STEP),
                                     "available_at": iso(as_of), "import_tariff_cny_per_kwh": num(tariff(target)),
                                     "export_tariff_cny_per_kwh": "0.000000", "unit": "CNY/kWh"})
            scenario_truth.append({"station_id": STATION, "scenario_id": sid, "dataset_id": DATASET,
                                   "is_simulated": "true", "source_versions": "load_v1|pv_v1",
                                   "timestamp": iso(target), "interval_end": iso(target + STEP),
                                   "available_at": iso(target + STEP), "truth_view": "post_interval_only",
                                   "base_load_kw": actual["base_load_kw"],
                                   "historical_charge_kw": actual["historical_charge_kw"],
                                   "total_load_kw": actual["total_load_kw"],
                                   "pv_interval_mean_kw_approx": actual["pv_interval_mean_kw_approx"],
                                   "pv_interval_energy_kwh_approx": actual["pv_interval_energy_kwh_approx"],
                                   "import_tariff_cny_per_kwh": num(tariff(target))})

    write_csv("booking_events.csv", events, ["station_id", "scenario_id", "dataset_id", "is_simulated", "source_version", "event_id", "event_type", "event_at", "available_at", "session_id", "template_session_id", "port_id", "committed_energy_kwh", "unit"])
    write_csv("vehicle_requests.csv", requests, list(requests[0]))
    write_csv("scenario_inputs.csv", scenario_inputs, list(scenario_inputs[0]))
    write_csv("scenario_truth.csv", scenario_truth, list(scenario_truth[0]))
    write_csv("tariff_intervals.csv", tariff_intervals, list(tariff_intervals[0]))
    write_csv("forecast_vintages.csv", vintages, list(vintages[0]))
    write_csv("forecast_values.csv", forecast_values, list(forecast_values[0]))
    # Only source files are listed. The manifest itself and report are excluded to avoid circular hashes.
    source_names = sorted(set(CONFIG["source_hashes_sha256"]) | {"储能参数与初始状态/状态校验轨迹.csv"} | {
        f"{folder}/{name}" for folder in ("负荷数据", "光伏数据", "场站配置与设备台账", "储能参数与初始状态")
        for name in ("数据说明.md", "simulation_config.json", "validation_report.json")
    })
    manifest = [{"station_id": STATION, "scenario_id": "PACKAGE", "is_simulated": True,
                 "path": name, "sha256": sha(DATA / name),
                 "pinned": name in CONFIG["source_hashes_sha256"], "read_only": True} for name in source_names]
    write_json("source_manifest.json", manifest)
    assert {name: sha(DATA / name) for name in before} == before, "Source modified during generation"
    outputs = ["station_interval_truth.csv", "booking_events.csv", "vehicle_requests.csv",
               "scenario_inputs.csv", "scenario_truth.csv", "tariff_intervals.csv",
               "forecast_vintages.csv", "forecast_values.csv",
               "source_manifest.json"]
    write_json("validation_report.json", {
        "dataset_id": DATASET, "is_simulated": True, "execution_mode": "offline_replay",
        "status": "generated_pending_independent_validation",
        "counts": {"truth_intervals": len(truth), "scenarios": len(specs), "scenario_truth_intervals": len(scenario_truth),
                   "booking_events": len(events), "vehicle_requests": len(requests), "forecast_vintages": len(vintages),
                   "forecast_values": len(forecast_values)},
        "source_hashes_sha256": before,
        "output_hashes_sha256": {name: sha(HERE / name) for name in outputs},
        "boundary": "Last PV interval uses documented synthetic 2026-01-01 00:00 zero anchor, explicitly flagged; no 2026 observed row",
        "forecast_limitation": "Past-only baselines and heuristic intervals; no calibrated coverage or 2025-published model claim",
        "test_set_limitation": "Scenario windows selected post hoc from known synthetic truth; no blind-test claim",
    })


if __name__ == "__main__":
    main()
