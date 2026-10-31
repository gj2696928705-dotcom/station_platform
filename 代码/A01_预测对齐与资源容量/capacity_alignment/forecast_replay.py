"""Read D01 decision-visible files; never open scenario_truth or future sessions.

Run from 代码/A01_预测对齐与资源容量: python -m capacity_alignment.forecast_replay --data-dir ../数据
  --output evidence/replay --commitment-policy rated_from_as_of
Without that explicit assumption all energy-only bookings remain unavailable.
"""
import argparse
import csv
import hashlib
import json
from copy import deepcopy
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from .platform_support import enable_platform
enable_platform()
from station_platform.domain import evaluation, resources, schedule
from .forecast_capacity import BASE, PV
from .forecast_runtime import attach
from station_platform.legacy.demand import MockStationAdapter
from station_platform.util import PlatformError
from .common import digest, parse_time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_scenario(data_dir, scenario_id, commitment_policy=None):
    data_dir = Path(data_dir)
    linked = data_dir/"场站联动模拟数据"
    names = ["scenario_inputs.csv", "forecast_vintages.csv", "forecast_values.csv", "booking_events.csv", "vehicle_requests.csv", "tariff_intervals.csv"]
    files = {name: [r for r in read_csv(linked/name) if r["scenario_id"] == scenario_id] for name in names}
    if len(files["scenario_inputs.csv"]) != 1:
        raise ValueError("场景不存在或重复")
    scenario = files["scenario_inputs.csv"][0]
    station = scenario["station_id"]
    if scenario["is_simulated"] != "true" or scenario["execution_mode"] != "offline_replay":
        raise ValueError("仅支持 D01 模拟离线场景")
    start, end, as_of = (parse_time(scenario[k]) for k in ("window_start", "window_end", "as_of"))
    registry_path = data_dir/"场站配置与设备台账"/"设备台账.csv"
    meters_path = data_dir/"场站配置与设备台账"/"计量关系.csv"
    # Verify evidence rather than simply renaming the mock site to Huanghua.
    meters = [r for r in read_csv(meters_path) if r["station_id"] == station and r["meter_id"] == "M-PCC"]
    if len(meters) != 1 or meters[0]["is_simulated"] != "true" or parse_time(meters[0]["available_at"]) > as_of:
        raise ValueError("缺少同站点的模拟 AC 并网计量证据")
    ports = [r for r in read_csv(registry_path) if r["station_id"] == station and r["asset_type"] == "charging_port"]
    if len(ports) != int(scenario["available_port_count"]) or any(parse_time(p["available_at"]) > as_of for p in ports):
        raise ValueError("台账端口规模或可见时间不匹配")
    meta = {"station_id": station, "meter_boundary": "M-PCC/AC-BUS", "simulation": True,
            "issued_at": as_of.isoformat(), "available_at": as_of.isoformat(), "expires_at": end.isoformat()}
    source_hashes = {f"场站联动模拟数据/{name}": sha(linked/name) for name in names}
    source_hashes.update({"场站配置与设备台账/设备台账.csv": sha(registry_path), "场站配置与设备台账/计量关系.csv": sha(meters_path)})
    inputs = {"execution_mode": "offline_replay", "meter_boundary": meta["meter_boundary"],
              "binding": {**meta, "evidence_ref": "D01 synthetic registry/M-PCC; sha256=" + sha(meters_path)},
              "source_hashes": source_hashes}
    for resource, field, semantics in (("load", "load_vintage_id", BASE), ("pv", "pv_vintage_id", PV)):
        vintages = [r for r in files["forecast_vintages.csv"] if r["vintage_id"] == scenario[field]]
        if len(vintages) != 1:
            raise ValueError("预测版本不存在或重复")
        vintage = vintages[0]
        rows = [r for r in files["forecast_values.csv"] if r["vintage_id"] == vintage["vintage_id"]]
        if any(r["station_id"] != station or r["resource"] != resource or r["native_grid"] != "15min_interval_average" for r in rows):
            raise ValueError("原生预测身份或网格不匹配")
        inputs["load_forecast" if resource == "load" else "pv_forecast"] = {
            **meta, "station_id": vintage["station_id"], "record_id": vintage["vintage_id"],
            "issued_at": vintage["issued_at"], "available_at": vintage["issued_at"],
            "target_semantics": semantics, "unit": "kW", "value_semantics": "interval_average_power",
            "status": "available" if vintage["status"] == "usable" else "unavailable", "reason": vintage["reason"],
            "model_version": vintage["source_version"], "method_hash": vintage["method_sha256"],
            "bundle_hash": vintage["bundle_sha256"] or None, "input_hash": vintage["input_sha256"],
            "source_ref": "forecast_vintages.csv#" + vintage["vintage_id"], "raw_vintage": vintage,
            "raw_values": rows, "publication_status": "simulation_baseline",
            "values": [{"start": r["timestamp"], "end": r["interval_end"], "value": r["point_kw"],
                        "source_observation_end": r["source_observation_end"]} for r in rows]}
    visible_events = sorted([r for r in files["booking_events.csv"] if parse_time(r["available_at"]) <= as_of],
                            key=lambda r: (parse_time(r["available_at"]), parse_time(r["event_at"]), r["event_id"]))
    bookings = {}
    for event in visible_events:
        if event["event_type"] == "booking": bookings[event["session_id"]] = event
        elif event["event_type"] == "cancellation": bookings.pop(event["session_id"], None)
    commitments = {**meta, "status": "unknown", "source_ref": "booking_events.csv: energy-only; no power profile",
                   "coverage_start": start.isoformat(), "coverage_end": end.isoformat(), "sessions": [],
                   "visible_events": visible_events}
    if commitment_policy == "rated_from_as_of" and scenario["commitment_status"] != "unknown":
        commitments.update(status="known" if bookings else "known_empty",
                           source_ref="A01 explicit counterfactual: rated_from_as_of; not observed charging",
                           assumption="Each visible uncancelled booking starts at as_of at port maximum until its energy is delivered; no arrival truth used")
        for sid, booking in bookings.items():
            port = next((p for p in ports if p["asset_id"] == booking["port_id"]), None)
            if port is None: raise ValueError("预约端口不存在")
            power = Decimal(scenario["derated_port_max_kw"] if booking["port_id"] == scenario["derated_port_id"] else port["max_available_power_kw"])
            remaining = Decimal(booking["committed_energy_kwh"])
            values = []
            for i in range(4):
                delivered = min(remaining, power/4)
                values.append({"start": (start+timedelta(minutes=15*i)).isoformat(),
                               "end": (start+timedelta(minutes=15*(i+1))).isoformat(), "value": str(delivered*4)})
                remaining -= delivered
            commitments["sessions"].append({**meta, "session_id": sid, "asset_id": booking["port_id"],
                                             "target_semantics": "committed_charge_ac", "unit": "kW",
                                             "value_semantics": "interval_average_power", "values": values,
                                             "booking_event_ref": booking["event_id"], "remaining_after_window_kwh": str(remaining)})
    inputs["commitments"] = commitments
    equipment = [{"asset_id": p["asset_id"], "max_power_kw": float(scenario["derated_port_max_kw"] if p["asset_id"] == scenario["derated_port_id"] else p["max_available_power_kw"]),
                  "connector": "DC", "availability": "derated" if p["asset_id"] == scenario["derated_port_id"] else p["availability"],
                  "occupied_slots": [], "data_quality": "B"} for p in ports]
    tariffs = sorted(files["tariff_intervals.csv"], key=lambda r: r["timestamp"])
    if len(tariffs) != 4 or any(parse_time(r["available_at"]) > as_of for r in tariffs):
        raise ValueError("缺少可见的完整费率")
    site = deepcopy(MockStationAdapter().site)
    site.update(station_id=station, equipment=equipment, rates_cny_per_kwh=[float(r["import_tariff_cny_per_kwh"]) for r in tariffs])
    # Legacy placeholder curves are never advertised as Huanghua forecasts.
    # Successful simulation_use replaces them before any resource/search/evaluation.
    site["energy"].update(grid_import_limit_kw=float(scenario["grid_import_limit_kw"]))
    rows = [r for r in files["vehicle_requests.csv"] if parse_time(r["available_at"]) <= as_of]
    demand = {"vehicles": [{"vehicle_id": r["request_id"], "requested_energy_kwh": float(r["requested_energy_kwh"]),
                            "arrival_minutes": int((parse_time(r["arrival_at"])-start).total_seconds()/60),
                            "departure_minutes": int((parse_time(r["departure_deadline_at"])-start).total_seconds()/60),
                            "priority": "ordinary"} for r in rows]}
    request = MockStationAdapter(site).capture(demand, at=start, offline_replay=True)
    request["station_state_snapshot"]["source_state_id"] = scenario_id+"-synthetic-state"
    for vehicle, source in zip(request["station_state_snapshot"]["vehicles"], rows):
        vehicle["max_charge_power_kw"] = float(source["max_power_kw"])
        vehicle["required_charger_id"] = source["port_id"]
    request["station_state_snapshot"]["simulation_assumptions"] = ["D01 simulated station, AC ports and tariffs; not a real station binding",
        "A01 replay uses only decision-visible files; storage idle", "Existing charge profile policy: " + str(commitment_policy)]
    return request, {"mode": "simulation_use", "inputs": inputs}


def run(data_dir, output, commitment_policy=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scenarios = read_csv(Path(data_dir)/"场站联动模拟数据"/"scenario_inputs.csv")
    summary = []
    for row in scenarios:
        sid = row["scenario_id"]
        request, config = load_scenario(data_dir, sid, commitment_policy)
        request = attach(request, config, "replay-"+sid, 1)
        result = {"scenario_id": sid, "simulation": True, "request": request, "resource": resources(request)}
        try:
            result["service"] = schedule(request)
            result["evaluation"] = evaluation(request, result["resource"], result["service"])
            result["status"] = "planned_simulation"
        except PlatformError as error:
            result.update(status="blocked", code=error.code, message=error.message)
        (output/(sid+".json")).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        summary.append({"scenario_id": sid, "status": result["status"], "code": result.get("code"),
                        "context_hash": request["forecast_context"]["context"]["context_hash"],
                        "issues": request["forecast_context"]["context"]["issues"]})
    report = {"simulation": True, "execution_mode": "offline_replay", "commitment_policy": commitment_policy,
              "scenarios": summary, "planned_count": sum(r["status"] == "planned_simulation" for r in summary)}
    (output/"summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commitment-policy", choices=["rated_from_as_of"])
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.output, args.commitment_policy), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
