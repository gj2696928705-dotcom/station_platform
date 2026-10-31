"""End-to-end candidate smoke run with EXPLICITLY SYNTHETIC weather/bookings."""
import argparse
import csv
import json
import math
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

from capacity_alignment.platform_support import enable_platform
enable_platform()
from station_platform.domain import evaluation, resources, schedule
from capacity_alignment.forecast_adapter import run_candidate, sha
from station_platform.forecast_api import attach
from capacity_alignment.forecast_replay import read_csv
from station_platform.legacy.demand import MockStationAdapter
from station_platform.util import parse_time


def run(data_dir, model_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    as_of = "2025-12-31T09:50:00+08:00"
    start = parse_time(as_of)
    end = (start+timedelta(hours=1)).isoformat()
    weather = output/"synthetic_weather_archive.csv"
    with weather.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["issued_at", "timestamp", "SWGDN", "DHI", "DNI", "rain"])
        writer.writeheader()
        for hour in range(7,20):
            sun = max(0, math.sin((hour-7)*math.pi/12))
            writer.writerow({"issued_at": "2025-12-31T06:00:00+08:00", "timestamp": f"2025-12-31T{hour:02}:00:00+08:00",
                             "SWGDN": 600*sun, "DHI": 100*sun, "DNI": 500*sun, "rain": 0})
    # This is a fictional forecast scenario, never archived realized weather.
    load = run_candidate(model_dir/"load-v1/deployment.json", data_dir/"负荷数据/黄花站_2025年_15分钟负荷模拟数据.csv", as_of, output/"load")
    pv = run_candidate(model_dir/"pv-day13-v1/deployment.json", data_dir/"光伏数据/黄花站_2025年_07-19_小时数据.csv", as_of, output/"pv", weather=weather)
    registry = data_dir/"场站配置与设备台账/设备台账.csv"
    ports = [r for r in read_csv(registry) if r["asset_type"] == "charging_port"]
    site = deepcopy(MockStationAdapter().site)
    site["station_id"] = "HH-Huanghua-CSS"
    site["equipment"] = [{"asset_id": p["asset_id"], "max_power_kw": float(p["max_available_power_kw"]), "connector": "DC",
                          "availability": p["availability"], "occupied_slots": [], "data_quality": "B"} for p in ports]
    site["energy"]["grid_import_limit_kw"] = 10000
    request = MockStationAdapter(site).capture({"vehicles": [{"vehicle_id": "synthetic-new-1", "requested_energy_kwh": 50,
        "arrival_minutes": 0, "departure_minutes": 60, "priority": "ordinary"}]}, at=start)
    meta = {"station_id": "HH-Huanghua-CSS", "meter_boundary": "M-PCC/AC-BUS", "simulation": True,
            "issued_at": as_of, "available_at": as_of, "expires_at": end}
    inputs = {"execution_mode": "offline_replay", "meter_boundary": "M-PCC/AC-BUS", "load_forecast": load, "pv_forecast": pv,
              "binding": {**meta, "evidence_ref": "Synthetic registry AC-BUS: " + sha(registry)},
              "commitments": {**meta, "status": "known_empty", "coverage_start": as_of, "coverage_end": end, "sessions": [],
                              "source_ref": "A01 fictional empty-booking scenario; not historical station bookings"}}
    request = attach(request, {"mode": "simulation_use", "inputs": inputs}, "candidate-smoke-1", 1)
    resource, service = resources(request), schedule(request)
    result = {"scenario_id": "candidate_smoke_synthetic_weather_and_empty_bookings", "status": "planned_simulation",
              "simulation": True, "assumptions": ["Retrospective 2026 candidate bundles", "Synthetic mathematical weather batch, not 2025 realized weather", "Explicit fictional empty existing bookings"],
              "request": request, "resource": resource, "service": service, "evaluation": evaluation(request, resource, service)}
    (output/"result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "load_status": load["status"], "pv_status": pv["status"], "context_hash": resource["frozen_context_hash"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.model_dir, args.output)


if __name__ == "__main__":
    main()
