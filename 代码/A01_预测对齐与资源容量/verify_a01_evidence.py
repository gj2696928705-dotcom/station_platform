"""Independent A01 recomputation from frozen raw records, without algorithm imports."""
import argparse
import hashlib
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path


def dt(value):
    return datetime.fromisoformat(value)


def dec(value):
    return Decimal(str(value))


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def average(record, left, right):
    total, coverage = Decimal(0), Decimal(0)
    for point in record["values"]:
        start, end = dt(point["start"]), dt(point["end"])
        overlap = max(Decimal(0), dec((min(right, end)-max(left, start)).total_seconds()))
        power = dec(point["value"])
        if record["unit"] == "kWh":
            power = power*3600/dec((end-start).total_seconds())
        total += power*overlap
        coverage += overlap
    assert coverage == dec((right-left).total_seconds()), "Incomplete source coverage"
    return total/coverage


def near(left, right):
    assert abs(dec(left)-dec(right)) < Decimal("0.000002"), (left, right)


def verify(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    context = result["request"]["forecast_context"]["context"]
    inputs = context["inputs"]
    assert context["context_hash"] == sha({k: v for k,v in context.items() if k != "context_hash"})
    assert context["input_hash"] == sha(inputs)
    if result["status"] == "blocked":
        assert context["status"] == "unavailable"
        assert result["resource"]["charging_headroom_kw"] is None
        assert all(s["capacity_kw"] is None for s in context["slots"])
        return {"scenario_id": result["scenario_id"], "blocked_verified": True}
    assert context["status"] == "available"
    for stage in ("resource", "service", "evaluation"):
        assert result[stage]["frozen_context_hash"] == context["context_hash"]
    for i, slot in enumerate(context["slots"]):
        left, right = dt(slot["start"]), dt(slot["end"])
        base, pv = (average(inputs[k], left, right) for k in ("load_forecast", "pv_forecast"))
        old, occupied = Decimal(0), set()
        for session in inputs["commitments"]["sessions"]:
            p = average(session, left, right)
            old += p
            if p: occupied.add(session["asset_id"])
        device_limit = Decimal(0)
        for device in inputs["state"]["devices"]:
            busy = device["asset_id"] in occupied or any(min(right,dt(r["end"])) > max(left,dt(r["start"])) for r in device["occupied_intervals"])
            if not busy and not device["frozen"] and device["availability"] in {"available", "derated"}:
                device_limit += dec(device["max_power_kw"])
        grid_limit = dec(inputs["state"]["grid_import_limit_kw"])
        grid_available = grid_limit + pv-base-old
        assert grid_available >= 0
        for field, expected in (("base_load_kw",base), ("committed_charge_kw",old), ("pv_available_kw",pv),
                                ("device_new_limit_kw",device_limit), ("grid_new_limit_kw",grid_available),
                                ("capacity_kw",min(device_limit,grid_available))):
            near(slot[field], expected)
        near(result["resource"]["charging_headroom_kw"][i], min(device_limit,grid_available))
    for candidate in result["service"]["assignment_candidates"]:
        total_grid, pv_used, curtailed = Decimal(0), Decimal(0), Decimal(0)
        for i, slot in enumerate(context["slots"]):
            charging = sum(dec(series[i]) for series in candidate["charger_power_kw"].values())
            assert charging <= dec(slot["capacity_kw"])+Decimal("0.000002")
            load, pv = dec(slot["existing_load_kw"])+charging, dec(slot["pv_available_kw"])
            grid = max(Decimal(0), load-pv)
            used = min(load,pv)
            near(grid+used, load)
            total_grid += grid/4
            pv_used += used/4
            curtailed += (pv-used)/4
        row = next(r for r in result["evaluation"]["candidate_evaluations"] if r["candidate_id"] == candidate["candidate_id"])
        assert row["feasible"]
        near(row["grid_import_kwh"], total_grid)
        near(row["local_green_kwh"], pv_used)
        near(row["curtailed_kwh"], curtailed)
    return {"scenario_id": result["scenario_id"], "slots_verified": len(context["slots"]),
            "candidates_verified": len(result["service"]["assignment_candidates"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [p for p in args.path.glob("*.json") if p.name != "summary.json"] if args.path.is_dir() else [args.path]
    checks = [verify(path) for path in sorted(paths)]
    report = {"passed": True, "independent_verifier": "standard-library Decimal; no station algorithm imports", "checks": checks}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
