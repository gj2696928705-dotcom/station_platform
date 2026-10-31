"""Read-only D01 adapter. Truth CSVs are never used as optimization inputs."""
import csv
from datetime import timedelta
import hashlib
import json
from pathlib import Path

from .contract import DT, N, SCHEMA, InputError, require, time_at, validate_input
from .solver import solve
from .verify import simulation_gate


def _sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


class LinkedReplay:
    """A01-compatible frozen 15-minute baseline, not a published prediction model.

    Booking energy alone is insufficient. Default loading fails until the caller
    supplies profiles for every visible booking. demo_assumptions=True explicitly
    creates separate hypothetical profiles, with the assumptions in the input hash.
    """
    def __init__(self, data_root):
        self.root = Path(data_root).resolve()
        self.linked = self.root / "场站联动模拟数据"
        self.refs = {}
        self.tables = {}
        report_path = self.linked / "validation_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
        require(report["status"] == "passed", "D01 data validation not passed", "DATA_UNAVAILABLE")
        self.refs["linked_validation_report"] = _sha(report_path)
        names = ("scenario_inputs.csv", "forecast_vintages.csv", "forecast_values.csv",
                 "booking_events.csv", "vehicle_requests.csv", "tariff_intervals.csv", "source_manifest.json")
        for name in names:
            path = self.linked / name
            actual = _sha(path)
            require(actual == report["output_hashes_sha256"][name], f"hash mismatch: {name}", "SOURCE_HASH_MISMATCH")
            self.refs[name] = actual
            if name.endswith(".csv"):
                with path.open(encoding="utf-8-sig", newline="") as f:
                    rows = list(csv.DictReader(f))
                require(all(r["station_id"] == report["station_id"] if "station_id" in report else r["station_id"] == "HH-Huanghua-CSS" for r in rows), "mixed stations")
                require(all(r["is_simulated"] == "true" for r in rows), "non-simulated source")
                self.tables[name] = rows
        for relative in ("场站配置与设备台账/设备台账.csv", "储能参数与初始状态/储能参数.csv"):
            path = self.root / relative
            actual = _sha(path)
            require(actual == report["source_hashes_sha256"][relative], f"hash mismatch: {relative}", "SOURCE_HASH_MISMATCH")
            self.refs[relative] = actual
            with path.open(encoding="utf-8-sig", newline="") as f:
                self.tables[relative] = list(csv.DictReader(f))

    def scenario_ids(self):
        return [r["scenario_id"] for r in self.tables["scenario_inputs.csv"]]

    def load(self, scenario_id, *, existing_profiles=None, objective=None, demo_assumptions=False):
        rows = [r for r in self.tables["scenario_inputs.csv"] if r["scenario_id"] == scenario_id]
        require(len(rows) == 1, "unknown or duplicate scenario", "SCENARIO_INVALID")
        r = rows[0]
        require(r["forecast_status"] == "usable", r["forecast_status"], "FORECAST_UNAVAILABLE")
        require(r["commitment_status"] != "unknown", "no known existing commitments", "COMMITMENT_UNKNOWN")
        at, start, end = (time_at(r[k]) for k in ("as_of", "window_start", "window_end"))
        require(time_at(r["storage_state_available_at"]) <= at, "storage state not visible", "STATE_UNAVAILABLE")
        grid = [start + timedelta(minutes=15*t) for t in range(N)]
        require(end == start+timedelta(hours=1), "wrong window")

        def selected(name):
            return [x for x in self.tables[name] if x["scenario_id"] == scenario_id]

        def ordered_points(points):
            points = sorted(points, key=lambda x: time_at(x["timestamp"]))
            require(len(points) == N and [time_at(x["timestamp"]) for x in points] == grid,
                    "missing/duplicate/nonaligned intervals", "FORECAST_COVERAGE")
            require(all(time_at(x["interval_end"]) == grid[i]+timedelta(minutes=15) for i, x in enumerate(points)), "wrong interval semantics")
            return points

        forecasts = {}
        for kind in ("load", "pv"):
            vid = r[f"{kind}_vintage_id"]
            versions = [x for x in selected("forecast_vintages.csv") if x["vintage_id"] == vid]
            require(len(versions) == 1, "missing/duplicate forecast vintage")
            v = versions[0]
            issued = time_at(v["issued_at"])
            require(v["resource"] == kind and v["status"] == "usable" and v["selected"] == "true" and issued <= at
                    and v["native_grid"] == "15min_interval_average" and time_at(v["window_start"]) == start
                    and time_at(v["window_end"]) == end and int(v["point_count"]) == N,
                    "unusable/late/non-native forecast", "FORECAST_UNAVAILABLE")
            points = ordered_points([x for x in selected("forecast_values.csv") if x["vintage_id"] == vid])
            require(all(x["resource"] == kind and x["status"] == "usable" and time_at(x["issued_at"]) == issued
                        and time_at(x["source_observation_end"]) <= issued and time_at(x["as_of"]) <= at
                        and x["native_grid"] == "15min_interval_average" for x in points), "forecast visibility/semantics violation")
            forecasts[kind] = [float(x["point_kw"]) for x in points]
        tariffs = ordered_points(selected("tariff_intervals.csv"))
        require(all(time_at(x["available_at"]) <= at and x["unit"] == "CNY/kWh" for x in tariffs), "tariff unavailable")
        bookings = {}
        events = sorted(selected("booking_events.csv"), key=lambda x: (time_at(x["available_at"]), x["event_id"]))
        for event in events:
            if time_at(event["available_at"]) > at:
                continue
            require(time_at(event["event_at"]) <= at, "future booking event")
            if event["event_type"] == "booking":
                bookings[event["session_id"]] = event
            elif event["event_type"] == "cancellation":
                bookings.pop(event["session_id"], None)
        require(len(bookings) == int(r["visible_booking_count"]), "booking visibility count mismatch")
        assumptions = ["D01 past-only synthetic baselines; offline replay, no calibrated forecast coverage.",
            "AC-equivalent charging energy, without vehicle DC/battery conversion losses.",
            "Window terminal energy equals initial energy; midnight target remains unverified outside this window."]
        if demo_assumptions:
            require(existing_profiles is None, "choose explicit profiles or demo assumptions")
            # Independent hypothetical service, NOT inferred from energy or future arrivals.
            existing_profiles = {sid: {"port_id": b["port_id"], "power_kw": [min(120.0, float(b["committed_energy_kwh"]))]*N,
                                       "occupied_slots": list(range(N))} for sid, b in bookings.items()}
            assumptions.append("DEMO ONLY: visible bookings reserve their ports for all four slots; assumed flat AC power min(120kW, booking_kWh/1h). Residual energy has no asserted deadline or cross-window feasibility. Not an observed/forecast committed profile.")
        require(isinstance(existing_profiles, dict) and set(existing_profiles) == set(bookings),
                "booking energy has no power/window profile; supply every visible booking explicitly", "COMMITMENT_PROFILE_MISSING")
        if objective is None:
            require(demo_assumptions, "explicit objective costs required", "OBJECTIVE_MISSING")
            objective = {"throughput_cost_cny_per_kwh": 0.02, "curtailment_cost_cny_per_kwh": 0.001}
            assumptions.append("DEMO ONLY: AC throughput cost 0.02 CNY/kWh; curtailment penalty 0.001 CNY/kWh, not tariff/settlement revenue.")
        assets = self.tables["场站配置与设备台账/设备台账.csv"]
        ports = []
        for p in assets:
            if p["asset_type"] != "charging_port":
                continue
            require(p["station_id"] == r["station_id"] and time_at(p["available_at"]) <= at
                    and time_at(p["effective_from"]) <= start and time_at(p["effective_to"]) >= end, "asset not effective")
            blocked = any(p[k] == "true" for k in ("fault", "protection_active", "manual_takeover", "frozen")) or p["availability"] not in {"available", "derated"}
            limit = 0.0 if blocked else float(p["max_available_power_kw"])
            if p["port_id"] == r["derated_port_id"]:
                limit = min(limit, float(r["derated_port_max_kw"]))
            ports.append({"port_id": p["port_id"], "connector": p["connector_type"], "max_power_kw": [limit]*N, "occupied_slots": []})
        require(len(ports) == int(r["available_port_count"]), "port count mismatch")
        port_map = {p["port_id"]: p for p in ports}
        existing = {pid: [0.0]*N for pid in port_map}
        for sid, profile in existing_profiles.items():
            pid = profile["port_id"]
            require(pid == bookings[sid]["port_id"] and pid in port_map, "booking port mismatch")
            from .contract import curve, slots
            values = curve(profile["power_kw"], "existing booking power")
            occupied = profile["occupied_slots"]
            slots(occupied, "existing booking occupied slots")
            require(not set(occupied) & set(port_map[pid]["occupied_slots"]), "existing booking conflict")
            require(sum(values)*DT <= float(bookings[sid]["committed_energy_kwh"])+1e-5, "booking overdelivered")
            port_map[pid]["occupied_slots"].extend(occupied)
            existing[pid] = [a+b for a, b in zip(existing[pid], values)]
        vehicles = []
        requests = selected("vehicle_requests.csv")
        require(len(requests) == int(r["request_count"]), "request count mismatch")
        for v in requests:
            require(time_at(v["available_at"]) <= at, "future request not visible")
            a = (time_at(v["arrival_at"])-start).total_seconds()/900
            b = (time_at(v["departure_deadline_at"])-start).total_seconds()/900
            require(a.is_integer() and b.is_integer(), "nonaligned request")
            require(v["port_id"] in port_map, "unmapped vehicle port")
            vehicles.append({"vehicle_id": v["request_id"], "connector": port_map[v["port_id"]]["connector"],
                "allowed_port_ids": [v["port_id"]], "arrival_slot": int(a), "deadline_slot": int(b),
                "required_energy_kwh": float(v["requested_energy_kwh"]), "max_power_kw": float(v["max_power_kw"])})
        params = [p for p in self.tables["储能参数与初始状态/储能参数.csv"] if p["storage_id"] == r["storage_id"] and p["station_id"] == r["station_id"]]
        require(len(params) == 1, "storage identity mismatch")
        p = params[0]
        require(time_at(p["parameter_available_at"]) <= at and time_at(p["parameter_valid_from"]) <= start
                and time_at(p["parameter_valid_to"]) >= end, "storage parameters not effective")
        cap = float(p["soc_reference_capacity_kwh"])
        initial = float(r["storage_initial_energy_kwh"])
        require(abs(initial-cap*float(r["storage_initial_soc_percent"])/100) < 1e-5, "storage SOC/energy mismatch")
        require(time_at(r["storage_terminal_at"]) > end, "midnight target falls inside window; extend contract explicitly")
        return validate_input({"schema_version": SCHEMA, "simulation": True, "execution_mode": "offline_replay",
            "station_id": r["station_id"], "meter_boundary": "station_ac", "task_id": scenario_id, "task_version": 1,
            "as_of": r["as_of"], "window_start": r["window_start"], "window_end": r["window_end"], "interval_minutes": 15,
            "source_refs": self.refs, "assumptions": assumptions, "commitment_status": "complete",
            "base_load_kw": forecasts["load"], "pv_available_kw": forecasts["pv"], "grid_import_limit_kw": [float(r["grid_import_limit_kw"])]*N,
            "grid_export_limit_kw": float(r["grid_export_limit_kw"]), "rates_cny_per_kwh": [float(t["import_tariff_cny_per_kwh"]) for t in tariffs],
            "new_charge_limit_kw": None, "ports": ports, "existing_charge_by_port_kw": existing, "vehicles": vehicles,
            "storage": {"storage_id": r["storage_id"], "capacity_kwh": cap, "initial_energy_kwh": initial,
                "min_soc": float(p["soc_min_percent"])/100, "reserve_soc": float(r["storage_reserve_soc_percent"])/100,
                "max_soc": float(r["storage_soc_max_percent"])/100, "charge_efficiency": float(r["storage_charge_efficiency"]),
                "discharge_efficiency": float(r["storage_discharge_efficiency"]), "max_charge_kw": [float(r["storage_max_charge_kw"])]*N,
                "max_discharge_kw": [float(r["storage_max_discharge_kw"])]*N, "terminal_energy_kwh": initial,
                "terminal_rule": "hard_equality_at_window_end"}, "objective": objective,
            "deferred_midnight_target": {"energy_kwh": cap*float(r["storage_terminal_target_percent"])/100,
                "at": r["storage_terminal_at"], "status": "not_verified_outside_window"},
            "source_site_dispatch_ready": r["storage_site_dispatch_ready"] == "true"})


def compare_idle(data, *, time_limit_seconds=10, allow_assumptions=False):
    """Same frozen input/objective/commitments/terminal; only storage is forced idle."""
    joint = solve(data, time_limit_seconds=time_limit_seconds)
    idle = solve(data, time_limit_seconds=time_limit_seconds, storage_idle=True)
    comparison = {"joint": joint, "idle": idle, "delta_joint_minus_idle": None,
                  "baseline_kind": "optimized_charging_with_idle_storage_on_identical_input"}
    if joint["plan"] is not None:
        comparison["gate"] = simulation_gate(data, joint["plan"], allow_assumptions=allow_assumptions)
    if joint["plan"] is not None and idle["plan"] is not None:
        keys = ("purchase_cost_cny", "total_objective_cny", "pv_used_kwh", "pv_curtailed_kwh", "peak_import_kw",
                "storage_throughput_kwh", "storage_loss_kwh", "storage_terminal_kwh", "grid_import_kwh")
        comparison["delta_joint_minus_idle"] = {k: joint["objective"][k]-idle["objective"][k] for k in keys}
    return comparison
