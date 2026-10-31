"""Continuous AC powers plus binary connection/storage modes, solved by SCIP."""
import math
from time import perf_counter

from .contract import DT, N, SCHEMA, InputError, fingerprint, number, validate_input


def classify_status(code):
    # OR-Tools MPSolver status values, not inferred from elapsed time.
    return {0: "OPTIMAL", 1: "FEASIBLE", 2: "INFEASIBLE", 3: "ERROR",
            4: "ERROR", 5: "ERROR", 6: "UNKNOWN"}.get(code, "ERROR")


def _conflict_hints(d, idle):
    """Sufficient, independently readable conflicts; not an IIS or a relaxation."""
    reasons = []
    s = d["storage"]
    floor = max(s["min_soc"], s["reserve_soc"]) * s["capacity_kwh"]
    ceiling = s["max_soc"] * s["capacity_kwh"]
    initial, target = s["initial_energy_kwh"], s["terminal_energy_kwh"]
    for label, energy in (("INITIAL", initial), ("TERMINAL", target)):
        if energy < floor-1e-5 or energy > ceiling+1e-5:
            reasons.append({"code": f"STORAGE_{label}_OUTSIDE_RESERVE_SOC", "actual_kwh": energy,
                            "minimum_kwh": floor, "maximum_kwh": ceiling})
    up = initial + (0 if idle else sum(s["max_charge_kw"])*DT*s["charge_efficiency"])
    down = initial - (0 if idle else sum(s["max_discharge_kw"])*DT/s["discharge_efficiency"])
    if target > up+1e-5 or target < down-1e-5:
        reasons.append({"code": "TERMINAL_UNREACHABLE_WITH_PCS", "target_kwh": target,
                        "relaxed_minimum_kwh": down, "relaxed_maximum_kwh": up})
    ports = {p["port_id"]: p for p in d["ports"]}
    for v in d["vehicles"]:
        capacities = []
        for pid in v["allowed_port_ids"]:
            p = ports[pid]
            if p["connector"] == v["connector"]:
                capacities.append(DT*sum(min(v["max_power_kw"], p["max_power_kw"][t])
                    for t in range(v["arrival_slot"], v["deadline_slot"]) if t not in p["occupied_slots"]))
        upper = max(capacities, default=0)
        if upper + 1e-5 < v["required_energy_kwh"]:
            reasons.append({"code": "VEHICLE_WINDOW_CAPACITY_SHORTFALL", "vehicle_id": v["vehicle_id"],
                            "required_kwh": v["required_energy_kwh"], "relaxed_maximum_kwh": upper})
    for t in range(N):
        existing = sum(x[t] for x in d["existing_charge_by_port_kw"].values())
        relaxed_minimum_grid = d["base_load_kw"][t] + existing - d["pv_available_kw"][t] - (0 if idle else s["max_discharge_kw"][t])
        if relaxed_minimum_grid > d["grid_import_limit_kw"][t]+1e-5:
            reasons.append({"code": "PREEXISTING_GRID_CONFLICT", "slot": t, "minimum_import_kw": relaxed_minimum_grid,
                            "limit_kw": d["grid_import_limit_kw"][t]})
        for pid, p in ports.items():
            actual = d["existing_charge_by_port_kw"][pid][t]
            if actual > p["max_power_kw"][t]+1e-5:
                reasons.append({"code": "EXISTING_PORT_OVERLOAD", "port_id": pid, "slot": t,
                                "actual_kw": actual, "limit_kw": p["max_power_kw"][t]})
    return reasons or [{"code": "COMBINED_CONSTRAINT_CONFLICT", "detail": "SCIP proved the full model infeasible; individual relaxed bounds do not identify a unique cause. No IIS is claimed."}]


def solve(data, *, time_limit_seconds=10, storage_idle=False):
    """No plan is returned without independent numeric verification.

    INFEASIBLE means proven within this exact four-slot model, UNKNOWN means no
    incumbent was found before termination. No promise is relaxed by a penalty.
    """
    started = perf_counter()
    try:
        d = validate_input(data)
        number(time_limit_seconds, "time limit seconds", 0.001, 60)
        if type(storage_idle) is not bool:
            raise InputError("INPUT_INVALID", "storage_idle must be a boolean")
    except InputError as e:
        return {"status": "INVALID_INPUT", "plan": None, "reasons": [{"code": e.code, "detail": str(e)}]}
    result = {"schema_version": SCHEMA, "simulation": True, "input_hash": fingerprint(d),
              "status": "UNKNOWN", "plan": None, "reasons": [], "storage_idle": storage_idle}
    try:
        from ortools.linear_solver import pywraplp
    except ImportError:
        return {**result, "status": "ERROR", "reasons": [{"code": "SOLVER_UNAVAILABLE", "detail": "OR-Tools required"}]}
    m = pywraplp.Solver.CreateSolver("SCIP")
    if m is None:
        return {**result, "status": "ERROR", "reasons": [{"code": "SOLVER_UNAVAILABLE", "detail": "SCIP backend required"}]}
    m.SetNumThreads(1)
    m.SetTimeLimit(max(1, int(time_limit_seconds * 1000)))
    m.SetSolverSpecificParametersAsString("numerics/feastol = 1e-9\nlimits/gap = 0\nlimits/absgap = 0\nrandomization/randomseedshift = 0")
    ports = {p["port_id"]: p for p in d["ports"]}
    s = d["storage"]
    power, connection, assigned = {}, {}, {}
    for vi, v in enumerate(d["vehicles"]):
        vid = v["vehicle_id"]
        choices = []
        all_power = []
        starts = []
        for pi, pid in enumerate(v["allowed_port_ids"]):
            p = ports[pid]
            if p["connector"] != v["connector"]:
                continue
            y = m.BoolVar(f"assign_{vi}_{pi}")
            assigned[vid, pid] = y
            choices.append(y)
            previous = 0
            for t in range(N):
                valid = v["arrival_slot"] <= t < v["deadline_slot"] and t not in p["occupied_slots"]
                z = m.IntVar(0, int(valid), f"connect_{vi}_{pi}_{t}")
                x = m.NumVar(0, min(v["max_power_kw"], p["max_power_kw"][t]) if valid else 0, f"power_{vi}_{pi}_{t}")
                h = m.BoolVar(f"start_{vi}_{pi}_{t}")
                m.Add(z <= y)
                m.Add(h >= z - previous)
                m.Add(x <= min(v["max_power_kw"], p["max_power_kw"][t]) * z)
                power[vid, pid, t], connection[vid, pid, t] = x, z
                all_power.append(x)
                starts.append(h)
                previous = z
        m.Add(m.Sum(choices) == 1)
        m.Add(m.Sum(starts) <= 1)  # one contiguous plugged interval, one physical port
        m.Add(DT * m.Sum(all_power) == v["required_energy_kwh"])
    for pid, p in ports.items():
        for t in range(N):
            zs = [z for (vid, port, slot), z in connection.items() if port == pid and slot == t]
            xs = [x for (vid, port, slot), x in power.items() if port == pid and slot == t]
            m.Add(m.Sum(zs) <= (0 if t in p["occupied_slots"] else 1))
            m.Add(m.Sum(xs) + d["existing_charge_by_port_kw"][pid][t] <= p["max_power_kw"][t])
    floor = max(s["min_soc"], s["reserve_soc"]) * s["capacity_kwh"]
    ceiling = s["max_soc"] * s["capacity_kwh"]
    energy = [m.NumVar(floor, ceiling, f"energy_{t}") for t in range(N + 1)]
    m.Add(energy[0] == s["initial_energy_kwh"])
    m.Add(energy[N] == s["terminal_energy_kwh"])
    grid, pv, charge, discharge = [], [], [], []
    for t in range(N):
        cmax = 0 if storage_idle else s["max_charge_kw"][t]
        dmax = 0 if storage_idle else s["max_discharge_kw"][t]
        pc, pd = m.NumVar(0, cmax, f"pc_{t}"), m.NumVar(0, dmax, f"pd_{t}")
        mode = m.BoolVar(f"storage_charge_mode_{t}")
        m.Add(pc <= cmax * mode)
        m.Add(pd <= dmax * (1 - mode))
        m.Add(energy[t+1] == energy[t] + DT * s["charge_efficiency"] * pc - DT / s["discharge_efficiency"] * pd)
        g = m.NumVar(0, d["grid_import_limit_kw"][t], f"grid_{t}")
        u = m.NumVar(0, d["pv_available_kw"][t], f"pv_used_{t}")
        new = m.Sum([x for (vid, pid, slot), x in power.items() if slot == t])
        existing = sum(values[t] for values in d["existing_charge_by_port_kw"].values())
        m.Add(g == d["base_load_kw"][t] + existing + new + pc - pd - u)
        if d["new_charge_limit_kw"] is not None:
            m.Add(new <= d["new_charge_limit_kw"][t])
        grid.append(g)
        pv.append(u)
        charge.append(pc)
        discharge.append(pd)
    wear, curtail = (d["objective"][k] for k in ("throughput_cost_cny_per_kwh", "curtailment_cost_cny_per_kwh"))
    m.Minimize(DT * m.Sum([grid[t] * d["rates_cny_per_kwh"][t] + wear * (charge[t] + discharge[t])
                           + curtail * (d["pv_available_kw"][t] - pv[t]) for t in range(N)]))
    code = m.Solve()
    status = classify_status(code)
    bound = m.Objective().BestBound()
    result.update(status=status, solver={"backend": m.SolverVersion(), "raw_status": code,
        "time_limit_seconds": time_limit_seconds, "elapsed_seconds": perf_counter() - started,
        "nodes": m.nodes(), "variables": m.NumVariables(), "constraints": m.NumConstraints(),
        "best_bound_cny": bound if math.isfinite(bound) and status in {"OPTIMAL", "FEASIBLE"} else None,
        "objective_cny": None, "absolute_gap_cny": None, "relative_gap": None,
        "optimality_proven": status == "OPTIMAL"})
    if status not in {"OPTIMAL", "FEASIBLE"}:
        reason = "MODEL_PROVEN_INFEASIBLE" if status == "INFEASIBLE" else "NO_INCUMBENT_FEASIBILITY_UNKNOWN" if status == "UNKNOWN" else "SOLVER_ERROR"
        result["reasons"] = [{"code": reason, "detail": "Check vehicle windows/connectors/port conflicts, grid, PCS, reserve and terminal equality; no constraint was relaxed."}]
        if status == "INFEASIBLE":
            result["reasons"].extend(_conflict_hints(d, storage_idle))
        return result
    value = m.Objective().Value()
    gap = max(0.0, value - bound) if math.isfinite(bound) else None
    result["solver"].update(objective_cny=value, absolute_gap_cny=gap,
                            relative_gap=gap / max(1.0, abs(value)) if gap is not None else None)
    assignments = []
    by_port = {pid: [0.0] * N for pid in ports}
    by_vehicle = {}
    for v in d["vehicles"]:
        vid = v["vehicle_id"]
        pid = next(pid for (vehicle, pid), y in assigned.items() if vehicle == vid and y.solution_value() > 0.5)
        values = [power[vid, pid, t].solution_value() for t in range(N)]
        plugged = [t for t in range(N) if connection[vid, pid, t].solution_value() > 0.5]
        assignments.append({"vehicle_id": vid, "port_id": pid, "power_kw": values, "connected_slots": plugged})
        by_vehicle[vid] = list(values)  # exported views must not share mutable lists
        for t in range(N):
            by_port[pid][t] += values[t]
    plan = {"schema_version": SCHEMA, "simulation": True, "input_hash": result["input_hash"],
        "assignments": assignments, "charge_by_vehicle_kw": by_vehicle, "charge_by_port_kw": by_port,
        "pv_used_kw": [x.solution_value() for x in pv], "storage_charge_kw": [x.solution_value() for x in charge],
        "storage_discharge_kw": [x.solution_value() for x in discharge],
        "storage_energy_kwh": [x.solution_value() for x in energy], "net_import_kw": [x.solution_value() for x in grid]}
    plan["fingerprint"] = fingerprint(plan)
    from .verify import verify_plan
    check = verify_plan(d, plan)
    result["verification"] = check
    if not check["passed"]:
        result.update(status="ERROR", reasons=[{"code": "INDEPENDENT_CHECK_FAILED", "detail": check["violations"]}])
        result["solver"]["optimality_proven"] = False
        return result
    if abs(check["metrics"]["total_objective_cny"] - value) > 1e-5:
        result.update(status="ERROR", reasons=[{"code": "OBJECTIVE_RECOMPUTATION_FAILED"}])
        result["solver"]["optimality_proven"] = False
        return result
    if storage_idle and any(abs(v) > 1e-5 for k in ("storage_charge_kw", "storage_discharge_kw") for v in plan[k]):
        raise RuntimeError("idle baseline violated")
    result.update(plan=plan, objective=check["metrics"])
    return result
