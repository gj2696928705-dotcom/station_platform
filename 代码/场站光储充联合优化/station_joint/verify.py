"""Deterministic recomputation, without importing OR-Tools or solver expressions.

This is the A02 numeric gate, not a substitute for the full live-state A03 service.
"""
from .contract import DT, EPS, N, SCHEMA, InputError, curve, fingerprint, slots, validate_input


def verify_plan(data, plan, *, live_input=None):
    errors = []

    def fail(code, **values):
        errors.append({"code": code, **values})

    result = {"passed": False, "simulation": True, "certified_safety_service": False,
              "violations": errors, "metrics": None, "plan_fingerprint": None,
              "input_hash": None, "checked_input_hash": None}
    try:
        frozen = validate_input(data)
        d = validate_input(live_input) if live_input is not None else frozen
        result.update(input_hash=fingerprint(frozen), checked_input_hash=fingerprint(d))
        for key in ("station_id", "task_id", "task_version", "window_start", "window_end", "meter_boundary"):
            if d[key] != frozen[key]:
                fail("LIVE_CONTEXT_MISMATCH", field=key)
        if not isinstance(plan, dict):
            raise InputError("PLAN_CONTENT_INVALID", "plan must be an object")
        result["plan_fingerprint"] = plan.get("fingerprint")
        actual_hash = fingerprint({k: v for k, v in plan.items() if k != "fingerprint"})
        if plan.get("fingerprint") != actual_hash:
            fail("PLAN_HASH_MISMATCH")
        if plan.get("input_hash") != result["input_hash"] or plan.get("schema_version") != SCHEMA or plan.get("simulation") is not True:
            fail("PLAN_SOURCE_INVALID")
        ports = {p["port_id"]: p for p in d["ports"]}
        fleet = {v["vehicle_id"]: v for v in d["vehicles"]}
        require_keys = (("charge_by_port_kw", ports), ("charge_by_vehicle_kw", fleet))
        for key, expected in require_keys:
            if not isinstance(plan[key], dict) or set(plan[key]) != set(expected):
                raise InputError("PLAN_CONTENT_INVALID", f"{key} keys must exactly match inputs")
            for values in plan[key].values():
                curve(values, key, -EPS)
        arrays = {}
        for key in ("pv_used_kw", "storage_charge_kw", "storage_discharge_kw", "net_import_kw"):
            arrays[key] = curve(plan[key], key, -EPS)
        energy = curve(plan["storage_energy_kwh"], "energy", -EPS, size=N+1)
        if not isinstance(plan["assignments"], list):
            raise InputError("PLAN_CONTENT_INVALID", "assignments must be a list")
        totals = {pid: [0.0]*N for pid in ports}
        occupied, seen = set(), set()
        delivered = {}
        for a in plan["assignments"]:
            vid, pid = a["vehicle_id"], a["port_id"]
            if vid not in fleet or pid not in ports or vid in seen:
                fail("ASSIGNMENT_ID_INVALID")
                continue
            seen.add(vid)
            v, p = fleet[vid], ports[pid]
            power = curve(a["power_kw"], "assignment power", -EPS)
            connected = a["connected_slots"]
            slots(connected, "connected slots")
            if not connected or connected != list(range(min(connected), max(connected)+1)):
                fail("NONCONTIGUOUS_CONNECTION", vehicle_id=vid)
            if pid not in v["allowed_port_ids"] or v["connector"] != p["connector"]:
                fail("CONNECTOR_OR_PORT_MISMATCH", vehicle_id=vid, port_id=pid)
            for t in connected:
                if not v["arrival_slot"] <= t < v["deadline_slot"]:
                    fail("VEHICLE_WINDOW", vehicle_id=vid, slot=t)
                if (pid, t) in occupied or t in p["occupied_slots"]:
                    fail("PORT_CONFLICT", port_id=pid, slot=t)
                occupied.add((pid, t))
            for t, kw in enumerate(power):
                limit = min(v["max_power_kw"], p["max_power_kw"][t]) if t in connected else 0
                if kw < -EPS or kw > limit + EPS:
                    fail("VEHICLE_POWER", vehicle_id=vid, slot=t, actual=kw, limit=limit)
                if abs(kw - plan["charge_by_vehicle_kw"][vid][t]) > EPS:
                    fail("VEHICLE_CURVE_MISMATCH", vehicle_id=vid, slot=t)
                totals[pid][t] += kw
            kwh = sum(power)*DT
            delivered[vid] = kwh
            if abs(kwh - v["required_energy_kwh"]) > EPS:
                fail("COMMITMENT_ENERGY", vehicle_id=vid, actual=kwh, required=v["required_energy_kwh"])
        if seen != set(fleet):
            fail("MISSING_VEHICLE")
        for pid, p in ports.items():
            for t in range(N):
                if abs(totals[pid][t] - plan["charge_by_port_kw"][pid][t]) > EPS:
                    fail("PORT_CURVE_MISMATCH", port_id=pid, slot=t)
                combined = totals[pid][t] + d["existing_charge_by_port_kw"][pid][t]
                if combined > p["max_power_kw"][t] + EPS:
                    fail("PORT_POWER_LIMIT", port_id=pid, slot=t, actual=combined, limit=p["max_power_kw"][t])
        s = d["storage"]
        floor = max(s["min_soc"], s["reserve_soc"]) * s["capacity_kwh"]
        ceiling = s["max_soc"] * s["capacity_kwh"]
        if abs(energy[0] - s["initial_energy_kwh"]) > EPS:
            fail("STORAGE_INITIAL", actual=energy[0], required=s["initial_energy_kwh"])
        if abs(energy[-1] - s["terminal_energy_kwh"]) > EPS:
            fail("STORAGE_TERMINAL", actual=energy[-1], required=s["terminal_energy_kwh"])
        for t, e in enumerate(energy):
            if e < floor-EPS or e > ceiling+EPS:
                fail("STORAGE_SOC_RESERVE", boundary=t, actual=e, minimum=floor, maximum=ceiling)
        residuals = []
        new_charge = [sum(v[t] for v in totals.values()) for t in range(N)]
        existing = [sum(v[t] for v in d["existing_charge_by_port_kw"].values()) for t in range(N)]
        for t in range(N):
            pc, pd, pv, grid = (arrays[k][t] for k in ("storage_charge_kw", "storage_discharge_kw", "pv_used_kw", "net_import_kw"))
            for name, actual, limit in (("STORAGE_CHARGE_POWER", pc, s["max_charge_kw"][t]),
                                        ("STORAGE_DISCHARGE_POWER", pd, s["max_discharge_kw"][t]),
                                        ("PV_LIMIT", pv, d["pv_available_kw"][t]),
                                        ("GRID_IMPORT_EXPORT", grid, d["grid_import_limit_kw"][t])):
                if actual < -EPS or actual > limit+EPS:
                    fail(name, slot=t, actual=actual, minimum=0, maximum=limit)
            if pc > EPS and pd > EPS:
                fail("STORAGE_SIMULTANEOUS", slot=t, charge_kw=pc, discharge_kw=pd)
            expected_energy = energy[t] + s["charge_efficiency"] * pc*DT - pd*DT/s["discharge_efficiency"]
            if abs(energy[t+1] - expected_energy) > EPS:
                fail("STORAGE_DYNAMICS", slot=t, actual=energy[t+1], expected=expected_energy)
            expected_grid = d["base_load_kw"][t] + existing[t] + new_charge[t] + pc - pd - pv
            residual = grid-expected_grid
            residuals.append(residual)
            if abs(residual) > EPS:
                fail("POWER_BALANCE", slot=t, actual=grid, expected=expected_grid)
            if d["new_charge_limit_kw"] is not None and new_charge[t] > d["new_charge_limit_kw"][t]+EPS:
                fail("NEW_CHARGE_BUDGET", slot=t, actual=new_charge[t], maximum=d["new_charge_limit_kw"][t])
        purchase = sum(g*r*DT for g, r in zip(arrays["net_import_kw"], d["rates_cny_per_kwh"]))
        throughput = sum(arrays["storage_charge_kw"] + arrays["storage_discharge_kw"])*DT
        curtailed = sum(a-u for a, u in zip(d["pv_available_kw"], arrays["pv_used_kw"]))*DT
        wear_cost = throughput*d["objective"]["throughput_cost_cny_per_kwh"]
        curtail_cost = curtailed*d["objective"]["curtailment_cost_cny_per_kwh"]
        loss = sum((pc*(1-s["charge_efficiency"]) + pd*(1/s["discharge_efficiency"]-1))*DT
                   for pc, pd in zip(arrays["storage_charge_kw"], arrays["storage_discharge_kw"]))
        result["metrics"] = {"purchase_cost_cny": purchase, "throughput_cost_cny": wear_cost,
            "curtailment_cost_cny": curtail_cost, "total_objective_cny": purchase+wear_cost+curtail_cost,
            "grid_import_kwh": sum(arrays["net_import_kw"])*DT, "pv_used_kwh": sum(arrays["pv_used_kw"])*DT,
            "pv_curtailed_kwh": curtailed, "storage_throughput_kwh": throughput, "storage_loss_kwh": loss,
            "storage_initial_kwh": energy[0], "storage_terminal_kwh": energy[-1],
            "peak_import_kw": max(arrays["net_import_kw"]), "delivered_energy_by_vehicle_kwh": delivered,
            "max_balance_residual_kw": max(abs(x) for x in residuals)}
    except (InputError, ValueError, TypeError, KeyError, IndexError, ArithmeticError) as e:
        fail("PLAN_OR_INPUT_INVALID", detail=str(e))
    result["passed"] = not errors
    return result


def simulation_gate(data, plan, *, allow_assumptions=False, live_input=None):
    """Both numeric and versioned simulation governance must pass on one hash.

    Returns eligibility for this offline artifact only, never a legacy plan or a
    device command. Caller cannot pass in a fabricated numeric 'passed' result.
    """
    numeric = verify_plan(data, plan, live_input=live_input)
    governance_errors = []
    try:
        d = validate_input(data)
        if d["assumptions"] and allow_assumptions is not True:
            governance_errors.append("EXPLICIT_ASSUMPTIONS_NOT_ACCEPTED")
        if plan.get("input_hash") != fingerprint(d) or plan.get("fingerprint") != fingerprint({k: v for k, v in plan.items() if k != "fingerprint"}):
            governance_errors.append("GOVERNANCE_HASH_MISMATCH")
    except (InputError, ValueError, TypeError, AttributeError):
        governance_errors.append("SIMULATION_SOURCE_INVALID")
    governance = {"passed": not governance_errors, "violations": governance_errors,
                  "rule_version": "a02-offline-governance-v1", "plan_fingerprint": numeric["plan_fingerprint"],
                  "input_hash": numeric["input_hash"], "checked_input_hash": numeric["checked_input_hash"]}
    return {"eligible_for_offline_simulation": numeric["passed"] and governance["passed"],
            "numeric": numeric, "governance": governance, "device_dispatch_authorized": False}
