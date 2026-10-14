"""A transparent, deterministic mock, not a certified optimizer or safety service.

Four 15-minute slots, no export, generated or legacy fixed assignments, idle storage.
All arithmetic for cost/energy uses Decimal. Feasibility only compares the supplied
synthetic candidates against the supplied synthetic boundaries; it never clips them.
"""
from decimal import Decimal

from .common import PocError, digest, now, parse_time


def dec(value):
    if isinstance(value, bool):
        raise ValueError("boolean is not a number")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("non-finite number")
    return result


def number(value):
    return float(value.quantize(Decimal("0.000001")))


def economic_available(request):
    ctx = request["station_state_snapshot"].get("economic_context")
    window = request["poc_context"]["evaluation_window"]
    try:
        for key in ("tariff_config_ref", "tariff_version", "cost_tolerance_ref", "cost_tolerance_version", "joint_approval_ref"):
            if not isinstance(ctx[key], str) or not ctx[key].startswith("mock-"):
                return False
        return (ctx["data_quality_summary"] in ("A", "B")
            and parse_time(ctx["effective_from"]) <= parse_time(window["effective_from"])
            and parse_time(ctx["effective_to"]) >= parse_time(window["effective_to"])
            and parse_time(ctx["effective_to"]) > now()
            and len(ctx["rates_cny_per_kwh"]) == 4
            and all(dec(rate) >= 0 for rate in ctx["rates_cny_per_kwh"])
            and dec(ctx["cost_tolerance_cny"]) >= 0)
    except (KeyError, TypeError, ValueError, ArithmeticError, PocError):
        return False


def evaluate(request, energy_recommendation):
    snap, poc, service = (request[k] for k in ("station_state_snapshot", "poc_context", "service_orchestration_plan"))
    ctx, e = snap.get("economic_context"), snap["energy"]
    economic = economic_available(request)
    green = poc["green_data_available"]
    rows, exact_costs = [], {}
    for candidate in energy_recommendation["power_plan_candidates"]:
        violations = []
        chargers = candidate["charger_power_kw"]
        for equipment in snap["equipment"]:
            series = chargers[equipment["asset_id"]]
            if any(dec(p) > dec(equipment["max_power_kw"]) for p in series):
                violations.append("CHARGER_POWER_LIMIT")
            if (equipment["availability"] not in ("available", "derated") or equipment["asset_id"] in snap["safety"]["frozen_assets"]) and any(series):
                violations.append("DEVICE_UNAVAILABLE")
        if "user_demand" in request:
            from .service import assignment_violations
            violations.extend(assignment_violations(snap, candidate))
        else:
            for vehicle in snap["vehicles"]:
                supplied = sum(dec(p) for p in chargers[vehicle["charger_id"]][:vehicle["deadline_slot"]]) * Decimal("0.25")
                if supplied < dec(vehicle["requested_energy_kwh"]):
                    violations.append("SERVICE_COMMITMENT_UNMET")
        if any(candidate["storage_discharge_kw"]):
            violations.append("STORAGE_IDLE_BOUNDARY")
        grid, consumed, curtailed = [], [], []
        for i in range(4):
            load = sum(dec(p[i]) for p in chargers.values()) + dec(e["base_load_kw"][i])
            pv = dec(e["pv_available_kw"][i])
            purchase = max(Decimal(0), load-pv)
            if purchase > dec(e["grid_import_limit_kw"]):
                violations.append("GRID_CAPACITY_LIMIT")
            grid.append(purchase * Decimal("0.25"))
            consumed.append(min(load, pv) * Decimal("0.25"))
            curtailed.append(max(Decimal(0), pv-load) * Decimal("0.25"))
        cost = sum(kwh*dec(rate) for kwh, rate in zip(grid, ctx["rates_cny_per_kwh"])) if economic else None
        exact_costs[candidate["candidate_id"]] = cost
        rows.append({"candidate_id": candidate["candidate_id"], "feasible": not violations,
                     "violations": sorted(set(violations)), "grid_import_kwh": number(sum(grid)),
                     "purchase_cost_cny": number(cost) if cost is not None else None,
                     "local_green_kwh": number(sum(consumed)) if green else None,
                     "curtailed_kwh": number(sum(curtailed)) if green else None})
    feasible = [r for r in rows if r["feasible"]]
    if not feasible:
        raise PocError("CAPACITY_LIMIT", "给定候选均不满足模拟边界或服务承诺；需要新的服务编排或人工处理", "station_control")
    minimum = min(exact_costs[r["candidate_id"]] for r in feasible) if economic else None
    if economic:
        tolerance = dec(ctx["cost_tolerance_cny"])
        eligible = [r for r in feasible if exact_costs[r["candidate_id"]] <= minimum+tolerance]
        if green:
            selected = min(eligible, key=lambda r: (-r["local_green_kwh"], r["curtailed_kwh"], exact_costs[r["candidate_id"]], r["candidate_id"]))
        else:
            selected = min(feasible, key=lambda r: (exact_costs[r["candidate_id"]], r["candidate_id"]))
    else:
        baseline = poc.get("baseline")
        try:
            valid = (baseline["simulation"] is True and baseline["snapshot_id"] == snap["snapshot_id"]
                and baseline["ref"].startswith("mock-")
                and parse_time(baseline["expires_at"]) >= parse_time(poc["evaluation_window"]["effective_to"])
                and parse_time(baseline["expires_at"]) > now())
        except (KeyError, TypeError, AttributeError, PocError):
            valid = False
        eligible = [r for r in feasible if valid and r["candidate_id"] == baseline["candidate_id"]]
        if not eligible:
            raise PocError("ECONOMIC_CONFIG_UNAVAILABLE", "经济配置不可用，且不存在覆盖本窗口的可行模拟基线", "station_control")
        selected = eligible[0]
    computation_ref = "mock-objective-" + digest({"request": request, "energy": energy_recommendation})[:20]
    audit_ref = computation_ref + "-candidates"
    status = "economic_unavailable" if not economic else "available" if green else "green_unavailable"
    reason = {"available": "先过滤边界与服务承诺，在成本容差内按绿电消纳、弃光量、成本、候选标识依次选择",
              "green_unavailable": "绿电目标数据不可用，保留购电成本最低的可行候选",
              "economic_unavailable": "经济配置不可用，采用有效的模拟基线；不声称经济最优"}[status]
    objective = {"status": status, "evaluation_window": poc["evaluation_window"],
        "candidate_id": selected["candidate_id"],
        "purchase_cost": {"value": selected["purchase_cost_cny"], "unit": "CNY"} if economic else None,
        "cost_minimum_reference": {"value": number(minimum), "unit": "CNY", "audit_ref": audit_ref} if economic else None,
        "cost_tolerance_assessment": {"config_ref": ctx["cost_tolerance_ref"] if economic else None,
            "within_tolerance": True if economic else None,
            "tolerance_cny": ctx["cost_tolerance_cny"] if economic else None},
        "local_green_consumption": {"value": selected["local_green_kwh"], "unit": "kWh"} if green else None,
        "curtailment_risk": {"level": "none" if selected["curtailed_kwh"] == 0 else "present",
            "curtailed_kwh": selected["curtailed_kwh"], "basis": "mock four-slot energy balance"} if green else None,
        "selection_reason": reason, "computation_ref": computation_ref, "audit_refs": [audit_ref]}
    return {"simulation": True, "computation_version": "mock-objective-v1", "computation_ref": computation_ref,
            "request_hash": digest(request), "energy_hash": digest(energy_recommendation),
            "snapshot_id": snap["snapshot_id"], "energy_recommendation_ref": energy_recommendation["energy_recommendation_id"],
            "service_plan_ref": service["service_plan_id"], "rule_version": snap["rule_version"],
            "strategy_version": snap["strategy_version"], "expires_at": snap["expires_at"],
            "candidate_evaluations": rows, "selected_candidate": next(c for c in energy_recommendation["power_plan_candidates"] if c["candidate_id"] == selected["candidate_id"]),
            "objective_evaluation": objective, "audit_refs": [audit_ref]}
