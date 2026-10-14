"""Generate service schedules from orders and frozen resources; no preset plans.

Bounded backtracking over single-charger, contiguous 15-minute sessions. Each
session spreads exact integer Wh across its slots. Three search orderings produce
feasible alternatives, not a claim of global economic optimality.
"""
from decimal import Decimal

from .common import PocError, digest
from .computation import dec, economic_available


def assignment_violations(snap, candidate):
    """Independent reconstruction from vehicle assignments, also used by evaluation."""
    errors = []
    fleet = {v["vehicle_id"]: v for v in snap["vehicles"]}
    equipment = {c["asset_id"]: c for c in snap["equipment"]}
    totals = {key: [Decimal(0)]*4 for key in equipment}
    seen, occupied = set(), set()
    assignments = candidate.get("vehicle_assignments", [])
    for a in assignments:
        vid, cid = a["vehicle_id"], a["charger_id"]
        if vid not in fleet or cid not in equipment or vid in seen:
            errors.append("ASSIGNMENT_ID_INVALID")
            continue
        seen.add(vid)
        v, c = fleet[vid], equipment[cid]
        powers = [dec(p) for p in a["power_kw"]]
        active = [i for i, p in enumerate(powers) if p > 0]
        if not active or active != list(range(active[0], active[-1]+1)):
            errors.append("SERVICE_SESSION_INVALID")
        if v["connector"] != c["connector"]:
            errors.append("CONNECTOR_MISMATCH")
        for i, p in enumerate(powers):
            if p < 0 or p > dec(v["max_charge_power_kw"]):
                errors.append("VEHICLE_POWER_LIMIT")
            if p > 0:
                if not v["arrival_slot"] <= i < v["deadline_slot"]:
                    errors.append("SERVICE_WINDOW_VIOLATION")
                if (cid, i) in occupied or i in c.get("occupied_slots", []):
                    errors.append("CHARGER_SESSION_CONFLICT")
                occupied.add((cid, i))
            totals[cid][i] += p
        if sum(powers)*Decimal("0.25") != dec(v["requested_energy_kwh"]):
            errors.append("SERVICE_COMMITMENT_UNMET")
    if seen != set(fleet):
        errors.append("SERVICE_COMMITMENT_UNMET")
    if set(candidate["charger_power_kw"]) != set(equipment) or any(
        totals[cid] != [dec(p) for p in candidate["charger_power_kw"].get(cid, [])] for cid in equipment):
        errors.append("ASSIGNMENT_POWER_MISMATCH")
    return sorted(set(errors))


def generate(request, max_nodes=50000, charging_budget_kw=None):
    snap = request["station_state_snapshot"]
    fleet = sorted(snap["vehicles"], key=lambda v: (v["priority"] != "critical", v["deadline_slot"],
        v["soc_percent"], v["arrival_slot"], v["vehicle_id"]))
    equipment, e = snap["equipment"], snap["energy"]
    headroom = [dec(e["grid_import_limit_kw"])+dec(p)-dec(b) for p, b in zip(e["pv_available_kw"], e["base_load_kw"])]
    # The operator's simulated charging budget is distinct from the physical grid limit.
    if charging_budget_kw is not None:
        headroom = [min(h, dec(charging_budget_kw)) for h in headroom]
    if any(h < 0 for h in headroom):
        raise PocError("CAPACITY_LIMIT", "模拟基础负荷已超过光伏与并网容量，不能新增补能任务", "station_orchestrator")
    options = {}
    for v in fleet:
        choices = []
        wh = int(dec(v["requested_energy_kwh"])*1000)
        for c in equipment:
            if c["availability"] not in ("available", "derated") or c["asset_id"] in snap["safety"]["frozen_assets"] or c["connector"] != v["connector"]:
                continue
            for start in range(v["arrival_slot"], v["deadline_slot"]):
                for end in range(start+1, v["deadline_slot"]+1):
                    if any(i in c.get("occupied_slots", []) for i in range(start, end)):
                        continue
                    size = end-start
                    series = [Decimal(0)]*4
                    for i in range(start, end):
                        series[i] = Decimal(wh//size + (i-start < wh % size))*Decimal("0.004")
                    if max(series) <= min(dec(c["max_power_kw"]), dec(v["max_charge_power_kw"])):
                        choices.append({"vehicle_id": v["vehicle_id"], "charger_id": c["asset_id"], "power_kw": series})
        if not choices:
            raise PocError("SERVICE_REQUIREMENT_UNMET", f"车辆 {v['vehicle_id']}：在到离站窗口内无匹配的可用桩或充电能力不足；请调整补能量/时间或提供可用资源", "station_orchestrator")
        options[v["vehicle_id"]] = choices
    rates = snap["economic_context"]["rates_cny_per_kwh"] if economic_available(request) else [0]*4
    candidates, stats, fingerprints = [], [], set()
    for strategy in ("early", "cost", "green"):
        totals, occupied, nodes = [Decimal(0)]*4, set(), 0
        limited = False

        def search(index, chosen):
            nonlocal nodes, limited
            if index == len(fleet):
                return list(chosen)
            v = fleet[index]

            def key(a):
                powers = a["power_kw"]
                active = [i for i, p in enumerate(powers) if p]
                delta_grid = [max(Decimal(0), dec(e["base_load_kw"][i])+totals[i]+powers[i]-dec(e["pv_available_kw"][i])) -
                    max(Decimal(0), dec(e["base_load_kw"][i])+totals[i]-dec(e["pv_available_kw"][i])) for i in range(4)]
                cost = sum(delta_grid[i]*dec(rates[i]) for i in range(4))
                green = sum(powers)-sum(delta_grid)
                score = (active[-1], active[0]) if strategy == "early" or v["priority"] == "critical" else ((cost, -green) if strategy == "cost" else (-green, cost))
                return (*score, active[-1], active[0], a["charger_id"])

            for a in sorted(options[v["vehicle_id"]], key=key):
                nodes += 1
                if nodes > max_nodes:
                    limited = True
                    return None
                powers, cid = a["power_kw"], a["charger_id"]
                slots = {(cid, i) for i, p in enumerate(powers) if p}
                if slots & occupied or any(totals[i]+powers[i] > headroom[i] for i in range(4)):
                    continue
                occupied.update(slots)
                for i in range(4):
                    totals[i] += powers[i]
                found = search(index+1, chosen+[a])
                for i in range(4):
                    totals[i] -= powers[i]
                occupied.difference_update(slots)
                if found is not None:
                    return found
                if limited:
                    return None
            return None

        assignments = search(0, [])
        stats.append({"strategy": strategy, "visited_nodes": nodes, "limit_reached": limited, "found": assignments is not None})
        if assignments is None:
            continue
        charger_wh = {c["asset_id"]: [Decimal(0)]*4 for c in equipment}
        for a in assignments:
            for i, p in enumerate(a["power_kw"]):
                charger_wh[a["charger_id"]][i] += p
        candidate = {"charger_power_kw": {c: [float(p) for p in series] for c, series in charger_wh.items()},
            "storage_discharge_kw": [0]*4,
            "vehicle_assignments": [{**a, "power_kw": [float(p) for p in a["power_kw"]]} for a in assignments]}
        fingerprint = digest(candidate)
        if fingerprint not in fingerprints:
            fingerprints.add(fingerprint)
            candidate["candidate_id"] = "service-" + strategy + "-" + fingerprint[:8]
            if assignment_violations(snap, candidate):
                raise PocError("PROFILE_CALL_FAILED", "服务生成结果未通过独立分桩校验", "station_orchestrator", "FAILED")
            candidates.append(candidate)
    if not candidates:
        code = "SERVICE_SEARCH_LIMIT" if any(s["limit_reached"] for s in stats) else "SERVICE_REQUIREMENT_UNMET"
        reason = "搜索达到上限，尚不能判定是否存在可行方案" if code == "SERVICE_SEARCH_LIMIT" else "单桩连续服务模型内没有同时满足全部需求的方案，桩占用/站级容量/离站窗口存在冲突"
        raise PocError(code, reason + "；请调整需求或资源，未生成可执行承诺", "station_orchestrator")
    ids = {key: snap[key] for key in ("request_id", "snapshot_id", "station_id", "rule_version", "strategy_version")}
    details = []
    for c in candidates:
        rows = []
        for a in c["vehicle_assignments"]:
            v = next(v for v in fleet if v["vehicle_id"] == a["vehicle_id"])
            active = [i for i, p in enumerate(a["power_kw"]) if p]
            finish = (active[-1]+1)*15
            rows.append({"vehicle_id": v["vehicle_id"], "charger_id": a["charger_id"],
                "start_minutes": active[0]*15, "completion_minutes": finish,
                "wait_minutes": (active[0]-v["arrival_slot"])*15, "slack_minutes": v["deadline_slot"]*15-finish,
                "risk": "no_slack" if finish == v["deadline_slot"]*15 else "within_mock_window",
                "delivered_energy_kwh": v["requested_energy_kwh"], "power_kw": a["power_kw"],
                "priority_reason": "关键任务优先，然后离站期限、SOC 和到站顺序；所有已填写补能需求均须满足"})
        details.append({"candidate_id": c["candidate_id"], "vehicles": rows})
    return {**ids, "service_plan_id": "generated-service-" + digest({"snapshot": snap, "candidates": candidates})[:16],
        "issued_at": snap["generated_at"], "expires_at": snap["expires_at"], "data_quality_summary": snap["data_quality_summary"],
        "producer_id": "charging-service-orchestrator", "producer_version": "bounded-service-search-v1",
        "source_demand_hash": digest(request["user_demand"]), "source_snapshot_hash": digest(snap),
        "service_queue": [v["vehicle_id"] for v in fleet], "assignment_candidates": candidates, "candidate_details": details,
        "service_boundary_ref": request["poc_context"]["capacity_result"]["result_id"], "search": stats,
        "unresolved_items": ["仅比较搜索生成的候选，不保证全局经济最优；完成时间是模拟估算"] + snap.get("simulation_assumptions", []),
        "audit_refs": ["mock-service-generator-v1"]}
