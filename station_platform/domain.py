from copy import deepcopy
from datetime import timedelta
from decimal import Decimal
from .legacy.demand import MockStationAdapter, validate_demand
from .legacy.service import generate, assignment_violations
from .legacy.computation import evaluate, dec
from .legacy.common import PocError
from .util import PlatformError, digest, finite, now, parse_time, stamp

def validate_payload(payload):
    if not isinstance(payload, dict) or payload.get("type") not in {"tariff", "vpp"}:
        raise PlatformError("TASK_INVALID", "任务类型需要为电价规划或虚拟电厂模拟")
    try:
        validate_demand(payload.get("demand"))
    except PocError as error:
        raise PlatformError(error.code, error.message) from None
    if payload["type"] == "vpp":
        if payload.get("business_type", "charging_budget") != "charging_budget":
            raise PlatformError("BUSINESS_UNSUPPORTED", "首期仅支持充电功率预算模拟，调频及真实辅助服务未接入")
        finite(payload.get("budget_kw"), 0, 2000)
    return deepcopy(payload)

def request_from_site(demand, state):
    if not state or not state.get("simulation"):
        raise PlatformError("STATE_UNAVAILABLE", "需要有效的模拟站点状态", 409)
    if parse_time(state["expires_at"]) <= now() or state.get("quality") not in {"A", "B"}:
        raise PlatformError("STATE_EXPIRED", "场站量测过期或质量不足", 409)
    site = deepcopy(state["site"])
    if site["safety"]["protection_active"] or site["safety"]["manual_takeover"] or not site["safety"]["limits_known"]:
        raise PlatformError("LOCAL_RESTRICTED", "现场保护、接管或边界不明，不能新增计划", 409)
    try:
        request = MockStationAdapter(site).capture(demand)
    except PocError as error:
        raise PlatformError(error.code, error.message) from None
    request["station_state_snapshot"]["source_state_id"] = state["id"]
    request["station_state_snapshot"]["site_revision"] = state["revision"]
    request["station_state_snapshot"]["source_measured_at"] = state["measured_at"]
    return request

def capability(state):
    site = state["site"]
    usable = [c for c in site["equipment"] if c["availability"] in {"available", "derated"}
              and c["asset_id"] not in site["safety"]["frozen_assets"]]
    valid = parse_time(state["expires_at"]) > now() and state["quality"] in {"A", "B"}
    restricted = site["safety"]["protection_active"] or site["safety"]["manual_takeover"] or not site["safety"]["limits_known"]
    headroom = [max(0, min(sum(c["max_power_kw"] for c in usable), site["energy"]["grid_import_limit_kw"] + p - b))
                for p, b in zip(site["energy"]["pv_available_kw"], site["energy"]["base_load_kw"])]
    return {"station_id": state["station_id"], "state_ref": state["id"], "revision": state["revision"],
            "issued_at": state["measured_at"], "expires_at": state["expires_at"], "simulation": True,
            "charging_headroom_kw": headroom if valid and not restricted else [0]*4,
            "duration_minutes": 60, "interval_minutes": 15, "storage_dispatch": "idle",
            "direction_basis": "可新增充电负荷上限；不是已认证上/下调能力",
            "usable": valid and not restricted, "forecast_status": "not_used"}

def resources(request):
    snap = request["station_state_snapshot"]
    energy = snap["energy"]
    return {"energy_recommendation_id": "resource-" + digest(snap)[:20], "snapshot_id": snap["snapshot_id"],
            "simulation": True, "forecast_status": "not_used", "storage_dispatch": "idle",
            "grid_import_limit_kw": energy["grid_import_limit_kw"],
            "charging_headroom_kw": [max(0, energy["grid_import_limit_kw"]+p-b)
                for p, b in zip(energy["pv_available_kw"], energy["base_load_kw"])],
            "expires_at": snap["expires_at"], "source_ref": snap["source_state_id"]}

def schedule(request, budget=None):
    try:
        return generate(request, charging_budget_kw=budget)
    except PocError as error:
        raise PlatformError(error.code, error.message, 409) from None

def evaluation(request, resource, service, budget=None):
    work = deepcopy(request)
    work["service_orchestration_plan"] = service
    energy = {**resource, "power_plan_candidates": service["assignment_candidates"]}
    try:
        result = evaluate(work, energy)
    except PocError as error:
        raise PlatformError(error.code, error.message, 409) from None
    if budget is not None and any(sum(series[i] for series in result["selected_candidate"]["charger_power_kw"].values()) > budget+1e-6 for i in range(4)):
        raise PlatformError("BUDGET_EXCEEDED", "候选超过本次模拟充电预算", 409)
    return result

def check_local(plan, state, check_expiry=True):
    errors = []
    if plan.get("simulation") is not True or plan.get("station_id") != state["station_id"]:
        errors.append("PLAN_SOURCE_INVALID")
    expected = digest({k: plan[k] for k in ["task_id", "version", "station_id", "request", "selected_candidate", "budget_kw"]})
    if expected != plan.get("fingerprint"):
        errors.append("PLAN_HASH_MISMATCH")
    if parse_time(state["expires_at"]) <= now() or state["quality"] not in {"A", "B"}:
        errors.append("LIVE_STATE_INVALID")
    request = deepcopy(plan["request"])
    if check_expiry and parse_time(request["station_state_snapshot"]["expires_at"]) <= now():
        errors.append("PLAN_EXPIRED")
    live = deepcopy(request["station_state_snapshot"])
    site = state["site"]
    live.update({k: deepcopy(site[k]) for k in ["energy", "equipment", "safety"]})
    if site["safety"]["protection_active"] or site["safety"]["manual_takeover"] or not site["safety"]["limits_known"]:
        errors.append("LOCAL_RESTRICTED")
    candidate = plan["selected_candidate"]
    try:
        errors.extend(assignment_violations(live, candidate))
        if set(candidate["charger_power_kw"]) != {c["asset_id"] for c in live["equipment"]}:
            errors.append("ASSET_CHANGED")
        for c in live["equipment"]:
            series = candidate["charger_power_kw"].get(c["asset_id"], [0]*4)
            if len(series) != 4 or any(dec(p) < 0 or dec(p) > dec(c["max_power_kw"]) for p in series):
                errors.append("CHARGER_POWER_LIMIT")
            if c["availability"] not in {"available", "derated"} or c["asset_id"] in live["safety"]["frozen_assets"]:
                if any(series): errors.append("DEVICE_UNAVAILABLE")
        if any(candidate["storage_discharge_kw"]): errors.append("STORAGE_IDLE_BOUNDARY")
        for i in range(4):
            charging = sum(dec(p[i]) for p in candidate["charger_power_kw"].values())
            grid = max(Decimal(0), charging+dec(site["energy"]["base_load_kw"][i])-dec(site["energy"]["pv_available_kw"][i]))
            if grid > dec(site["energy"]["grid_import_limit_kw"]): errors.append("GRID_CAPACITY_LIMIT")
            if plan["budget_kw"] is not None and charging > dec(plan["budget_kw"]): errors.append("BUDGET_EXCEEDED")
    except (ValueError, KeyError, TypeError, ArithmeticError, IndexError):
        errors.append("PLAN_CONTENT_INVALID")
    return {"passed": not errors, "violations": sorted(set(errors)), "checked_at": stamp(),
            "state_ref": state["id"], "site_revision": state["revision"], "plan_ref": plan["id"],
            "simulation": True, "certified_safety_service": False}
