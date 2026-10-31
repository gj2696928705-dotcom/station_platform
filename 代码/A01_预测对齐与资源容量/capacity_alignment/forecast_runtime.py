"""Task boundary for A01. No inference or model paths are accepted from HTTP."""
from copy import deepcopy
from datetime import timedelta

from .forecast_capacity import MODES, fail, freeze, verify
from .common import digest, now, parse_time


def request_hash(request):
    return digest({k: v for k, v in request.items() if k not in {"forecast_context", "service_orchestration_plan"}})


def attach(request, config, task_id, version):
    """Freeze once at task creation. Persist the returned request with the task."""
    request = deepcopy(request)
    if "forecast_context" in request:
        fail("CONTEXT_ALREADY_FROZEN", "同一请求不得重复挂接预测")
    mode = config.get("mode", "disabled")
    if mode not in MODES:
        fail("FORECAST_MODE_INVALID", "预测模式须为 disabled/shadow/simulation_use")
    if mode == "disabled":
        return request
    snap = request["station_state_snapshot"]
    window = request["poc_context"]["evaluation_window"]
    if request["poc_context"].get("simulation") is not True:
        fail("SIMULATION_REQUIRED", "预测消费仅支持模拟任务")
    start = parse_time(window["effective_from"])
    supplied = deepcopy(config.get("inputs", {}))
    boundary = supplied.get("meter_boundary")
    state = {"station_id": snap["station_id"], "meter_boundary": boundary, "simulation": True,
             "issued_at": snap.get("source_measured_at", snap["generated_at"]),
             "available_at": snap.get("source_measured_at", snap["generated_at"]),
             "expires_at": snap.get("source_expires_at", snap["expires_at"]), "quality": snap.get("source_quality", snap["data_quality_summary"]),
             "source_state_id": snap.get("source_state_id", snap["snapshot_id"]), "site_revision": snap.get("site_revision"),
             "grid_import_limit_kw": snap["energy"]["grid_import_limit_kw"],
             **{key: snap["safety"][key] for key in ("protection_active", "manual_takeover", "limits_known")},
             "devices": [{"asset_id": d["asset_id"], "max_power_kw": d["max_power_kw"],
                          "availability": d["availability"], "frozen": d["asset_id"] in snap["safety"]["frozen_assets"],
                          "occupied_intervals": [{"start": (start+timedelta(minutes=15*i)).isoformat(),
                                                  "end": (start+timedelta(minutes=15*(i+1))).isoformat()}
                                                 for i in d["occupied_slots"]]} for d in snap["equipment"]]}
    supplied.update({"task_id": task_id, "task_version": version, "station_id": snap["station_id"],
                     "as_of": snap["generated_at"], "start": window["effective_from"], "end": window["effective_to"],
                     "interval_minutes": request["poc_context"]["interval_minutes"], "state": state,
                     "new_session_ids": [v["vehicle_id"] for v in snap["vehicles"]]})
    context = freeze(supplied)
    if len(context["slots"]) != 4 and context["status"] == "available":
        fail("WINDOW_INVALID", "现有编排器只支持未来一小时四段")
    if mode == "simulation_use" and context["status"] == "available":
        slots = context["slots"]
        snap["energy"].update({"base_load_kw": [s["existing_load_kw"] for s in slots],
                               "pv_available_kw": [s["pv_available_kw"] for s in slots]})
        snap["energy"]["storage"]["dispatch_mode"] = "idle"
        for device in snap["equipment"]:
            device["occupied_slots"] = [i for i, slot in enumerate(slots)
                                        if next(d for d in slot["sources"]["devices"] if d["asset_id"] == device["asset_id"])["occupied"]]
        # Forecast and measurement expiries are independent from the horizon.
        snap["expires_at"] = context["expires_at"]
        request["poc_context"]["capacity_result"]["expires_at"] = context["expires_at"]
        request["poc_context"]["capacity_result"]["result_id"] = "capacity-" + context["context_hash"][:20]
    wrapper = {"mode": mode, "context": context, "request_hash": request_hash(request)}
    wrapper["wrapper_hash"] = digest(wrapper)
    request["forecast_context"] = wrapper
    return request


def decision_context(request, require_available=True):
    wrapper = request.get("forecast_context")
    if not wrapper:
        return None
    if wrapper.get("wrapper_hash") != digest({k: v for k, v in wrapper.items() if k != "wrapper_hash"}) or wrapper["request_hash"] != request_hash(request):
        fail("CONTEXT_HASH_MISMATCH", "冻结请求或预测模式被修改")
    if wrapper["mode"] not in {"shadow", "simulation_use"}:
        fail("FORECAST_MODE_INVALID", "冻结模式无效")
    context = verify(wrapper["context"])
    if wrapper["mode"] == "simulation_use":
        if require_available and context["status"] != "available":
            fail("FORECAST_CONTEXT_UNAVAILABLE", "仿真预测上下文不可用：" + ",".join(i["code"] for i in context["issues"]))
        if require_available and is_expired(context):
            fail("FORECAST_CONTEXT_EXPIRED", "冻结上下文已失效，需要新的任务版本及输入")
        return context
    return None


def is_expired(context):
    return context["inputs"].get("execution_mode") == "simulation" and context.get("expires_at") is not None and parse_time(context["expires_at"]) <= now()


def context_metadata(request):
    context = decision_context(request, require_available=False)
    wrapper = request.get("forecast_context")
    if not wrapper:
        return {"forecast_status": "not_used", "forecast_mode": "disabled", "frozen_context_hash": None}
    return {"forecast_status": ("expired" if is_expired(context) else context["status"]) if context else "not_used", "forecast_mode": wrapper["mode"],
            "frozen_context_hash": context["context_hash"] if context else None,
            "forecast_diagnostics": {"context_hash": wrapper["context"]["context_hash"],
                                     "status": wrapper["context"]["status"], "issues": wrapper["context"]["issues"]}}
