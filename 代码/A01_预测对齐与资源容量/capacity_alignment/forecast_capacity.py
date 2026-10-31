"""A01: frozen, simulation-only interval capacity; standard library only.

All timestamps are aware, intervals are half open, and missing data is never zero.
Native values are held constant inside their declared intervals. This does not
create sub-interval forecast accuracy or a certified dispatch capability.
"""
from copy import deepcopy
from datetime import timedelta
from decimal import Decimal
import math

from .common import CapacityError as PlatformError, digest, parse_time

SCHEMA = "simulation-capacity-1"
BASE = "base_load_excluding_charging_pv_storage"
PV = "pv_available_ac"
MODES = {"disabled", "shadow", "simulation_use"}


def fail(code, message):
    raise PlatformError(code, message, 409)


def number(value):
    if isinstance(value, bool):
        fail("NUMBER_INVALID", "布尔值不能作为功率")
    try:
        result = Decimal(str(value))
        if not result.is_finite() or result < 0:
            raise ValueError()
        return result
    except (ValueError, ArithmeticError):
        fail("NUMBER_INVALID", "功率/电量须为非负有限值")


def seconds(delta):
    return Decimal(delta.days * 86400 + delta.seconds) + Decimal(delta.microseconds) / 1000000


def as_float(value):
    converted = float(value)
    if not math.isfinite(converted):
        fail("NUMBER_INVALID", "数值超出有限 JSON 数值的可表示范围")
    return converted


def identity(record, inputs):
    if record.get("station_id") != inputs["station_id"]:
        fail("STATION_MISMATCH", "预测、承诺及设备必须属于同一模拟站点")
    if record.get("meter_boundary") != inputs["meter_boundary"]:
        fail("METER_BOUNDARY_MISMATCH", "计量边界不一致")
    if record.get("simulation") is not True:
        fail("SIMULATION_REQUIRED", "本接口仅接收明确标记的模拟输入")


def visible(record, as_of):
    if parse_time(record["issued_at"]) > parse_time(record["available_at"]):
        fail("VISIBILITY_INVALID", "可见时间不能早于发布时间")
    for field in ("issued_at", "available_at"):
        if parse_time(record[field]) > as_of:
            fail("LATE_INPUT", f"{field} 晚于 as_of")
    if parse_time(record["expires_at"]) <= as_of:
        fail("INPUT_EXPIRED", "输入在决策时已失效")


def intervals(record):
    """Validate the complete native record before any clipping or projection."""
    unit, semantics = record["unit"], record["value_semantics"]
    if (unit, semantics) not in {("kW", "interval_average_power"), ("kWh", "interval_energy")}:
        fail("UNIT_SEMANTICS_MISMATCH", "须显式声明区间平均 kW 或区间电量 kWh")
    rows = []
    for index, point in enumerate(record["values"]):
        start, end = parse_time(point["start"]), parse_time(point["end"])
        if start >= end:
            fail("INTERVAL_INVALID", "原生区间须满足 start < end")
        if point.get("unit", unit) != unit:
            fail("UNIT_SEMANTICS_MISMATCH", "同一序列存在单位混用")
        power = number(point["value"])
        if unit == "kWh":
            power = power * 3600 / seconds(end-start)
        if "source_observation_end" in point and parse_time(point["source_observation_end"]) > parse_time(record["issued_at"]):
            fail("FUTURE_OBSERVATION", "基线读取了发布时尚未完成的历史区间")
        rows.append((start, end, power, index))
    rows.sort(key=lambda row: row[0])
    if any(left[1] > right[0] for left, right in zip(rows, rows[1:])):
        fail("NATIVE_OVERLAP", "原生预测区间重叠或重复")
    return rows


def project(rows, start, end, ref):
    weighted, covered, sources = Decimal(0), Decimal(0), []
    for left, right, power, index in rows:
        overlap = seconds(min(end, right)-max(start, left))
        if overlap > 0:
            weighted += power * overlap
            covered += overlap
            sources.append({"record_id": ref, "index": index, "overlap_seconds": as_float(overlap),
                            "native_power_kw": as_float(power)})
    if covered != seconds(end-start):
        fail("COVERAGE_INCOMPLETE", f"{ref} 未完整覆盖 [{start.isoformat()},{end.isoformat()})")
    return weighted / covered, sources


def freeze(inputs):
    """Return a JSON-serializable immutable-by-hash context, including failures.

    A failure invalidates the entire window: no slot advertises usable capacity.
    The raw inputs and overlap contributions remain available for independent QA.
    """
    inputs = deepcopy(inputs)
    invalid_json = False
    try:
        digest(inputs)
    except (ValueError, TypeError):
        invalid_json = True
        def safe(value):
            if isinstance(value, dict): return {str(k): safe(v) for k,v in value.items()}
            if isinstance(value, (list, tuple)): return [safe(v) for v in value]
            if isinstance(value, float) and not math.isfinite(value): return {"invalid_numeric": str(value)}
            if value is None or isinstance(value, (str, int, float, bool)): return value
            return {"invalid_type": type(value).__name__, "representation": str(value)}
        inputs = safe(inputs)
    result = {"schema_version": SCHEMA, "simulation": True, "real_dispatch_eligible": False,
              "storage_dispatch": "idle", "status": "unavailable", "issues": [], "slots": [],
              "inputs": inputs, "input_hash": digest(inputs), "expires_at": None,
              "projection": "constant_native_interval_overlap_weighted",
              "limitations": ["模拟点预测容量；不保证未来实况或区间覆盖率", "储能 idle；端口按独占会话保守占用"]}
    try:
        if invalid_json:
            fail("NUMBER_INVALID", "输入含非有限数值或非 JSON 数据；原始非法项已标记保留")
        _calculate(inputs, result)
    except (PlatformError, KeyError, TypeError, ValueError, AttributeError, ArithmeticError) as error:
        result["issues"].append({"code": getattr(error, "code", "INPUT_INVALID"), "message": str(error)})
    if result["issues"]:
        result["status"] = "unavailable"
        for slot in result["slots"]:
            slot["capacity_kw"] = None
            if slot["status"] == "available":
                slot["status"] = "window_unavailable"
    result["context_hash"] = digest(result)
    return result


def _calculate(inputs, result):
    if inputs.get("execution_mode") not in {"offline_replay", "simulation"}:
        fail("SIMULATION_REQUIRED", "必须显式选择模拟或离线回放时钟")
    if not inputs["station_id"] or not inputs["meter_boundary"] or not inputs["task_id"] or type(inputs["task_version"]) is not int or inputs["task_version"] < 1:
        fail("IDENTITY_INVALID", "缺少站点、计量边界或任务版本")
    as_of, start, end = (parse_time(inputs[key]) for key in ("as_of", "start", "end"))
    if start < as_of or start >= end or inputs["interval_minutes"] != 15 or seconds(end-start) % 900:
        fail("WINDOW_INVALID", "业务窗口须在 as_of 之后，由完整的 15 分钟段组成")
    if seconds(end-start) > 86400:
        fail("WINDOW_INVALID", "单次最多 96 段")
    binding, state = inputs["binding"], inputs["state"]
    for record in (binding, state):
        identity(record, inputs)
        visible(record, as_of)
    if not binding["evidence_ref"]:
        fail("BINDING_REQUIRED", "缺少模拟站点和计量绑定证据")
    if state["quality"] not in {"A", "B"}:
        fail("STATE_QUALITY_INVALID", "量测质量不足")
    if state["protection_active"] or state["manual_takeover"] or not state["limits_known"]:
        fail("LOCAL_RESTRICTED", "现场保护、接管或边界不明")
    grid_limit = number(state["grid_import_limit_kw"])
    devices = {}
    for device in state["devices"]:
        asset = device["asset_id"]
        if not asset or asset in devices:
            fail("DEVICE_DUPLICATE", "设备 ID 缺失或重复，禁止重复累计容量")
        number(device["max_power_kw"])
        if device["availability"] not in {"available", "derated", "unavailable", "offline", "fault", "maintenance"} or type(device["frozen"]) is not bool:
            fail("DEVICE_STATE_INVALID", "设备状态未知")
        for occupied in device["occupied_intervals"]:
            if parse_time(occupied["start"]) >= parse_time(occupied["end"]):
                fail("INTERVAL_INVALID", "端口占用区间无效")
        devices[asset] = device
    expiries = [parse_time(binding["expires_at"]), parse_time(state["expires_at"]), end]
    forecasts = []
    for key, semantics in (("load_forecast", BASE), ("pv_forecast", PV)):
        record = inputs[key]
        identity(record, inputs)
        visible(record, as_of)
        if record["status"] != "available":
            fail("FORECAST_UNAVAILABLE", f"{key}: {record.get('reason', record['status'])}")
        if record["target_semantics"] != semantics:
            fail("TARGET_SEMANTICS_MISMATCH", "基础负荷必须排除全部充电/PV/储能；PV 须为 AC 可用出力")
        for field in ("record_id", "model_version", "method_hash", "input_hash", "source_ref"):
            if not record[field]:
                fail("PROVENANCE_REQUIRED", f"{key} 缺少 {field}")
        if record.get("sampling_layout") == "day13_07_19":
            weather = parse_time(record["weather_issued_at"])
            cutoff = parse_time(record["values"][0]["start"])
            if weather > min(as_of, cutoff):
                fail("LATE_WEATHER", "day13 天气批次晚于 as_of 或当日 07:00")
        rows = intervals(record)
        forecasts.append((record, rows))
        expiries.append(parse_time(record["expires_at"]))
    commitments = inputs["commitments"]
    identity(commitments, inputs)
    visible(commitments, as_of)
    if commitments["status"] not in {"known", "known_empty"}:
        fail("COMMITMENT_UNKNOWN", "既有预约仅有电量或状态未知，不能推断逐段功率为零")
    if commitments["status"] == "known_empty" and commitments["sessions"]:
        fail("COMMITMENT_STATUS_CONFLICT", "空承诺声明与会话列表冲突")
    if parse_time(commitments["coverage_start"]) > start or parse_time(commitments["coverage_end"]) < end:
        fail("COMMITMENT_COVERAGE_INCOMPLETE", "承诺清单未完整覆盖业务窗口")
    if not commitments["source_ref"]:
        fail("PROVENANCE_REQUIRED", "缺少承诺清单来源")
    expiries.append(parse_time(commitments["expires_at"]))
    sessions, seen, by_asset = [], set(), {}
    for session in commitments["sessions"]:
        sid, asset = session["session_id"], session["asset_id"]
        if not sid or sid in seen:
            fail("COMMITMENT_DUPLICATE", "同一既有会话被重复计入")
        if asset not in devices:
            fail("COMMITMENT_DEVICE_UNKNOWN", "既有承诺引用未知设备")
        if sid in inputs.get("new_session_ids", []):
            fail("COMMITMENT_NEW_DUPLICATE", "既有承诺与本次新增任务重复")
        seen.add(sid)
        visible(session, as_of)
        expiries.append(parse_time(session["expires_at"]))
        if session["target_semantics"] != "committed_charge_ac":
            fail("TARGET_SEMANTICS_MISMATCH", "承诺功率须为充电 AC 支路功率")
        rows = intervals(session)
        # Even explicit zero power needs full coverage: no implied zero padding.
        for left, right, power, index in rows:
            if power and (power > number(devices[asset]["max_power_kw"]) or devices[asset]["availability"] not in {"available", "derated"} or devices[asset]["frozen"]):
                fail("EXISTING_DEVICE_CONFLICT", "既有承诺超过当前设备能力")
            if power:
                for other_left, other_right in by_asset.get(asset, []):
                    if min(right, other_right) > max(left, other_left):
                        fail("EXISTING_SESSION_CONFLICT", "同一端口的既有会话重叠")
                by_asset.setdefault(asset, []).append((left, right))
        sessions.append((session, rows))
    result["expires_at"] = min(expiries).isoformat()
    slot_start = start
    while slot_start < end:
        slot_end = slot_start + timedelta(minutes=15)
        slot = {"start": slot_start.isoformat(), "end": slot_end.isoformat(), "unit": "kW",
                "status": "unavailable", "capacity_kw": None, "sources": {}}
        result["slots"].append(slot)
        try:
            base, base_sources = project(forecasts[0][1], slot_start, slot_end, forecasts[0][0]["record_id"])
            pv, pv_sources = project(forecasts[1][1], slot_start, slot_end, forecasts[1][0]["record_id"])
            committed, commit_sources, occupied = Decimal(0), [], set()
            for session, rows in sessions:
                power, sources = project(rows, slot_start, slot_end, session["session_id"])
                committed += power
                commit_sources.extend(sources)
                if power:
                    occupied.add(session["asset_id"])
            device_available = Decimal(0)
            device_sources = []
            for asset, device in devices.items():
                busy = asset in occupied or any(min(slot_end, parse_time(row["end"])) > max(slot_start, parse_time(row["start"])) for row in device["occupied_intervals"])
                usable = not busy and not device["frozen"] and device["availability"] in {"available", "derated"}
                capacity = number(device["max_power_kw"]) if usable else Decimal(0)
                device_available += capacity
                device_sources.append({"asset_id": asset, "occupied": busy, "new_limit_kw": as_float(capacity)})
            existing = base + committed
            grid_available = grid_limit + pv - existing
            slot.update({"base_load_kw": as_float(base), "committed_charge_kw": as_float(committed),
                         "existing_load_kw": as_float(existing), "pv_available_kw": as_float(pv),
                         "device_new_limit_kw": as_float(device_available), "grid_new_limit_kw": as_float(grid_available),
                         "sources": {"load": base_sources, "pv": pv_sources, "commitments": commit_sources, "devices": device_sources}})
            if grid_available < 0:
                fail("EXISTING_GRID_CONFLICT", "既有负荷已超过 PV 与并网限额，不能生成可用新增容量")
            slot.update({"status": "available", "capacity_kw": as_float(min(device_available, grid_available))})
        except PlatformError as error:
            slot["status"] = error.code
            result["issues"].append({"code": error.code, "message": error.message, "slot_start": slot_start.isoformat()})
        slot_start = slot_end
    if not result["issues"]:
        result["status"] = "available"


def verify(context):
    if context.get("schema_version") != SCHEMA or context.get("context_hash") != digest({k: v for k, v in context.items() if k != "context_hash"}):
        fail("CONTEXT_HASH_MISMATCH", "冻结上下文已变化")
    if context.get("input_hash") != digest(context["inputs"]):
        fail("CONTEXT_HASH_MISMATCH", "冻结原始输入已变化")
    return context
