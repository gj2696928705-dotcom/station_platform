"""Reproducible local storage data and validation, with no platform connection.

Only Python 3.10+ standard-library modules are used.  Source paths are relative
to --project-root.  --validate-only and --self-test never write files.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation, getcontext
import hashlib
import io
import json
from pathlib import Path
import platform
import sys

getcontext().prec = 50
D = Decimal
ENERGY_TOL = D("0.000001")
EPS = D("0.000000000001")


class StorageError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def require(condition, code, message):
    if not condition:
        raise StorageError(code, message)


def num(value, label="number", signed=False):
    require(value not in (None, "") and not isinstance(value, bool), "INVALID_NUMBER", f"{label}缺失或不是数值")
    try:
        result = D(str(value))
    except InvalidOperation as exc:
        raise StorageError("INVALID_NUMBER", f"{label}不是数值") from exc
    require(result.is_finite(), "INVALID_NUMBER", f"{label}不是有限数")
    require(signed or result >= 0, "INVALID_NUMBER", f"{label}不能为负数")
    return result


def when(value):
    try:
        dt = datetime.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise StorageError("INVALID_TIME", f"时间格式错误: {value}") from exc
    require(dt.utcoffset() == timedelta(hours=8), "INVALID_TIME", f"时间必须显式使用+08:00: {value}")
    return dt


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def cell(key, value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if isinstance(value, Decimal) or isinstance(value, (int, float)) and key.endswith(("_kw", "_kwh", "_percent")):
        value = num(value, key, signed=True)
        places = 6 if key.endswith("_kw") else 9
        formatted = format(value, f".{places}f")
        return formatted.lstrip("-") if D(formatted) == 0 else formatted
    return str(value)


def csv_bytes(rows):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows({key: cell(key, value) for key, value in row.items()} for row in rows)
    return out.getvalue().encode("utf-8-sig")


def read_csv(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline="")))


def flag(value):
    require(value in (True, False, "true", "false"), "INVALID_FLAG", "布尔字段必须为true/false")
    return value is True or value == "true"


def near(a, b, code="ENERGY_BALANCE", label="数值不一致", tolerance=ENERGY_TOL):
    require(abs(a - b) <= tolerance, code, label)


def validate_parameters(cfg):
    p, validity = cfg["parameters"], cfg["validity"]
    require(cfg["schema_version"] == "huanghua-storage-profile-1" and cfg["is_simulated"] is True, "SCHEMA", "结构版本或模拟属性不符")
    require(cfg["identity"]["timezone"] == "Asia/Shanghai" and validity["interval_minutes"] == 15, "TIME_CONTRACT", "本版本使用上海时区、15分钟")
    start, end = when(validity["effective_from"]), when(validity["effective_to"])
    require(when(validity["available_at"]) <= start < end, "TIME_CONTRACT", "参数有效期错误")
    require(cfg["calculation"]["decimal_precision"] == 50 and num(cfg["calculation"]["energy_tolerance_kwh"]) == ENERGY_TOL
            and num(cfg["calculation"]["numeric_boundary_epsilon"]) == EPS, "NUMERIC_CONTRACT", "数值精度/容差约定不符")
    for key in ("nameplate_capacity_kwh", "usable_capacity_kwh", "soc_reference_capacity_kwh", "pcs_rated_power_kw", "max_charge_kw", "max_discharge_kw"):
        require(num(p[key], key) > 0, "PARAMETER_RANGE", f"{key}必须为正数")
    for key in ("charge_efficiency", "discharge_efficiency"):
        require(0 < num(p[key], key) <= 1, "EFFICIENCY_RANGE", f"{key}必须在(0,1]内")
    require(0 < num(p["soh_percent"]) <= 100 and p["capacity_already_adjusted_for_soh"] is True, "CAPACITY_BASIS", "SOH或容量口径错误")
    near(num(p["usable_capacity_kwh"]), num(p["nameplate_capacity_kwh"]) * num(p["soh_percent"]) / 100,
         "CAPACITY_BASIS", "有效满容量应已反映SOH")
    near(num(p["soc_reference_capacity_kwh"]), num(p["usable_capacity_kwh"]), "CAPACITY_BASIS", "SOC参考容量必须使用当前满容量，不再乘SOH")
    low, high, reserve, terminal = (num(p[key]) for key in ("soc_min_percent", "soc_max_percent", "reserve_soc_percent", "terminal_soc_target_percent"))
    require(0 <= low < high <= 100 and low <= reserve <= high and max(low, reserve) <= terminal <= high,
            "SOC_PARAMETER_RANGE", "SOC下限/上限/备用/末端目标矛盾")
    require(p["max_charge_kw"] <= p["pcs_rated_power_kw"] and p["max_discharge_kw"] <= p["pcs_rated_power_kw"], "PCS_LIMIT", "方向上限超过PCS额定功率")
    require(p["reserve_rule"] == "max(soc_min_percent,reserve_soc_percent); not_additive"
            and p["terminal_rule"] == "hard_equality_at_each_next_local_midnight; no_rolling_window_reset",
            "POLICY", "备用和日终规则不符")
    require(p["simultaneous_charge_discharge_allowed"] is False and p["power_side"] == "AC", "POWER_BASIS", "本版本使用AC侧非负互斥功率")
    require(p["self_discharge_enabled"] is False and num(p["self_discharge_rate_per_hour"]) == 0,
            "MODEL_SCOPE", "本版本显式禁用自放电")
    require(all(p[key] is False for key in ("ramp_model_enabled", "temperature_model_enabled", "ageing_model_enabled", "auxiliary_power_model_enabled", "economic_model_enabled", "site_dispatch_ready"))
            and all(p[key] is None for key in ("ramp_rate_kw_per_minute", "auxiliary_power_kw", "throughput_cost_cny_per_kwh"))
            and p["local_dynamic_model_enabled"] is True, "MODEL_SCOPE", "不得暗中引入未启用模型/费用或宣称全站调度就绪")
    require(cfg["output"]["csv_encoding"] == "utf-8-sig" and cfg["output"]["csv_newline"] == "LF", "CSV_FORMAT", "CSV编码或换行不符")
    names = [cfg["output"][k] for k in ("parameters", "initial_state", "trace", "provenance", "notes", "report")]
    require(len(names) == len(set(names)) and all(Path(n).name == n and n not in ("simulation_config.json", "generate_storage_data.py") for n in names), "OUTPUT_PATH", "输出名冲突或含目录")


def meta(cfg, source_type):
    i, v = cfg["identity"], cfg["validity"]
    return {**i, "dataset_id": cfg["dataset_id"], "schema_version": cfg["schema_version"], "config_version": cfg["config_version"],
            "recorded_on": cfg["recorded_on"], "parameter_valid_from": v["effective_from"], "parameter_valid_to": v["effective_to"],
            "parameter_available_at": v["available_at"], "execution_mode": "offline_simulation", "source_type": source_type, "is_simulated": True}


def parameter_row(cfg):
    p = deepcopy(cfg["parameters"])
    cap, eta_c, eta_d = (num(p[k]) for k in ("soc_reference_capacity_kwh", "charge_efficiency", "discharge_efficiency"))
    floor = max(num(p["soc_min_percent"]), num(p["reserve_soc_percent"]))
    return {**meta(cfg, "synthetic_storage_parameters"), **p,
            "effective_discharge_floor_percent": floor, "physical_min_energy_kwh": cap * num(p["soc_min_percent"]) / 100,
            "physical_max_energy_kwh": cap * num(p["soc_max_percent"]) / 100,
            "reserve_floor_energy_kwh": cap * floor / 100,
            "operating_window_energy_kwh": cap * (num(p["soc_max_percent"]) - num(p["soc_min_percent"])) / 100,
            "ordinary_dispatch_window_energy_kwh": cap * (num(p["soc_max_percent"]) - floor) / 100,
            "round_trip_efficiency": eta_c * eta_d, "interval_minutes": cfg["validity"]["interval_minutes"]}


def make_state(cfg, scenario=None):
    state = deepcopy(cfg["initial_state"])
    if scenario:
        state.update(deepcopy(scenario.get("initial_overrides", {})))
        state["state_id"] = f"{scenario['id']}-STATE-000"
        shift = timedelta(minutes=scenario.get("state_time_shift_minutes", 0))
        for key in ("business_time", "measured_at", "available_at", "valid_from", "valid_to"):
            state[key] = (when(state[key]) + shift).isoformat()
        state["available_at"] = (when(state["available_at"]) + timedelta(minutes=scenario.get("available_delay_minutes", 0))).isoformat()
    state["energy_kwh"] = num(state["soc_percent"]) * num(cfg["parameters"]["soc_reference_capacity_kwh"]) / 100
    if scenario:
        for key in scenario.get("drop_state_fields", []):
            state.pop(key, None)
    return state


def validate_state(cfg, state, as_of):
    required = ("state_id", "business_time", "energy_kwh", "soc_percent", "measured_at", "available_at", "valid_from", "valid_to",
                "online", "fault", "manual_lock", "protection_active", "derating_factor", "committed_charge_energy_ac_kwh",
                "committed_discharge_energy_ac_kwh", "committed_charge_power_kw", "committed_discharge_power_kw")
    require(all(k in state and state[k] not in (None, "") for k in required), "STATE_MISSING", "必要状态字段缺失")
    measured, available, begin, end = (when(state[k]) for k in ("measured_at", "available_at", "valid_from", "valid_to"))
    require(measured <= available and measured <= as_of and begin == measured
            and end - begin == timedelta(minutes=cfg["validity"]["interval_minutes"])
            and when(state["business_time"]) == measured, "STATE_TIME_ORDER", "量测/可见/有效/业务时间顺序错误")
    require(available <= as_of, "STATE_NOT_YET_AVAILABLE", "本条状态在决策时尚不可见")
    require(begin <= as_of < end, "STATE_EXPIRED", "状态过期或超出有效窗口")
    require(when(cfg["validity"]["effective_from"]) <= as_of < when(cfg["validity"]["effective_to"]), "PARAMETERS_EXPIRED", "储能参数不在有效期")
    p = cfg["parameters"]
    soc, energy = num(state["soc_percent"]), num(state["energy_kwh"])
    require(num(p["soc_min_percent"]) <= soc <= num(p["soc_max_percent"]), "STATE_SOC_OUT_OF_RANGE", "当前SOC超出物理运行边界，不裁剪伪造")
    near(energy, soc * num(p["soc_reference_capacity_kwh"]) / 100, "STATE_ENERGY_MISMATCH", "SOC和电量不一致")
    require(0 <= num(state["derating_factor"]) <= 1, "STATE_DERATING_RANGE", "降额系数不在[0,1]")
    for key in ("online", "fault", "manual_lock", "protection_active"):
        require(type(state[key]) is bool, "STATE_FLAG", "状态布尔量格式错误")
    for key in ("committed_charge_energy_ac_kwh", "committed_discharge_energy_ac_kwh", "committed_charge_power_kw", "committed_discharge_power_kw"):
        num(state[key], key)


def capabilities(cfg, state, as_of):
    unknown = {"capability_status": "unavailable", "capability_error_code": "", "local_max_charge_kw": None,
               "local_max_discharge_kw": None, "available_discharge_energy_ac_kwh": None, "charge_headroom_ac_kwh": None,
               "device_charge_limit_kw": None, "device_discharge_limit_kw": None}
    try:
        validate_state(cfg, state, as_of)
    except (StorageError, KeyError, TypeError) as exc:
        unknown["capability_error_code"] = getattr(exc, "code", "STATE_MISSING")
        return unknown
    p = cfg["parameters"]
    cap, energy, factor, eta_c, eta_d = num(p["soc_reference_capacity_kwh"]), num(state["energy_kwh"]), num(state["derating_factor"]), num(p["charge_efficiency"]), num(p["discharge_efficiency"])
    dt = D(cfg["validity"]["interval_minutes"]) / 60
    floor = cap * max(num(p["soc_min_percent"]), num(p["reserve_soc_percent"])) / 100
    ceiling = cap * num(p["soc_max_percent"]) / 100
    charge_energy = max(D(0), (ceiling - energy) / eta_c - num(state["committed_charge_energy_ac_kwh"]))
    discharge_energy = max(D(0), (energy - floor) * eta_d - num(state["committed_discharge_energy_ac_kwh"]))
    device_c = max(D(0), num(p["max_charge_kw"]) * factor - num(state["committed_charge_power_kw"]))
    device_d = max(D(0), num(p["max_discharge_kw"]) * factor - num(state["committed_discharge_power_kw"]))
    blocked = not state["online"] or state["fault"] or state["manual_lock"] or state["protection_active"]
    if blocked:
        charge_energy = discharge_energy = device_c = device_d = D(0)
    status = "blocked" if blocked else "recovery_only" if energy < floor else "derated" if factor < 1 else "available"
    return {"capability_status": status, "capability_error_code": "DEVICE_BLOCKED" if blocked else "",
            "local_max_charge_kw": min(device_c, charge_energy / dt), "local_max_discharge_kw": min(device_d, discharge_energy / dt),
            "available_discharge_energy_ac_kwh": discharge_energy, "charge_headroom_ac_kwh": charge_energy,
            "device_charge_limit_kw": device_c, "device_discharge_limit_kw": device_d}


def initial_row(cfg):
    state = make_state(cfg)
    ability = capabilities(cfg, state, when(state["business_time"]))
    require(ability["capability_status"] == "available", "INITIAL_STATE", "基准初始状态须可用")
    require(num(state["soc_percent"]) == num(cfg["parameters"]["terminal_soc_target_percent"]), "INITIAL_TERMINAL", "本版日终目标须等于基准初始SOC")
    p = cfg["parameters"]
    duration = ability["available_discharge_energy_ac_kwh"] / num(p["max_discharge_kw"]) * 60
    return {**meta(cfg, "synthetic_initial_state"), **state, **ability,
            "full_power_discharge_duration_minutes": duration,
            "next_terminal_at": (when(state["business_time"]).replace(hour=0, minute=0, second=0) + timedelta(days=1)).isoformat(),
            "local_model_ready": True, "site_dispatch_ready": False}


def execute_step(cfg, state, as_of, charge_kw, discharge_kw, next_state_id):
    """Validate a request without mutation; invalid requests have no applied fields."""
    ability = capabilities(cfg, state, as_of)
    result = {**ability, "result_status": "unavailable" if ability["capability_status"] == "unavailable" else "rejected",
              "error_code": ability["capability_error_code"], "execution_applied": False, "output_state_id": None,
              "output_measured_at": None, "output_available_at": None, "output_valid_to": None,
              "applied_charge_kw": None, "applied_discharge_kw": None, "charge_energy_ac_kwh": None,
              "discharge_energy_ac_kwh": None, "charge_loss_kwh": None, "discharge_loss_kwh": None,
              "conversion_loss_kwh": None, "energy_after_kwh": None, "soc_after_percent": None,
              "energy_balance_residual_kwh": None}
    if result["result_status"] == "unavailable":
        return result, None
    p = cfg["parameters"]
    try:
        pc, pd = num(charge_kw, "charge_kw"), num(discharge_kw, "discharge_kw")
        require(not (pc > 0 and pd > 0), "SIMULTANEOUS_CHARGE_DISCHARGE", "禁止同时充放电，不自动净额抵消")
        require(ability["capability_status"] != "blocked" or (pc == pd == 0), "DEVICE_BLOCKED", "故障/离线/锁定/保护设备不可执行充放电")
        require(pc <= num(p["max_charge_kw"]) + EPS and pd <= num(p["max_discharge_kw"]) + EPS, "PCS_LIMIT", "超过PCS方向上限")
        require(pc <= ability["device_charge_limit_kw"] + EPS and pd <= ability["device_discharge_limit_kw"] + EPS,
                "DERATING_LIMIT", "超过降额/既有承诺后的局部设备功率")
        require(pc <= ability["local_max_charge_kw"] + EPS and pd <= ability["local_max_discharge_kw"] + EPS,
                "ENERGY_LIMIT", "超过当前SOC、备用或既有承诺后的区间能力")
    except StorageError as exc:
        result["error_code"] = exc.code
        return result, None
    dt = D(cfg["validity"]["interval_minutes"]) / 60
    ec, ed, eta_c, eta_d = pc * dt, pd * dt, num(p["charge_efficiency"]), num(p["discharge_efficiency"])
    loss_c, loss_d = ec * (1 - eta_c), ed / eta_d - ed
    energy = num(state["energy_kwh"]) + eta_c * ec - ed / eta_d
    next_at = as_of + timedelta(minutes=cfg["validity"]["interval_minutes"])
    after = deepcopy(state)
    after.update(state_id=next_state_id, business_time=next_at.isoformat(), measured_at=next_at.isoformat(), available_at=next_at.isoformat(),
                 valid_from=next_at.isoformat(), valid_to=(next_at + timedelta(minutes=cfg["validity"]["interval_minutes"])).isoformat(),
                 energy_kwh=energy, soc_percent=energy / num(p["soc_reference_capacity_kwh"]) * 100,
                 dispatch_status="charging" if pc > 0 else "discharging" if pd > 0 else "idle")
    result.update(result_status="accepted", error_code="", execution_applied=True, output_state_id=after["state_id"],
                  output_measured_at=after["measured_at"], output_available_at=after["available_at"], output_valid_to=after["valid_to"],
                  applied_charge_kw=pc, applied_discharge_kw=pd, charge_energy_ac_kwh=ec, discharge_energy_ac_kwh=ed,
                  charge_loss_kwh=loss_c, discharge_loss_kwh=loss_d, conversion_loss_kwh=loss_c + loss_d,
                  energy_after_kwh=energy, soc_after_percent=after["soc_percent"],
                  energy_balance_residual_kwh=energy - num(state["energy_kwh"]) - ec + ed + loss_c + loss_d)
    return result, after


def commands_for(scenario):
    if "commands" in scenario:
        return deepcopy(scenario["commands"])
    require(type(scenario["steps"]) is int and scenario["steps"] > 0, "SCENARIO", "场景步数错误")
    commands = [{"charge_kw": 0, "discharge_kw": 0} for _ in range(scenario["steps"])]
    occupied = set()
    for segment in scenario.get("segments", []):
        for i in range(segment["start_step"], segment["start_step"] + segment["count"]):
            require(0 <= i < len(commands) and i not in occupied, "SCENARIO", "指令段越界或重叠")
            commands[i] = {"charge_kw": segment.get("charge_kw", 0), "discharge_kw": segment.get("discharge_kw", 0)}
            occupied.add(i)
    return commands


def trajectories(cfg):
    rows, summaries = [], []
    ids = [s["id"] for s in cfg["scenarios"]]
    require(len(ids) == len(set(ids)), "SCENARIO_ID", "场景编号重复")
    start, step = when(cfg["initial_state"]["business_time"]), timedelta(minutes=cfg["validity"]["interval_minutes"])
    state_fields = list(cfg["initial_state"]) + ["energy_kwh"]
    for scenario in cfg["scenarios"]:
        state, error, accepted, scenario_rows = make_state(cfg, scenario), "", 0, []
        for index, command in enumerate(commands_for(scenario)):
            as_of = start + index * step
            charge, discharge = command.get("charge_kw", 0), command.get("discharge_kw", 0)
            result, after = execute_step(cfg, state, as_of, charge, discharge, f"{scenario['id']}-STATE-{index+1:03d}")
            row = {**meta(cfg, "synthetic_validation_trace"), "scenario_id": scenario["id"], "scenario_name": scenario["name"], "step_index": index,
                   "interval_start": as_of.isoformat(), "interval_end": (as_of + step).isoformat(), "decision_at": as_of.isoformat(),
                   **{f"input_{k}": state.get(k) for k in state_fields}, "requested_charge_kw": charge, "requested_discharge_kw": discharge,
                   **result, "terminal_check_status": "not_checked", "expected_scenario_error": scenario.get("expected_error", ""),
                   "quality_flags": ["synthetic_test_case"], "site_feasibility_checked": False}
            scenario_rows.append(row)
            if after is None:
                error = result["error_code"]
                break
            accepted += 1
            state = after
        require(error == scenario.get("expected_error", ""), "SCENARIO_EXPECTATION", f"{scenario['id']}期望错误{scenario.get('expected_error','')}，实际{error}")
        terminal, gap = "not_checked", D(0)
        if scenario.get("terminal_check"):
            require(not error and when(state["measured_at"]) == start.replace(hour=0, minute=0, second=0) + timedelta(days=1), "TERMINAL_TIME", "日终检查须到达次日00:00")
            target = num(cfg["parameters"]["soc_reference_capacity_kwh"]) * num(cfg["parameters"]["terminal_soc_target_percent"]) / 100
            gap = target - num(state["energy_kwh"])
            terminal = "passed" if abs(gap) <= ENERGY_TOL else "failed"
            require(terminal == scenario.get("expected_terminal", "passed"), "TERMINAL_EXPECTATION", "日终检查结果与预期不一致")
            scenario_rows[-1]["terminal_check_status"] = terminal
        if "expected_final_soc_percent" in scenario:
            near(num(state["soc_percent"]), num(scenario["expected_final_soc_percent"]), "SCENARIO_SOC", f"{scenario['id']}末端SOC不符", D("0.000000001"))
        summaries.append({"scenario_id": scenario["id"], "name": scenario["name"], "test_status": "passed",
                          "accepted_steps": accepted, "rows": len(scenario_rows), "observed_error": error, "expected_error": scenario.get("expected_error", ""),
                          "terminal_check_status": terminal, "terminal_gap_energy_kwh": float(gap),
                          "last_valid_state_soc_percent": float(num(state["soc_percent"])) if "soc_percent" in state else None,
                          "is_deliberate_negative_case": bool(error) or terminal == "failed"})
        rows.extend(scenario_rows)
    return rows, summaries


def audit_sources(cfg, root):
    records = {}

    def check(key, spec):
        path = (root / spec["path"]).resolve()
        require(path.is_relative_to(root) and path.is_file(), "SOURCE_MISSING", f"来源不存在或越出项目目录: {spec['path']}")
        actual = sha(path.read_bytes())
        require(actual == spec["sha256"], "SOURCE_HASH_MISMATCH", f"来源已变化: {spec['path']}；应核对并另建版本")
        records[key] = {"path": spec["path"], "sha256": actual, "matches_pinned_hash": True}

    for key, spec in cfg["sources"].items():
        check(key, spec)
    registry = json.loads((root / cfg["sources"]["registry_config"]["path"]).read_text(encoding="utf-8-sig"))
    for key, spec in registry["sources"].items():
        check("upstream:" + key, spec)
    identity, p = cfg["identity"], cfg["parameters"]
    require(registry["dataset_id"] == identity["base_registry_dataset_id"] and registry["config_version"] == identity["base_registry_config_version"]
            and registry["station"]["station_id"] == identity["station_id"], "REGISTRY_IDENTITY", "旧台账版本或站点身份不一致")
    assets = {r["asset_id"]: r for r in read_csv((root / cfg["sources"]["registry_assets"]["path"]).read_bytes())}
    battery, pcs = assets.get(identity["storage_id"]), assets.get(identity["pcs_id"])
    require(battery is not None and pcs is not None and battery["parent_asset_id"] == pcs["asset_id"], "REGISTRY_ASSETS", "储能/PCS身份或父子关系不符")
    near(num(battery["rated_capacity_kwh"]), num(p["nameplate_capacity_kwh"]), "REGISTRY_CAPACITY", "旧台账名义容量不符")
    near(num(pcs["rated_power_kw"]), num(p["pcs_rated_power_kw"]), "REGISTRY_CAPACITY", "旧台账PCS功率不符")
    meters = {r["meter_id"]: r for r in read_csv((root / cfg["sources"]["registry_meters"]["path"]).read_bytes())}
    require(identity["meter_id"] in meters and json.loads(meters[identity["meter_id"]]["covers_asset_ids"]) == [identity["pcs_id"]], "REGISTRY_METER", "储能计量点未对应PCS")
    constraints = {r["constraint_id"]: r for r in read_csv((root / cfg["sources"]["registry_constraints"]["path"]).read_bytes())}
    require(num(p["max_charge_kw"]) <= num(constraints["C-PCS-CHARGE"]["limit_value"])
            and num(p["max_discharge_kw"]) <= num(constraints["C-PCS-DISCHARGE"]["limit_value"]), "REGISTRY_LIMIT", "新参数不能扩大旧台账PCS边界")
    old_report = json.loads((root / cfg["sources"]["registry_report"]["path"]).read_text(encoding="utf-8"))
    require(old_report["status"] == "passed", "REGISTRY_REPORT", "上游台账报告未通过")
    return records


def provenance_rows(cfg, config_hash):
    rows = []

    def add(pid, scope, value, unit, basis, source, location, comment):
        spec = cfg["sources"].get(source)
        rows.append({**meta(cfg, "synthetic_parameter_provenance"), "provenance_id": pid, "field_scope": scope,
                     "value_or_rule": value, "unit": unit, "basis_type": basis, "source_ref": source,
                     "source_path": spec["path"] if spec else "simulation_config.json", "source_location": location,
                     "source_sha256": spec["sha256"] if spec else config_hash, "rationale": comment})

    p = cfg["parameters"]
    add("P-IDENTITY", "station/storage/PCS/meter", "HH-Huanghua-CSS; BESS-001; PCS-001; M-STORAGE", "", "既有模拟设定", "registry_assets", "asset_id/parent_asset_id/meter_id", "按设备身份关联旧台账，不覆盖v1；旧台账空SOC/效率由本包明确补齐。")
    add("P-CAPACITY", "nameplate_capacity_kwh; pcs_rated_power_kw", f"{p['nameplate_capacity_kwh']}kWh; {p['pcs_rated_power_kw']}kW", "kWh; kW", "既有模拟设定", "registry_config", "$.equipment.storage", "沿用用户已确定的2MW/4MWh模拟铭牌，不代表实站核实参数。")
    add("P-SOH", "usable_capacity_kwh; soc_reference_capacity_kwh; soh_percent", "4000; 4000; 100", "kWh; kWh; %", "新增模拟假设", "this_config", "$.parameters", "usable_capacity是当前满容量，不是SOC窗口电量；已折算SOH，不再次乘SOH。")
    add("P-EFFICIENCY", "charge_efficiency; discharge_efficiency", "0.95; 0.95", "0—1", "本次已选模拟参数", "this_config", "$.parameters.charge_efficiency/discharge_efficiency; $.approval_basis", "用户分别选择充放电95%；往返90.25%为乘积，不从单个往返效率任意拆分。")
    add("P-SOC", "initial_soc; soc_min/max; reserve_soc", "60; 10/90; 30", "%", "本次已选模拟参数", "this_config", "$.initial_state.soc_percent; $.parameters; $.approval_basis", "日常放电底线=max(10,30)=30%，不是40%；物理边界内低于备用时可充电恢复，不虚改SOC。")
    add("P-TERMINAL", "terminal_soc_target_percent; terminal_rule", "每天次日00:00恢复60%，硬等式", "%", "本次已选模拟参数", "this_config", "$.parameters.terminal_*; $.approval_basis", "滚动窗口连续衔接；只检查到达日终的轨迹，短轨迹不宣称已满足日终目标。")
    add("P-INITIAL", "initial availability/state/commitments", "在线、idle、无故障/锁定/保护/降额；既有储能承诺0", "", "新增模拟假设", "this_config", "$.initial_state", "基准初态只覆盖首个15分钟，后续状态由执行结果接续，不每轮重置60%。")
    add("P-TIME", "effective/measurement/availability/validity", "参数2025全年；初态2025-01-01 00:00，状态左闭右开15分钟", "", "新增模拟假设", "this_config", "$.validity; $.initial_state", "业务回放与实际建档日期分开；过期、缺失或未来才可见时能力为空，不写0。")
    add("P-SIMPLIFICATION", "self_discharge/ramp/temperature/ageing/auxiliary/economics", "自放电禁用且计算率0；其他模型未启用；吞吐费用留空", "", "新增模拟假设", "this_config", "$.parameters.*_enabled; throughput_cost_cny_per_kwh", "不是实站自放电或辅助用电为0的结论，未知费用不能计成0。")
    add("P-DYNAMICS", "energy_update; AC/DC boundary", "E_next=E+eta_c*P_c*dt-P_d*dt/eta_d; SOC=E/reference*100", "kW; kWh; h", "设计资料参考", "handoff", "第4.4节储能参数与实时状态；第5.1节能量平衡", "新动态模型实现效率转换；后续站级平衡仍在AC侧，不能再重复加入同一转换损耗。")
    add("P-LIMITS", "PCS directional bounds; local capability", "设备边界与SOC/备用/承诺共同限制，禁止同时充放电", "kW", "既有模拟设定", "registry_constraints", "C-PCS-CHARGE/C-PCS-DISCHARGE", "2MW只是PCS上限；局部能力未叠加10MW并网余量，不是全站可执行计划。")
    add("P-REFERENCE-SOC", "historical mock SOC proportions", "旧演示SOC60%、保留30%", "%", "既有模拟设定", "platform_reference", "MockStationAdapter.__init__.energy.storage", "仅说明比例沿革；本次用户已选这些比例，不沿用旧演示500kWh容量或修改平台。")
    add("P-DESIGN", "reserve and boundary categories", "最低应急SOC、容量/设备状态边界", "", "设计资料参考", "design", "word/document.xml段落188—189、264、427（1起算，含表格段落）", "资料没有给出本设备实测效率、SOH或完整SOC参数，不能把新数值写成文档已核实值。")
    add("P-MECHANISM", "state continuity and feedback", "SOC、可充放功率、PCS状态、最低应急电量和实际反馈", "", "设计资料参考", "mechanism", "word/document.xml段落106、238、254—255（1起算）", "只实现本包离线储能状态演算，不表示站端已接入或具备真实控制。")
    add("P-SLIDES", "power/energy reserve principle", "资源能力扣除备用与刚性占用；功率能力受可释放电量限制", "", "设计资料参考", "slides", "第3—5页；第8页孤岛阈值只作边界说明", "孤岛保供的60/40/25%分级规则不自动套入本轮日常调度。")
    add("P-SCENARIOS", "validation schedules and negative samples", "22个独立场景，各场景内部连续；故意失败项明确标记", "", "新增模拟假设", "this_config", "$.scenarios", "正常日轨迹是计算校验用指令，不是由全年真值、电价或优化器产生的运行策略。")
    return rows


def state_from_row(cfg, row, prefix=""):
    template = make_state(cfg)
    result = {}
    for key, example in template.items():
        value = row[prefix + key]
        if value in (None, ""):
            result[key] = None
        elif isinstance(example, bool):
            result[key] = flag(value)
        elif isinstance(example, (int, float, Decimal)):
            result[key] = num(value, key)
        elif isinstance(example, list):
            result[key] = json.loads(value) if isinstance(value, str) else value
        else:
            result[key] = value
    return result


def validate_tables(cfg, tables):
    """Recompute persisted numbers and state chains, rather than trusting PASS."""
    require(len(tables["parameters"]) == len(tables["initial_state"]) == 1, "TABLE_ROWS", "参数与基准初态各须1行")
    for records in tables.values():
        require(records, "TABLE_ROWS", "表不能为空")
        fields = set(records[0])
        for row in records:
            require(set(row) == fields, "COLUMNS", "表内列结构不一致")
            require(flag(row["is_simulated"]) and row["station_id"] == cfg["identity"]["station_id"]
                    and all(row[k] == cfg[k] for k in ("dataset_id", "schema_version", "config_version")), "ROW_IDENTITY", "行身份或模拟属性不符")
            require(all(row[k] == cfg["identity"][k] for k in ("storage_id", "pcs_id", "meter_id", "base_registry_dataset_id", "base_registry_config_version")), "ROW_IDENTITY", "设备或上游版本不符")
    parsed_cfg = deepcopy(cfg)
    prow = tables["parameters"][0]
    for key, example in cfg["parameters"].items():
        value = prow[key]
        if example is None:
            require(value in (None, ""), "MODEL_SCOPE", f"{key}未知值不应填数")
            parsed_cfg["parameters"][key] = None
        elif isinstance(example, bool):
            parsed_cfg["parameters"][key] = flag(value)
        elif isinstance(example, (int, float)):
            parsed_cfg["parameters"][key] = num(value, key)
        else:
            parsed_cfg["parameters"][key] = value
    validate_parameters(parsed_cfg)
    for key, expected in parameter_row(cfg).items():
        if isinstance(expected, (int, float, Decimal)) and not isinstance(expected, bool):
            near(num(prow[key]), num(expected), "PARAMETER_MISMATCH", f"参数或派生值{key}不符")
    initial = state_from_row(cfg, tables["initial_state"][0])
    validate_state(cfg, initial, when(cfg["initial_state"]["business_time"]))
    expected_initial = initial_row(cfg)
    for key, expected in expected_initial.items():
        actual = tables["initial_state"][0][key]
        if isinstance(expected, (int, float, Decimal)) and not isinstance(expected, bool):
            near(num(actual), num(expected), "INITIAL_STATE_MISMATCH", f"初态或能力{key}不符")
    grouped = {}
    accepted, rejected, unavailable, max_residual = 0, 0, 0, D(0)
    p = cfg["parameters"]
    dt = D(cfg["validity"]["interval_minutes"]) / 60
    energy_cap, eta_c, eta_d = num(p["soc_reference_capacity_kwh"]), num(p["charge_efficiency"]), num(p["discharge_efficiency"])
    scenario_defs = {s["id"]: s for s in cfg["scenarios"]}
    for row in tables["trace"]:
        sid = row["scenario_id"]
        require(sid in scenario_defs, "SCENARIO_ID", "轨迹存在未定义场景")
        previous = grouped.setdefault(sid, [])
        require(int(row["step_index"]) == len(previous), "TRACE_SEQUENCE", "场景步号重复、缺失或乱序")
        as_of, end = when(row["interval_start"]), when(row["interval_end"])
        require(when(row["decision_at"]) == as_of and end - as_of == timedelta(minutes=15), "TRACE_TIME", "区间或决策时间错误")
        require(as_of == when(cfg["initial_state"]["business_time"]) + timedelta(minutes=15 * len(previous)), "TRACE_TIME", "场景时钟不连续")
        state = state_from_row(cfg, row, "input_")
        if previous:
            last = previous[-1]
            require(last["result_status"] == "accepted" and row["input_state_id"] == last["output_state_id"], "TRACE_CHAIN", "下一步没有接续上一有效状态")
            near(num(row["input_energy_kwh"]), num(last["energy_after_kwh"]), "TRACE_RESET", "状态电量被重置或丢失")
        ability = capabilities(cfg, state, as_of)
        require(row["capability_status"] == ability["capability_status"], "CAPABILITY_STATUS", "能力状态不符")
        for key in ("local_max_charge_kw", "local_max_discharge_kw", "available_discharge_energy_ac_kwh", "charge_headroom_ac_kwh"):
            if ability[key] is None:
                require(row[key] in (None, ""), "UNKNOWN_AS_ZERO", "不可用能力不能填成0")
            else:
                near(num(row[key]), ability[key], "CAPABILITY_MISMATCH", f"局部能力{key}不符")
        status = row["result_status"]
        if status == "accepted":
            accepted += 1
            validate_state(cfg, state, as_of)
            require(flag(row["execution_applied"]) and row["error_code"] == "", "EXECUTION_STATUS", "成功状态应有执行且无错误")
            pc, pd = num(row["applied_charge_kw"]), num(row["applied_discharge_kw"])
            require(not (pc > 0 and pd > 0), "SIMULTANEOUS_CHARGE_DISCHARGE", "落盘轨迹同时充放电")
            require(pc <= ability["local_max_charge_kw"] + ENERGY_TOL and pd <= ability["local_max_discharge_kw"] + ENERGY_TOL, "TRACE_POWER_LIMIT", "落盘功率突破局部能力")
            near(pc, num(row["requested_charge_kw"]), "REQUEST_APPLIED_MISMATCH", "请求与执行充电功率不符")
            near(pd, num(row["requested_discharge_kw"]), "REQUEST_APPLIED_MISMATCH", "请求与执行放电功率不符")
            before, after = num(row["input_energy_kwh"]), num(row["energy_after_kwh"])
            near(after, before + pc * dt * eta_c - pd * dt / eta_d, "ENERGY_BALANCE", "逐区间电池能量更新不守恒")
            near(num(row["soc_after_percent"]) * energy_cap / 100, after, "TRACE_SOC_MISMATCH", "落盘SOC与能量不一致")
            require(num(p["soc_min_percent"]) - EPS <= num(row["soc_after_percent"]) <= num(p["soc_max_percent"]) + EPS, "TRACE_SOC_LIMIT", "执行后SOC越界")
            floor = energy_cap * max(num(p["soc_min_percent"]), num(p["reserve_soc_percent"])) / 100
            require(pd == 0 or after >= floor - ENERGY_TOL, "RESERVE_LIMIT", "执行放电突破备用底线")
            ec, ed = num(row["charge_energy_ac_kwh"]), num(row["discharge_energy_ac_kwh"])
            lc, ld = num(row["charge_loss_kwh"]), num(row["discharge_loss_kwh"])
            near(ec, pc * dt); near(ed, pd * dt)
            near(lc, ec * (1 - eta_c)); near(ld, ed / eta_d - ed)
            near(num(row["conversion_loss_kwh"]), lc + ld)
            residual = abs(after - before - ec + ed + lc + ld)
            require(residual <= ENERGY_TOL, "ENERGY_BALANCE", "转换损耗被漏算或重复计算")
            near(num(row["energy_balance_residual_kwh"], signed=True), after - before - ec + ed + lc + ld)
            max_residual = max(max_residual, residual)
            require(when(row["output_measured_at"]) == when(row["output_available_at"]) == end
                    and when(row["output_valid_to"]) == end + timedelta(minutes=15), "TRACE_OUTPUT_TIME", "输出状态可用时间错误")
        else:
            require(status in ("rejected", "unavailable") and not flag(row["execution_applied"]) and row["error_code"], "EXECUTION_STATUS", "错误请求伪造成成功")
            if status == "unavailable":
                unavailable += 1
                require(ability["capability_status"] == "unavailable" and row["error_code"] == ability["capability_error_code"], "UNAVAILABLE_REASON", "不可用原因不符")
            else:
                rejected += 1
                response, unused = execute_step(cfg, state, as_of, row["requested_charge_kw"], row["requested_discharge_kw"], "UNUSED")
                require(unused is None and response["error_code"] == row["error_code"], "REJECTION_REASON", "拒绝原因不符")
            require(all(row[key] in (None, "") for key in ("output_state_id", "applied_charge_kw", "applied_discharge_kw", "energy_after_kwh", "soc_after_percent", "conversion_loss_kwh")),
                    "FALSE_EXECUTION", "未执行请求不能有伪造实际功率或下一状态")
        previous.append(row)
    require(set(grouped) == set(scenario_defs), "SCENARIO_MISSING", "校验场景缺失")
    for sid, records in grouped.items():
        spec, last = scenario_defs[sid], records[-1]
        require(last["error_code"] == spec.get("expected_error", ""), "SCENARIO_EXPECTATION", "实际错误与场景预期不符")
        if spec.get("terminal_check"):
            require(when(last["interval_end"]).hour == when(last["interval_end"]).minute == 0
                    and len(records) == 96, "TERMINAL_TIME", "日终轨迹时长错误")
            target = energy_cap * num(p["terminal_soc_target_percent"]) / 100
            terminal = "passed" if abs(num(last["energy_after_kwh"]) - target) <= ENERGY_TOL else "failed"
            require(terminal == last["terminal_check_status"] == spec["expected_terminal"], "TERMINAL_MISMATCH", "日终结果或预期不符")
    provenance = tables["provenance"]
    require(len({r["provenance_id"] for r in provenance}) == len(provenance), "PROVENANCE_ID", "来源编号重复")
    require(all(r["basis_type"] in ("设计资料参考", "既有模拟设定", "本次已选模拟参数", "新增模拟假设") for r in provenance), "PROVENANCE_TYPE", "来源类别错误")
    return {"parameter_rows": 1, "initial_state_rows": 1, "trace_rows": len(tables["trace"]), "scenario_count": len(grouped),
            "accepted_steps": accepted, "rejected_requests": rejected, "unavailable_requests": unavailable,
            "provenance_rows": len(provenance), "max_energy_balance_residual_kwh": float(max_residual),
            "state_chain_and_time_checked": True, "invalid_requests_have_no_execution": True,
            "unavailable_capabilities_are_null": True, "storage_only_not_site_dispatch": True}


def self_tests(cfg, tables):
    tests = {}
    start = when(cfg["initial_state"]["business_time"])

    def bad_parameter(name, key, value, code):
        changed = deepcopy(cfg)
        changed["parameters"][key] = value
        try:
            validate_parameters(changed)
        except StorageError as exc:
            require(exc.code == code, "SELF_TEST", f"{name}错误码为{exc.code}，应为{code}")
            tests[name] = {"status": "passed", "rejected_with": code}
        else:
            raise StorageError("SELF_TEST", f"{name}未拒绝")

    for name, value, code in [("zero_efficiency", 0, "EFFICIENCY_RANGE"), ("efficiency_above_one", 1.01, "EFFICIENCY_RANGE"),
                              ("negative_efficiency", -0.1, "INVALID_NUMBER"), ("missing_efficiency", None, "INVALID_NUMBER"),
                              ("nan_efficiency", "NaN", "INVALID_NUMBER"), ("infinite_efficiency", "Infinity", "INVALID_NUMBER")]:
        bad_parameter(name, "charge_efficiency", value, code)
    bad_parameter("inverted_SOC_window", "soc_min_percent", 95, "SOC_PARAMETER_RANGE")
    bad_parameter("capacity_reduced_twice", "usable_capacity_kwh", 3800, "CAPACITY_BASIS")
    for name, value in [("negative_power", -1), ("missing_power", None), ("nan_power", "NaN"), ("infinite_power", "Infinity")]:
        response, after = execute_step(cfg, make_state(cfg), start, value, 0, "UNUSED")
        require(response["error_code"] == "INVALID_NUMBER" and after is None and response["applied_charge_kw"] is None, "SELF_TEST", f"{name}未正确拒绝")
        tests[name] = {"status": "passed", "rejected_with": "INVALID_NUMBER"}
    state = make_state(cfg)
    first, middle = execute_step(cfg, state, start, 2000, 0, "CONTINUOUS-1")
    second, after = execute_step(cfg, middle, start + timedelta(minutes=15), 2000, 0, "CONTINUOUS-2")
    require(second["result_status"] == "accepted", "SELF_TEST", "跨窗口未接续")
    near(num(after["energy_kwh"]), D(3350), "SELF_TEST", "跨窗口重置能量")
    snapshot = json.loads(json.dumps(middle, default=str))
    resumed, restored = execute_step(cfg, snapshot, start + timedelta(minutes=15), 2000, 0, "CONTINUOUS-2")
    near(num(restored["energy_kwh"]), num(after["energy_kwh"]), "SELF_TEST", "状态序列化续算丢失能量")
    tests["continuous_and_serialized_state_no_reset"] = {"status": "passed", "end_energy_kwh": 3350}
    require(capabilities(cfg, state, start + timedelta(minutes=15))["capability_error_code"] == "STATE_EXPIRED"
            and capabilities(cfg, middle, start + timedelta(minutes=15))["capability_status"] == "available", "SELF_TEST", "状态有效期左闭右开错误")
    tests["initial_state_expires_and_successor_becomes_available"] = {"status": "passed"}
    soh_cfg = deepcopy(cfg)
    soh_cfg["parameters"].update(soh_percent=95, usable_capacity_kwh=3800, soc_reference_capacity_kwh=3800)
    validate_parameters(soh_cfg)
    soh_state = make_state(soh_cfg)
    near(num(soh_state["energy_kwh"]), D(2280), "SELF_TEST", "SOH被重复折减")
    tests["SOH_applied_once_in_reference_capacity"] = {"status": "passed", "energy_at_60_percent_with_3800kWh_reference": 2280}
    committed = make_state(cfg)
    committed.update(committed_discharge_energy_ac_kwh=100, committed_discharge_power_kw=100)
    capability = capabilities(cfg, committed, start)
    near(capability["available_discharge_energy_ac_kwh"], D(1040), "SELF_TEST", "承诺电量未扣除")
    near(capability["local_max_discharge_kw"], D(1900), "SELF_TEST", "承诺功率未扣除")
    tests["existing_commitments_reduce_local_capability"] = {"status": "passed"}
    # A matched pair of 15-minute steps returns to an independent energy total.
    _, charged = execute_step(cfg, make_state(cfg), start, 1000, 0, "ROUND-1")
    _, restored = execute_step(cfg, charged, start + timedelta(minutes=15), 0, D("902.5"), "ROUND-2")
    near(num(restored["energy_kwh"]), D(2400), "SELF_TEST", "充放电损耗使能量凭空增加或减少")
    tests["conversion_losses_and_return_to_original_energy"] = {"status": "passed"}
    mutated = deepcopy(tables)
    accepted_row = next(r for r in mutated["trace"] if r["result_status"] == "accepted")
    accepted_row["soc_after_percent"] = "99"
    try:
        validate_tables(cfg, mutated)
    except StorageError as exc:
        require(exc.code == "TRACE_SOC_MISMATCH", "SELF_TEST", f"篡改轨迹应报SOC不一致，实际{exc.code}")
        tests["persisted_SOC_tamper_detected"] = {"status": "passed", "rejected_with": exc.code}
    else:
        raise StorageError("SELF_TEST", "落盘SOC篡改未发现")
    mutated = deepcopy(tables)
    unavailable_row = next(r for r in mutated["trace"] if r["result_status"] == "unavailable")
    unavailable_row["local_max_charge_kw"] = "0"
    try:
        validate_tables(cfg, mutated)
    except StorageError as exc:
        require(exc.code == "UNKNOWN_AS_ZERO", "SELF_TEST", f"未知能力填0错误码不符: {exc.code}")
        tests["unknown_capability_cannot_be_zero"] = {"status": "passed", "rejected_with": exc.code}
    else:
        raise StorageError("SELF_TEST", "未知能力填0未发现")
    return tests


FIELD_NOTES = {
    "station_id": "沿用年度模拟站点HH-Huanghua-CSS。", "storage_id": "旧台账储能身份BESS-001。", "pcs_id": "旧台账PCS身份PCS-001。",
    "meter_id": "M-STORAGE，PCS的AC侧模拟计量点。", "timezone": "Asia/Shanghai。",
    "base_registry_dataset_id": "被扩展的旧台账数据身份。", "base_registry_config_version": "被扩展的旧台账参数版本，不覆盖该版本。",
    "dataset_id": "本储能数据包身份。", "schema_version": "字段结构版本。", "config_version": "本储能参数版本。",
    "recorded_on": "实际建档日期，与模拟业务日期分开；不是每次运行的系统时间。", "parameter_valid_from": "参数业务有效期起点，包含。",
    "parameter_valid_to": "参数有效期终点，不包含。", "parameter_available_at": "静态参数在模拟业务时钟中最早可用时间。",
    "execution_mode": "offline_simulation，离线模拟。", "source_type": "模拟参数/初态/校验轨迹/来源，区别于量测质量。", "is_simulated": "固定true。",
    "nameplate_capacity_kwh": "电池名义容量4,000kWh。", "usable_capacity_kwh": "已反映SOH的当前满容量4,000kWh；不是SOC窗口内能量。",
    "soc_reference_capacity_kwh": "SOC计算分母4,000kWh，已反映SOH，不再重复折减。", "soh_percent": "健康状态100%，基准模拟假设。",
    "capacity_already_adjusted_for_soh": "true；当前满容量及SOC参考容量已经反映SOH。", "pcs_rated_power_kw": "PCS的AC侧额定功率。",
    "max_charge_kw": "AC侧局部设备充电上限，尚须受状态和能量限制。", "max_discharge_kw": "AC侧局部设备放电上限。", "power_side": "AC；电池能量为储能侧kWh。",
    "charge_efficiency": "0—1，AC充电电量转为电池增量的比例。", "discharge_efficiency": "0—1，电池释放能量转为AC交付的比例。",
    "soc_min_percent": "物理运行下限10%。", "soc_max_percent": "物理运行上限90%。", "reserve_soc_percent": "日常保留30%，不与物理下限相加。",
    "reserve_rule": "日常底线=max(运行下限,保留SOC)。", "terminal_soc_target_percent": "日终硬等式目标60%。", "terminal_rule": "每天次日00:00恢复目标，日内滚动不重置。",
    "simultaneous_charge_discharge_allowed": "false，不允许以净额抵消同时充放电。", "self_discharge_enabled": "false，本版本不模拟自放电。",
    "self_discharge_rate_per_hour": "显式禁用模型的计算率0，不是实测自放电为0。", "ramp_model_enabled": "false，未建爬坡模型。",
    "ramp_rate_kw_per_minute": "未知留空，不能把0理解为允许的爬坡率。", "temperature_model_enabled": "false，未建温度动态；场景直接设降额系数。",
    "ageing_model_enabled": "false，短轨迹不更新SOH。", "auxiliary_power_model_enabled": "false，未建辅助用电。", "auxiliary_power_kw": "未提供留空。",
    "throughput_cost_cny_per_kwh": "未知费用留空，交后续经济评价确定。", "economic_model_enabled": "false，不评价成本收益。",
    "site_dispatch_ready": "false，未评估并网余量和全站联合计划，也未接平台。", "local_dynamic_model_enabled": "true，只启用局部储能动态。",
    "effective_discharge_floor_percent": "max(10,30)=30%。", "physical_min_energy_kwh": "物理下限对应400kWh。", "physical_max_energy_kwh": "物理上限对应3,600kWh。",
    "reserve_floor_energy_kwh": "日常放电底线对应1,200kWh。", "operating_window_energy_kwh": "10%—90%跨度3,200kWh。",
    "ordinary_dispatch_window_energy_kwh": "30%—90%跨度2,400kWh，不等于初态可放电量。", "round_trip_efficiency": "两个独立效率相乘，0.9025。",
    "interval_minutes": "15分钟，计算为0.25小时。", "state_id": "状态唯一编号。", "business_time": "状态对应的模拟业务时刻。",
    "measured_at": "模拟状态发生时刻，不冒充真实BMS量测。", "available_at": "该条状态最早可用于决策的业务时刻。",
    "valid_from": "状态有效窗口起点，包含。", "valid_to": "状态有效窗口终点，不包含；下一状态应接续。",
    "soc_percent": "0—100的百分数，与energy_kwh/参考容量一致。", "energy_kwh": "当前电池能量。", "online": "通信/设备在线状态。",
    "dispatch_status": "模拟idle/charging/discharging，不能据此宣称现场已执行。", "fault": "故障标志。", "manual_lock": "人工锁定标志。",
    "protection_active": "保护动作标志。", "derating_factor": "0—1，直接作用于设备功率；本包不从温度推断。",
    "committed_charge_energy_ac_kwh": "其他既有承诺占用的AC充电空间，基准0。", "committed_discharge_energy_ac_kwh": "其他既有承诺占用的AC放电电量，基准0。",
    "committed_charge_power_kw": "其他既有承诺占用的充电功率，基准0。", "committed_discharge_power_kw": "其他既有承诺占用的放电功率，基准0。",
    "quality_flags": "JSON数组，轨迹标记synthetic_test_case；不是实测质量等级。", "capability_status": "available/derated/recovery_only/blocked/unavailable。",
    "capability_error_code": "能力阻断或不可用原因；正常为空。", "local_max_charge_kw": "15分钟局部充电上限=min(设备可用功率,剩余AC充电空间/0.25h)。",
    "local_max_discharge_kw": "15分钟局部放电上限，扣除备用和既有承诺；未知为空。", "available_discharge_energy_ac_kwh": "当前可交付AC电量，基准1,140kWh；阻断为0、未知为空。",
    "charge_headroom_ac_kwh": "达到90%前可接收AC电量，基准约1,263.157895kWh。", "device_charge_limit_kw": "降额和功率承诺后的局部设备上限。",
    "device_discharge_limit_kw": "降额和功率承诺后的局部设备放电上限。", "full_power_discharge_duration_minutes": "初态以2MW到保留底线的连续放电时长34.2分钟。",
    "next_terminal_at": "下一次本地日终目标时刻。", "local_model_ready": "仅参数和初态支持局部模型，不等于全站调度就绪。",
    "scenario_id": "独立校验场景编号；不同场景不能拼成一条运营轨迹。", "scenario_name": "校验场景名称。", "step_index": "场景内从0起的连续步号。",
    "interval_start": "执行区间起点，包含。", "interval_end": "执行区间终点，不包含。", "decision_at": "本次模拟决策时刻。",
    "requested_charge_kw": "请求的AC充电平均功率。", "requested_discharge_kw": "请求的AC放电平均功率。",
    "result_status": "accepted/rejected/unavailable；故意错误样本不是可行计划。", "error_code": "拒绝或不可用原因码。",
    "execution_applied": "仅accepted为true。", "output_state_id": "有效执行后新状态编号；未执行为空。",
    "output_measured_at": "新状态发生时刻，为区间终点。", "output_available_at": "新状态最早可用时刻，为区间终点。", "output_valid_to": "新状态有效期排他终点。",
    "applied_charge_kw": "实际应用于本模拟器的AC功率；未执行为空，不写成0。", "applied_discharge_kw": "实际应用于本模拟器的AC放电功率；未执行为空。",
    "charge_energy_ac_kwh": "区间AC充电电量=P_charge×0.25。", "discharge_energy_ac_kwh": "区间AC交付电量=P_discharge×0.25。",
    "charge_loss_kwh": "AC输入减电池增量，充电转换损耗。", "discharge_loss_kwh": "电池释放减AC交付，放电转换损耗。",
    "conversion_loss_kwh": "充放电转换损耗之和，不在站级AC平衡重复添加。", "energy_after_kwh": "执行后电池能量，未执行为空。",
    "soc_after_percent": "执行后百分数，未执行为空。", "energy_balance_residual_kwh": "E_after−E_before−AC_charge+AC_discharge+loss，允许±0.000001kWh。",
    "terminal_check_status": "仅到日终的末行写passed/failed，其余not_checked。", "expected_scenario_error": "本场景预期错误，正常为空。",
    "site_feasibility_checked": "false，本包只检查局部储能。", "provenance_id": "来源条目唯一编号。", "field_scope": "该依据覆盖的参数或规则。",
    "value_or_rule": "参数值或规则。", "unit": "量纲；规则文本可为空。", "basis_type": "设计资料参考/既有模拟设定/本次已选模拟参数/新增模拟假设。",
    "source_ref": "配置sources键名或this_config。", "source_path": "相对项目根目录的来源路径；this_config为本包配置。",
    "source_location": "JSON路径、章节、1起算的OOXML段落或幻灯片页。", "source_sha256": "冻结来源的完整SHA-256。", "rationale": "依据、简化及证据边界。",
}


def notes(cfg, tables, stats, summaries, tests):
    initial = initial_row(cfg)
    lines = ["# 黄花站储能参数与初始状态", "", f"建档日期：{cfg['recorded_on']}；数据：`{cfg['dataset_id']}`；配置：`{cfg['config_version']}`。", "",
             "本包补齐2MW/4MWh储能的局部运行参数与初态，提供可复算的充放电校验轨迹。全部为模拟，不代表真实BMS数据、实站效率或已经接入的平台调度。", "",
             "## 1. 参数与初始能力", "", "| 项目 | 值及口径 |", "| --- | --- |",
             "| 设备身份 | HH-Huanghua-CSS / BESS-001 / PCS-001 / M-STORAGE |",
             "| 名义、当前满容量、SOC参考容量 | 均为4,000kWh，SOH100%，已经反映SOH，不再重复折减 |",
             "| PCS与效率 | AC侧充电/放电上限各2,000kW；充电95%、放电95%，往返90.25% |",
             "| SOC边界与备用 | 物理10%—90%；日常保留30%；有效底线=max(10,30)=30%，不相加 |",
             "| 基准初态 | 2025-01-01 00:00（Asia/Shanghai），SOC60%、2,400kWh；在线idle，无故障/保护/锁定/降额或既有承诺 |",
             "| 日终规则 | 次日00:00硬等式恢复60%；滚动窗口接续有效执行状态，不逐小时重置 |",
             "| 容量窗口 | 物理窗口3,200kWh；日常调度窗口2,400kWh；都不等于初态可交付AC电量 |",
             f"| 初态可交付电量 | 高于备用的电池能量1,200kWh，经放电效率后为{initial['available_discharge_energy_ac_kwh']:.6f}kWh AC |",
             f"| 初态充电空间 | 充至90%需AC输入{initial['charge_headroom_ac_kwh']:.9f}kWh |",
             f"| 初态额定放电持续时间 | 以2MW到保留底线为{initial['full_power_discharge_duration_minutes']:.6f}分钟，不能直接说可放2小时 |", "",
             "参数业务有效期为2025全年；初态有效窗口为[2025-01-01 00:00,00:15)。后续每步输出新状态，其测量/可见时间为区间终点、有效期至再下一个15分钟边界。建档日期与业务回放时间分开，不声称2025年实站已部署这些参数。", "",
             "## 2. 与上一阶段的关系", "",
             "本包以配置引用扩展《场站配置与设备台账》v1.0.0，核对旧包全部8文件摘要及其原始输入摘要。旧包中的空SOC/效率和dispatch_ready=false保持原版本含义；仅明确选择本包的新消费者才能使用补齐的储能参数，旧任务不会自动改变。", "",
             "旧名义容量4MWh、PCS2MW、设备父子关系及M-STORAGE覆盖范围保持一致。原光伏、负荷、充电会话、旧台账、平台和模型均不修改。本轮不生成全年策略，也不执行真实设备控制。", "",
             "局部能力尚未叠加全站负荷、光伏和10MW并网余量，不是可直接下发的站级计划。site_dispatch_ready=false。", "",
             "## 3. 动态与边界", "", "```text", "E_next = E + 0.95 × P_charge × 0.25 − P_discharge × 0.25 / 0.95", "SOC_percent = E / 4000 × 100",
             "AC_charge_energy = P_charge × 0.25", "AC_discharge_energy = P_discharge × 0.25",
             "conversion_loss = AC_charge_energy × (1−0.95) + AC_discharge_energy × (1/0.95−1)",
             "E_next − E − AC_charge_energy + AC_discharge_energy + conversion_loss = 0", "```", "",
             "同一时段不能同时充放电，不自动以净额抵消。功率先受PCS、降额及既有承诺限制，再受SOC/备用后的能量余量限制。既有承诺基准为0，能力计算仍显式扣除其他承诺的功率和AC能量。", "",
             "20%等低于备用但仍在10%—90%内的有效状态标记recovery_only，日常放电能力为0，允许逐步充电恢复；既不拒绝一切恢复充电，也不强行把SOC改成30%。", "",
             "故障、离线、保护、锁定时，已知设备能力为blocked、上下限0。缺失、过期、未来才可见、时间顺序错误或越物理SOC边界的输入为unavailable，能力为空，不伪装成0。", "",
             "非法请求不修改输入状态：execution_applied=false，应用功率、损耗和下一状态均为空，场景停止。失败行是测试请求结果，不是“实际执行0kW”的回执。正常idle执行才可记录已执行0kW。", "",
             "自放电在本版显式禁用、计算率0；爬坡、温度、老化、辅助用电均未启用，相关未知数值留空。吞吐费用未知留空，不按0计算收益。转换损耗已经进入电池动态，未来站级AC平衡不能再重复加入同一损耗。", "",
             "## 4. 轨迹和校验结果", "",
             f"共{stats['scenario_count']}个独立场景、{stats['trace_rows']}行：成功模拟执行{stats['accepted_steps']}步，按预期拒绝{stats['rejected_requests']}条、状态不可用{stats['unavailable_requests']}条。每个场景独立初始化，场景内部连续；不同场景不能拼接成真实运营轨迹。", "",
             "正常日轨迹为00:00—01:00充1,000kW，18:00—19:00放902.5kW，其余闲置，共96步，末端恢复60%。这是一组验证用指令，不是按电价或未来实际光伏/负荷求出的优化策略。", "",
             "另一个96步反例从50%闲置至日终，明确产生failed和400kWh缺口；它的预期失败被正确识别，所以测试本身passed。总报告passed表示正例正确、反例被正确处理，不表示每条请求都可行。", "",
             "| 场景 | 名称 | 行数 | 错误/日终结果 |", "| --- | --- | --- | --- |"]
    lines.extend(f"| {s['scenario_id']} | {s['name']} | {s['rows']} | {s['observed_error'] or s['terminal_check_status']} |" for s in summaries)
    lines += ["", f"落盘CSV逐区间独立复算后最大能量平衡残差为{stats['max_energy_balance_residual_kwh']:.12f}kWh，门槛为0.000001kWh。另有{len(tests)}项参数、连续状态、SOH、承诺及篡改检查。", "",
              "## 5. 文件与复现", "", "| 文件 | 用途 |", "| --- | --- |",
              "| simulation_config.json | 唯一参数入口，来源摘要和场景定义 |", "| 储能参数.csv | 参数及容量窗口、往返效率派生值，1行 |",
              "| 初始状态.csv | 基准初态及当前局部能力，1行 |", "| 状态校验轨迹.csv | 成功与故意错误场景的请求/状态/校验结果 |",
              "| 来源与假设.csv | 四类依据、原文位置和完整SHA-256 |", "| generate_storage_data.py | Python3.10+标准库生成器及只读校验 |",
              "| validation_report.json | 实算计数、逐场景结果、来源与产物摘要 |", "",
              "CSV使用UTF-8 BOM、英文逗号、LF换行；功率6位小数、能量及SOC9位小数，数组为JSON。布尔为true/false；空值不是0。内部Decimal精度50位，状态持续使用未舍入结果，落盘后再次复算，容差1e-6kWh。", "",
              "以下在本文件夹执行，路径不依赖旧项目绝对路径；源文件路径相对项目根目录。", "", "```powershell",
              "python -B .\\generate_storage_data.py --validate-only", "python -B .\\generate_storage_data.py --self-test",
              "python -B .\\generate_storage_data.py --output-dir ..\\储能数据复现检查", "```", "",
              "默认不覆盖已有产物；--overwrite仅允许显式重建本包，始终禁止覆盖冻结来源。--output-dir复制配置与脚本，形成完整8文件包；移动至不同目录层级时使用--project-root指定原项目根目录。", "",
              "--validate-only和--self-test不写文件；只读校验同时比较检查前后文件字节和修改时间。生成先核对来源，生成后复核来源；四份CSV落盘回读通过后才写成功报告。相同输入、脚本、配置与Python环境下全部8文件可逐字节复现。报告不包含自身摘要，避免循环；generation_runtime记录环境，recorded_on为固定建档日期而非动态运行时间。", "",
              "## 6. 来源与假设", "",
              "2MW/4MWh和设备ID来自已确认旧台账；60%初始SOC、30%保留、10%—90%边界、日终60%、分别95%效率由本次用户选择。SOH100%、无既有承诺、初始健康状态和禁用模型为新模拟假设。", "",
              "资料只支持应急保留、状态连续反馈与资源能力边界等原则，没有提供该设备实测效率或健康参数；演示稿中的孤岛60/40/25%阈值属于另一个业务场景，本包不自动套用。", "",
              "## 7. 字段字典", ""]
    for kind, title in [("parameters", "储能参数"), ("initial_state", "初始状态"), ("trace", "状态校验轨迹"), ("provenance", "来源与假设")]:
        lines += [f"### {title}", "", "| 字段 | 含义 |", "| --- | --- |"]
        for key in tables[kind][0]:
            text = FIELD_NOTES.get(key)
            if key.startswith("input_"):
                text = "请求前输入状态：" + FIELD_NOTES[key[6:]] + " 故意异常样本可能缺失或无效，须结合result_status。"
            require(text is not None, "FIELD_DICTIONARY", f"字段缺少说明: {key}")
            lines.append(f"| `{key}` | {text} |")
        lines.append("")
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def prepare(cfg, config_data, root):
    validate_parameters(cfg)
    sources = audit_sources(cfg, root)
    trace, summaries = trajectories(cfg)
    tables = {"parameters": [parameter_row(cfg)], "initial_state": [initial_row(cfg)], "trace": trace,
              "provenance": provenance_rows(cfg, sha(config_data))}
    blobs = {cfg["output"][key]: csv_bytes(rows) for key, rows in tables.items()}
    decoded = {key: read_csv(blobs[cfg["output"][key]]) for key in tables}
    stats = validate_tables(cfg, decoded)
    tests = self_tests(cfg, decoded)
    repeat_trace, repeat_summaries = trajectories(cfg)
    repeat = {"parameters": [parameter_row(cfg)], "initial_state": [initial_row(cfg)], "trace": repeat_trace,
              "provenance": provenance_rows(cfg, sha(config_data))}
    require(summaries == repeat_summaries and all(csv_bytes(repeat[key]) == blobs[cfg["output"][key]] for key in tables),
            "REPRODUCIBILITY", "重复生成结果不一致")
    blobs[cfg["output"]["notes"]] = notes(cfg, tables, stats, summaries, tests)
    initial = initial_row(cfg)
    benchmarks = {key: float(initial[key]) for key in ("energy_kwh", "local_max_charge_kw", "local_max_discharge_kw", "available_discharge_energy_ac_kwh", "charge_headroom_ac_kwh", "full_power_discharge_duration_minutes")}
    report = {"schema_version": cfg["schema_version"], "dataset_id": cfg["dataset_id"], "config_version": cfg["config_version"],
              "is_simulated": True, "status": "passed", "config_recorded_on": cfg["recorded_on"],
              "generation_runtime": {"python": platform.python_version(), "dependencies": "standard_library_only", "decimal_precision": getcontext().prec},
              "checks": stats, "initial_benchmarks": benchmarks, "scenario_results": summaries, "additional_checks": tests,
              "source_files": sources, "reproducibility": {"repeated_four_csvs_byte_identical": True, "written_artifacts_read_back": True,
              "read_only_preserves_bytes_and_mtime": "checked_on_each_validate_only_run", "report_self_hash_excluded": True},
              "scope": {"platform_connected": False, "base_registry_rewritten": False, "annual_dispatch_generated": False,
              "local_storage_dynamics_validated": True, "site_dispatch_ready": False, "economic_benefit_evaluated": False,
              "negative_scenarios_are_expected_failures_not_feasible_plans": True}}
    return blobs, report


def package_bytes(cfg, config_data, script_data, blobs, report):
    expected = {"simulation_config.json": config_data, "generate_storage_data.py": script_data, **blobs}
    report["artifact_hashes_sha256"] = {key: sha(value) for key, value in sorted(expected.items())}
    expected[cfg["output"]["report"]] = json_bytes(report)
    return expected


def validate_disk(cfg, target, expected):
    for name in expected:
        require((target / name).is_file(), "ARTIFACT_MISSING", f"缺少产物: {name}")
    tables = {key: read_csv((target / cfg["output"][key]).read_bytes()) for key in ("parameters", "initial_state", "trace", "provenance")}
    validate_tables(cfg, tables)
    for name, data in expected.items():
        actual = (target / name).read_bytes()
        if name == cfg["output"]["report"]:
            saved, recomputed = json.loads(actual), json.loads(data)
            saved.pop("generation_runtime", None)
            recomputed.pop("generation_runtime", None)
            require(saved == recomputed, "REPORT_MISMATCH", "报告内容与当前实算结果不一致")
        else:
            require(actual == data, "ARTIFACT_MISMATCH", f"{name}与配置、来源或实算结果不一致")


def snapshot(target):
    return {p.name: (sha(p.read_bytes()), p.stat().st_mtime_ns) for p in target.iterdir() if p.is_file()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().with_name("simulation_config.json"))
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    root, target = (args.project_root or config_path.parents[3]).resolve(), (args.output_dir or config_path.parent).resolve()
    config_data, script_data = config_path.read_bytes(), Path(__file__).resolve().read_bytes()
    cfg = json.loads(config_data)
    before = snapshot(target) if args.validate_only else None
    blobs, report = prepare(cfg, config_data, root)
    expected = package_bytes(cfg, config_data, script_data, blobs, report)
    if args.validate_only:
        validate_disk(cfg, target, expected)
        require(snapshot(target) == before, "READ_ONLY_VIOLATION", "只读校验改动文件字节或修改时间")
    elif not args.self_test:
        protected = {(root / source["path"]).resolve() for source in report["source_files"].values()}
        for name in expected:
            path = target / name
            require(path.resolve() not in protected and not path.is_symlink(), "PROTECTED_SOURCE", f"禁止覆盖来源或符号链接: {path}")
            if path.exists() and path.resolve() not in (config_path, Path(__file__).resolve()):
                require(args.overwrite, "OUTPUT_EXISTS", f"产物已存在: {path}；另选目录或显式--overwrite")
        target.mkdir(parents=True, exist_ok=True)
        for name, data in expected.items():
            if name != cfg["output"]["report"]:
                (target / name).write_bytes(data)
        for name, data in expected.items():
            if name != cfg["output"]["report"]:
                require((target / name).read_bytes() == data, "READBACK", f"写后回读不一致: {name}")
        validate_tables(cfg, {key: read_csv((target / cfg["output"][key]).read_bytes()) for key in ("parameters", "initial_state", "trace", "provenance")})
        audit_sources(cfg, root)
        (target / cfg["output"]["report"]).write_bytes(expected[cfg["output"]["report"]])
        validate_disk(cfg, target, expected)
    audit_sources(cfg, root)
    print(json.dumps({"status": "passed", "mode": "validate_only" if args.validate_only else "self_test" if args.self_test else "generate",
                      "output_directory": str(target), "file_count": len(expected), "checks": report["checks"],
                      "additional_checks": len(report["additional_checks"]), "all_frozen_sources_unchanged": True,
                      "read_only_preserved_bytes_and_mtime": True if args.validate_only else None}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    try:
        raise SystemExit(main())
    except (StorageError, OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "failed", "code": getattr(exc, "code", "INPUT_OR_IO_ERROR"), "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
