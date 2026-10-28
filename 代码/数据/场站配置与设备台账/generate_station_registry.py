"""Deterministic, standard-library-only synthetic registry; never controls devices.

Source paths are relative to --project-root.  --validate-only and --self-test
perform no writes.  Actual session end times are used only for offline audits.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import platform
import sys


class RegistryError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def require(condition, code, message):
    if not condition:
        raise RegistryError(code, message)


def number(value, label="number", nonnegative=True):
    try:
        require(not isinstance(value, bool) and value not in (None, ""),
                "INVALID_NUMBER", f"{label} 缺失或不是数值")
        result = Decimal(str(value))
        require(result.is_finite(), "INVALID_NUMBER", f"{label} 不是有限数")
        require(not nonnegative or result >= 0, "INVALID_NUMBER", f"{label} 为负数")
        return result
    except InvalidOperation as exc:
        raise RegistryError("INVALID_NUMBER", f"{label} 不是合法数值: {value}") from exc


def instant(value):
    try:
        dt = datetime.fromisoformat(value)
        require(dt.utcoffset() == timedelta(hours=8), "INVALID_TIME", f"时间须显式使用+08:00: {value}")
        return dt
    except (TypeError, ValueError) as exc:
        if isinstance(exc, RegistryError):
            raise
        raise RegistryError("INVALID_TIME", f"时间格式错误: {value}") from exc


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if isinstance(value, (int, float, Decimal)):
        return format(number(value), ".6f")
    return str(value)


def csv_bytes(fields, records):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows({k: cell(row.get(k)) for k in fields} for row in records)
    return stream.getvalue().encode("utf-8-sig")


def csv_records(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline="")))


def array(row, key):
    value = row[key]
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise RegistryError("INVALID_COLLECTION", f"{key} 不是JSON数组") from exc
    require(isinstance(value, list), "INVALID_COLLECTION", f"{key} 不是数组")
    return value


COMMON = {
    "station_id": "站点身份，必须与年度数据一致。",
    "dataset_id": "本数据包身份。",
    "schema_version": "字段结构版本。",
    "config_version": "模拟参数版本；调整参数应另建版本。",
    "effective_from": "模拟业务有效起点，包含。",
    "effective_to": "模拟业务有效终点，不包含。",
    "available_at": "静态配置在模拟业务时钟中的可用时间；不是实际建档时刻。",
    "source_type": "synthetic_registry，表示模拟台账，不是量测质量等级。",
    "is_simulated": "固定true，不能因校验通过而改为实测。",
    "provenance_refs": "来源与假设.csv的provenance_id，JSON数组。",
}
ASSET = {
    "asset_id": "数据包内唯一设备或逻辑节点编号。",
    "asset_type": "station/grid_connection/ac_bus/branch/pv_inverter/charger/charging_port/pcs/battery_system/base_load/meter。",
    "asset_name": "可读名称，均为模拟身份。",
    "parent_asset_id": "等效拓扑父节点；唯一根站点为空。不是实站施工接线图。",
    "electrical_side": "AC、DC、AC_equivalent或logical；枪口按AC支路等效模型登记。",
    "power_role": "资源流向或measurement_only。",
    "port_id": "仅枪口填写，原会话编号一对一保留。",
    "connector_type": "DC为模拟兼容类型；并不表明已建立DC损耗模型。",
    "rated_power_kw": "名义局部功率；空值为未提供/不适用，不是0。购电合同限额不写成设备铭牌。",
    "rated_capacity_kwh": "名义电池容量，只是kWh，不代表当前可调能量。",
    "max_available_power_kw": "基准状态下的局部功率上限，尚须受所有上级/并网约束；不保证全站可同时满功率。",
    "capacity_aggregation_role": "leaf_dispatch/alias_of_child/aggregate_of_children/energy_only/excluded/contract_limit。父子不得重复累计。",
    "meter_id": "所属模拟计量点。",
    "availability": "设备运行可用性available，不表示枪口空闲或储能已具备调度资格。",
    "derating_reason": "基准场景无降额，空字符串。",
    "fault": "基准场景false；不是历史告警量测。",
    "protection_active": "基准场景false。",
    "manual_takeover": "基准场景false。",
    "frozen": "基准场景false。",
    "occupancy_source": "枪口占用从会话按业务时刻推导，其余为空。",
    "manufacturer": "未提供，留空。",
    "model": "未提供，留空。",
    "serial_number": "真实序列号未提供，留空。",
    "identity_status": "simulated_id_real_identity_not_provided。",
}
CONSTRAINT = {
    "constraint_id": "唯一约束编号。",
    "scope_asset_id": "受约束设备或逻辑节点。",
    "member_asset_ids": "实际参加计算的资源，JSON数组；不得同时相加祖先与子节点。",
    "aggregation_basis": "single_flow/sum_leaves/net_power_balance/signed_balance。",
    "quantity": "受限功率变量或平衡残差。",
    "operator": "<=或=。",
    "limit_value": "约束值；功率约束单位始终为kW。",
    "unit": "kW。",
    "positive_direction": "正方向说明。",
    "expression": "计算含义；并网限制作用于净功率，不是充电设备额定值简单求和。",
}
METER = {
    "meter_id": "唯一模拟电表编号，同时出现在设备台账。",
    "measured_asset_id": "安装位置对应的等效设备或支路。",
    "covers_asset_ids": "覆盖资源，JSON数组；上下级电表不相加计算站级总量。",
    "measurement_boundary": "AC计量边界及排除项。",
    "channels": "非负功率通道，JSON数组。",
    "positive_direction": "各通道流向，JSON对象。",
    "unit": "kW；未来区间电量使用kWh。",
    "value_semantics": "输入原列含义或未来平均功率；不是当前已生成量测。",
    "source_refs": "配置sources键名，JSON数组。",
    "field_mapping": "原始字段到通道的映射，JSON对象。",
    "is_independent_measurement": "固定false；只定义模拟计量位置。派生并网功率不是独立总表实测。",
    "meter_status": "definition_only，本轮不生成新电表时序。",
}
PROVENANCE = {
    "provenance_id": "唯一依据编号，供其他表关联。",
    "field_scope": "适用参数、设备或规则。",
    "value_or_rule": "参数值或处理规则。",
    "unit": "量纲；非数值规则可为空。",
    "basis_type": "设计资料参考/既有模拟设定/本次已选模拟参数/新增模拟假设。",
    "source_ref": "来源键名或this_config。",
    "source_path": "来源文件相对项目根目录路径；this_config为本包配置。",
    "source_location": "JSON路径、文档XML段落（1起算）、幻灯片页或章节。",
    "source_sha256": "当前冻结来源文件完整SHA-256。",
    "rationale": "依据及不能据此推断的事实。",
}
SCHEMAS = {"assets": ASSET, "constraints": CONSTRAINT, "meters": METER, "provenance": PROVENANCE}
BRANCHES = ["BR-PV", "BR-CHARGE", "BR-STORAGE", "BR-BASE"]
METER_IDS = ["M-PCC", "M-PV", "M-CHARGE", "M-STORAGE", "M-BASE"]


def common(cfg, refs):
    station = cfg["station"]
    return {"station_id": station["station_id"], "dataset_id": cfg["dataset_id"],
            "schema_version": cfg["schema_version"], "config_version": cfg["config_version"],
            **{key: station[key] for key in ("effective_from", "effective_to", "available_at")},
            "source_type": cfg["source_type"], "is_simulated": cfg["is_simulated"], "provenance_refs": refs}


def validate_config(cfg):
    require(cfg["schema_version"] == "huanghua-station-registry-1", "SCHEMA", "不支持此结构版本")
    require(cfg["is_simulated"] is True and cfg["source_type"] == "synthetic_registry", "SOURCE_TYPE", "必须明确为模拟")
    station, equip = cfg["station"], cfg["equipment"]
    require(station["timezone"] == "Asia/Shanghai" and station["interval_minutes"] == 15,
            "TIME_CONTRACT", "本包只支持Asia/Shanghai、15分钟")
    begin, end = instant(station["effective_from"]), instant(station["effective_to"])
    require(begin < end and instant(station["available_at"]) <= begin, "TIME_CONTRACT", "有效期或可用时间错误")
    require((end - begin).total_seconds() == cfg["history_contract"]["expected_interval_rows"] * 900,
            "TIME_CONTRACT", "有效期与区间行数不一致")
    require(number(station["grid_import_limit_kw"]) > 0, "GRID_LIMIT", "购电限额必须为正")
    require(station["export_allowed"] is False and number(station["grid_export_limit_kw"]) == 0,
            "EXPORT_POLICY", "本版本明确禁止反送")
    pv, charge, storage = equip["pv"], equip["charging"], equip["storage"]
    for label, value in [("逆变器数量", pv["inverter_count"]), ("终端数量", charge["terminal_count"]),
                         ("端口数量", charge["port_count"])]:
        require(type(value) is int and value > 0, "COUNT", f"{label}须为正整数")
    require(number(pv["capacity_kw"]) == pv["inverter_count"] * number(pv["inverter_rated_power_kw"]),
            "PV_CAPACITY", "光伏总容量不等于逐台逆变器之和")
    require(charge["terminal_count"] == charge["port_count"] and charge["ports_per_terminal"] == 1
            and charge["multiport_shared_cabinet"] is False, "CHARGER_STRUCTURE", "本版本采用独立单枪终端")
    require(number(charge["port_rated_power_kw"]) == number(charge["terminal_rated_power_kw"]) > 0,
            "CHARGER_POWER", "终端与枪口须共用同一正功率上限")
    require(charge["v2g_enabled"] is False, "V2G", "本包未建V2G")
    require(storage["system_count"] == storage["pcs_count"] == 1, "STORAGE_STRUCTURE", "本版本为单储能、单PCS")
    require(number(storage["rated_capacity_kwh"]) > 0 and number(storage["pcs_rated_power_kw"]) > 0,
            "STORAGE_CAPACITY", "储能容量与PCS功率须为正")
    for field in ("pcs_max_charge_kw", "pcs_max_discharge_kw"):
        require(0 < number(storage[field]) <= number(storage["pcs_rated_power_kw"]),
                "PCS_LIMIT", f"{field}超过PCS额定功率")
    deferred = ["charge_efficiency", "discharge_efficiency", "soc_percent", "soc_min_percent",
                "soc_max_percent", "reserve_soc_percent", "terminal_soc_target_percent"]
    require(storage["dispatch_ready"] is False and all(storage[k] is None for k in deferred),
            "STORAGE_SCOPE", "储能动态参数须留待后续，本包不提供调度资格")
    require(cfg["model"]["meter_count"] == 5, "METER_COUNT", "必须有5个定义计量点")
    state = cfg["baseline_state"]
    require(state["availability"] == "available" and state["availability_is_not_idle"] is True
            and not any(state[k] for k in ("fault", "protection_active", "manual_takeover", "frozen")),
            "BASELINE_STATE", "基准设备可用、无锁定；可用不等于空闲")
    history = cfg["history_contract"]
    require(history["pv_interval_rule"] == "adjacent_point_trapezoid" and number(history["pv_last_endpoint_kw"]) == 0
            and number(history["baseline_storage_charge_kw"]) == number(history["baseline_storage_discharge_kw"]) == 0,
            "AUDIT_POLICY", "历史兼容检查采用梯形积分、末端夜间0、储能闲置")
    outputs = list(cfg["output"][key] for key in ("assets", "constraints", "meters", "provenance", "notes", "report"))
    require(len(set(outputs)) == len(outputs) and all(Path(v).name == v for v in outputs),
            "OUTPUT_PATH", "输出文件名必须唯一且不带目录")
    require(cfg["output"]["csv_encoding"] == "utf-8-sig" and cfg["output"]["csv_newline"] == "LF"
            and history["power_decimal_places"] == 6, "CSV_FORMAT", "本版本使用UTF-8 BOM、LF、6位功率小数")


def audit_sources(cfg, project_root):
    result = {}
    for source_id, spec in cfg["sources"].items():
        path = (project_root / spec["path"]).resolve()
        require(path.is_relative_to(project_root), "SOURCE_PATH", f"来源越出项目目录: {source_id}")
        require(path.is_file(), "SOURCE_MISSING", f"来源文件不存在: {spec['path']}")
        actual = digest(path.read_bytes())
        require(actual == spec["sha256"], "SOURCE_HASH_MISMATCH", f"{source_id}已变化；核对来源并另建配置版本，禁止悄悄替换")
        result[source_id] = {"path": spec["path"], "sha256": actual, "matches_pinned_hash": True}
    return result


def provenance_rows(cfg, config_hash):
    rows = []

    def add(pid, scope, value, unit, basis, source, location, rationale):
        spec = cfg["sources"].get(source)
        rows.append({**common(cfg, []), "provenance_id": pid, "field_scope": scope, "value_or_rule": value,
                     "unit": unit, "basis_type": basis, "source_ref": source,
                     "source_path": spec["path"] if spec else "simulation_config.json",
                     "source_location": location, "source_sha256": spec["sha256"] if spec else config_hash,
                     "rationale": rationale})

    s, eq = cfg["station"], cfg["equipment"]
    add("P-IDENTITY", "station_id/timezone", f"{s['station_id']}; {s['timezone']}", "", "既有模拟设定", "pv_config",
        "$.station.station_id; $.timezone", "沿用年度模拟身份；不是平台mock-station-001的重命名。")
    add("P-TIME", "effective_from/to; available_at; interval_minutes", f"[{s['effective_from']},{s['effective_to']}); {s['interval_minutes']}min", "",
        "新增模拟假设", "this_config", "$.station", "静态规则在业务回放起点可见；实际建档日期单独记录，不能声称2025年已投运。")
    add("P-PV", "pv.capacity_kw; inverter_count/rated_power", f"{eq['pv']['capacity_kw']}kW = {eq['pv']['inverter_count']} × {eq['pv']['inverter_rated_power_kw']}kW", "kW",
        "既有模拟设定", "pv_config", "$.station.capacity_kw; inverter_count_reference; inverter_rated_kw_reference", "原数据是站级合成出力，没有单台逆变器量测。")
    add("P-PORTS", "charging.port_count", eq["charging"]["port_count"], "个", "既有模拟设定", "load_config",
        "$.reference_resources.charging_ports", "现有会话端口SIM-G-001至SIM-G-128完整一对一映射。")
    add("P-CHARGER", "charger/port.structure; rated_power; connector", f"{eq['charging']['terminal_count']}台独立单枪终端，每台{eq['charging']['terminal_rated_power_kw']}kW；DC兼容类型", "kW",
        "新增模拟假设", "this_config", "$.equipment.charging", "240kW参考既有模拟常规枪；一台一枪和AC支路等效口径为新假设，不是已知实站接线/铭牌。")
    add("P-GRID", "grid_import_limit_kw", s["grid_import_limit_kw"], "kW", "本次已选模拟参数", "this_config",
        "$.station.grid_import_limit_kw; $.approval_basis", "用户已选10MW。参考15.12MW为变流尺度，未当作实站购电安全容量。")
    add("P-EXPORT", "export_allowed; grid_export_limit_kw", "false; 0", "kW", "本次已选模拟参数", "this_config",
        "$.station.export_allowed; $.station.grid_export_limit_kw", "按用户要求实施禁止反送；光伏富余需储能吸收或限发。")
    add("P-STORAGE", "storage.rated_capacity; PCS.rated_power", f"{eq['storage']['rated_capacity_kwh']}kWh; {eq['storage']['pcs_rated_power_kw']}kW", "kWh; kW",
        "本次已选模拟参数", "this_config", "$.equipment.storage; $.approval_basis", "用户已选2MW/4MWh，一套储能一台PCS；不是2小时可保证运行承诺。SOC、效率和备用未定。")
    add("P-TOPOLOGY", "logical_topology; capacity_aggregation", "并网点—AC母线—光伏/充电/储能/基础负荷支路；父子功率不重复累计", "",
        "新增模拟假设", "this_config", "$.model.topology; $.model.pcc_balance", "有功等效拓扑，不是施工接线图；没有虚构变压器台数或阻抗。")
    add("P-METER", "meter_count; measurement_boundary", "并网点及四条支路，共5个定义计量点", "个", "新增模拟假设", "this_config",
        "$.model.meter_count; $.model.meter_status", "定义计量范围和方向，不生成独立电表量测。")
    add("P-STATE", "availability/fault/protection/manual/frozen", "available; false; false; false; false", "",
        "新增模拟假设", "this_config", "$.baseline_state", "静态基准全年可用；不是全年真实无故障，也不是端口空闲。")
    add("P-UNKNOWN", "manufacturer/model/serial; unmodelled", "未提供留空；未建模项明确列出", "", "新增模拟假设", "this_config",
        "$.model.unknown_identity; $.model.unmodelled", "空值不是0；设备编号为模拟ID，不伪造真实厂商、型号和序列号。")
    add("P-OCCUPANCY", "port_occupancy", "[start_at,end_at)；到开始时刻占用，到结束时刻释放", "", "既有模拟设定", "sessions",
        "port_id/start_at/end_at", "仅按业务时刻重建历史占用；实际结束时间不能当作规划时已知的预计离站。")
    add("P-PV-TIME", "pv.interval_conversion", "相邻瞬时点梯形平均；末端追加夜间0；右端点到达后才可用", "kW; kWh", "既有模拟设定", "pv_notes",
        "第3节字段字典；第5节积分口径", "全年右端点只用于离线检查；不将未来点混入决策输入。")
    add("P-LOAD-TIME", "base/charging/total_load", "平均功率×0.25h；total=base+charging；total不是并网净负荷", "kW; kWh", "既有模拟设定", "load_notes",
        "第1节字段契约与计量边界", "基础负荷排除充电、光伏、储能；历史充电不能与新计划重复计入。")
    add("P-DESIGN", "equipment_categories; capacity_boundaries", "并网点、充电、PCS、光伏、负荷；分别登记容量边界", "", "设计资料参考", "design",
        "word/document.xml段落13、51、66、69、127、130、155—156（包含表格段落，1起算）", "资料描述功能和约束类别，没有提供黄花站完整逐台铭牌，不能据此认定10MW或4MWh为实站参数。")
    add("P-MECHANISM", "state_and_feedback", "设备、储能、光伏、并网点和站用负荷状态需统一时间口径", "", "设计资料参考", "mechanism",
        "word/document.xml段落47、98、106、112—113（1起算）", "仅参考设备类别和状态关联，不在本轮扩展电压治理、消防或应急控制功能。")
    add("P-SLIDES", "resource_pool; meter_and_state_boundary", "资源能力应扣除刚性占用与备用；储能功率不等于可释放电量", "", "设计资料参考", "slides",
        "第3—5页：可调资源评估、预测阶段资源池、资源调用与能力公式", "备用和SOC规则尚未选定，不能仅凭2MW PCS铭牌承诺2MW可调能力。")
    add("P-HANDOFF", "registry_contract", "站点/设备/共享功率/计量/有效期/模拟属性统一", "", "设计资料参考", "handoff",
        "第3节；第4.1、4.2、4.4节；第5.1节", "交接中的运行示例作为资料，不执行训练、部署或平台状态修改。")
    add("P-PLATFORM", "platform_compatibility", "年度场景与平台默认小规模节点不同，尚未接入", "", "既有模拟设定", "platform_reference",
        "MockStationAdapter.__init__/capture", "保留平台默认配置；未来需专门适配，不将mock-station-001直接重命名为黄花站。")
    return rows


def build_tables(cfg, config_hash):
    s, eq, state = cfg["station"], cfg["equipment"], cfg["baseline_state"]
    assets, constraints, meters = [], [], []

    def asset(aid, kind, name, parent, refs, side="AC", role="excluded", power=None, capacity=None,
              available=None, aggregation="excluded", meter=None, port=None):
        assets.append({**common(cfg, refs), "asset_id": aid, "asset_type": kind, "asset_name": name,
                       "parent_asset_id": parent, "electrical_side": side, "power_role": role,
                       "port_id": port, "connector_type": eq["charging"]["connector_type"] if port else None,
                       "rated_power_kw": power, "rated_capacity_kwh": capacity,
                       "max_available_power_kw": available if available is not None else power,
                       "capacity_aggregation_role": aggregation, "meter_id": meter,
                       **{k: state[k] for k in ("availability", "derating_reason", "fault", "protection_active", "manual_takeover", "frozen")},
                       "occupancy_source": state["port_occupancy_source"] if port else None,
                       "manufacturer": None, "model": None, "serial_number": None,
                       "identity_status": "simulated_id_real_identity_not_provided"})

    pv_total = number(eq["pv"]["capacity_kw"])
    charge_power = number(eq["charging"]["port_rated_power_kw"])
    charge_total = eq["charging"]["port_count"] * charge_power
    pcs_power = number(eq["storage"]["pcs_rated_power_kw"])
    asset("STATION", "station", s["station_name"], None, ["P-IDENTITY", "P-TIME", "P-DESIGN"], side="logical")
    asset("PCC", "grid_connection", "模拟并网点", "STATION", ["P-GRID", "P-EXPORT"],
          role="grid_import_export", available=s["grid_import_limit_kw"], aggregation="contract_limit", meter="M-PCC")
    asset("BUS-AC", "ac_bus", "模拟公共AC母线", "PCC", ["P-TOPOLOGY"], role="balance_only")
    for aid, name, role, power, meter in [
        ("BR-PV", "光伏支路", "generation", pv_total, "M-PV"),
        ("BR-CHARGE", "充电支路", "consumption", charge_total, "M-CHARGE"),
        ("BR-STORAGE", "储能AC支路", "bidirectional", pcs_power, "M-STORAGE"),
        ("BR-BASE", "基础负荷支路", "consumption", None, "M-BASE")]:
        asset(aid, "branch", name, "BUS-AC", ["P-TOPOLOGY", "P-METER"], role=role, power=power,
              aggregation="aggregate_of_children", meter=meter)
    inverter_ids = []
    for i in range(1, eq["pv"]["inverter_count"] + 1):
        aid = f"PV-INV-{i:03d}"
        inverter_ids.append(aid)
        asset(aid, "pv_inverter", f"模拟光伏逆变器{i:03d}", "BR-PV", ["P-PV"], role="generation",
              power=eq["pv"]["inverter_rated_power_kw"], aggregation="leaf_dispatch", meter="M-PV")
    port_ids = []
    for i in range(1, eq["charging"]["port_count"] + 1):
        terminal, port = f"CHG-{i:03d}", f"{eq['charging']['port_id_prefix']}{i:03d}"
        port_ids.append(port)
        asset(terminal, "charger", f"模拟独立单枪终端{i:03d}", "BR-CHARGE", ["P-CHARGER", "P-PORTS"],
              role="consumption", power=charge_power, aggregation="alias_of_child", meter="M-CHARGE")
        asset(port, "charging_port", f"模拟端口{i:03d}", terminal, ["P-CHARGER", "P-PORTS", "P-OCCUPANCY"],
              side="AC_equivalent", role="consumption", power=charge_power, aggregation="leaf_dispatch", meter="M-CHARGE", port=port)
    asset("PCS-001", "pcs", "模拟储能PCS", "BR-STORAGE", ["P-STORAGE"], role="bidirectional",
          power=pcs_power, aggregation="leaf_dispatch", meter="M-STORAGE")
    asset("BESS-001", "battery_system", "模拟储能电池系统", "PCS-001", ["P-STORAGE", "P-UNKNOWN"],
          side="DC", role="energy_state", capacity=eq["storage"]["rated_capacity_kwh"], aggregation="energy_only")
    asset("LOAD-BASE", "base_load", "模拟基础负荷聚合", "BR-BASE", ["P-LOAD-TIME", "P-UNKNOWN"],
          role="consumption", aggregation="leaf_dispatch", meter="M-BASE")

    meter_specs = [
        ("M-PCC", "PCC", BRANCHES, "公共AC并网边界；全站净购售电，包含四支路，不能再与支路总表相加", ["grid_import_kw", "grid_export_kw"],
         {"grid_import_kw": "电网→站内母线", "grid_export_kw": "站内母线→电网"}, [], {}, "future_interval_average; derived_not_independent_meter_reading"),
        ("M-PV", "BR-PV", inverter_ids, "逆变器AC输出；原表为不限发的站级参考出力；未来计量通道为实际利用出力", ["pv_used_kw"],
         {"pv_used_kw": "光伏支路→站内母线"}, ["pv_data"], {"pv_power_kw": "unconstrained_reference_only; requires_trapezoidal_interval_conversion_and_curtailment_policy"}, "source_instantaneous; future_channel_interval_average"),
        ("M-CHARGE", "BR-CHARGE", port_ids, "充电AC支路；排除基础负荷与储能；终端/枪口不重复计量", ["charging_load_kw"],
         {"charging_load_kw": "站内母线→充电支路"}, ["load_data", "sessions"], {"charging_load_kw": "historical_interval_average", "port_id": "port_identity_mapping"}, "interval_average; completed_interval_available_at_end"),
        ("M-STORAGE", "BR-STORAGE", ["PCS-001"], "PCS的AC侧，充放电分别记录非负值；不把DC电池名义电量当作AC表读数", ["storage_charge_kw", "storage_discharge_kw"],
         {"storage_charge_kw": "站内母线→PCS", "storage_discharge_kw": "PCS→站内母线"}, [], {}, "future_interval_average; not_generated"),
        ("M-BASE", "BR-BASE", ["LOAD-BASE"], "站用基础负荷；排除全部车辆充电、光伏和储能", ["base_load_kw"],
         {"base_load_kw": "站内母线→基础负荷"}, ["load_data"], {"base_load_kw": "historical_interval_average"}, "interval_average; completed_interval_available_at_end")]
    for mid, measured, covers, boundary, channels, directions, refs, mapping, semantics in meter_specs:
        asset(mid, "meter", f"模拟计量点{mid}", measured, ["P-METER"], role="measurement_only", meter=mid)
        meters.append({**common(cfg, ["P-METER", "P-TOPOLOGY"]), "meter_id": mid, "measured_asset_id": measured,
                       "covers_asset_ids": list(covers), "measurement_boundary": boundary, "channels": list(channels),
                       "positive_direction": dict(directions), "unit": "kW", "value_semantics": semantics,
                       "source_refs": list(refs), "field_mapping": dict(mapping), "is_independent_measurement": False, "meter_status": "definition_only"})

    def limit(cid, scope, members, basis, quantity, value, direction, expression, refs, operator="<="):
        constraints.append({**common(cfg, refs), "constraint_id": cid, "scope_asset_id": scope,
                            "member_asset_ids": list(members), "aggregation_basis": basis, "quantity": quantity,
                            "operator": operator, "limit_value": value, "unit": "kW", "positive_direction": direction, "expression": expression})

    limit("C-PCC-IMPORT", "PCC", BRANCHES, "net_power_balance", "grid_import_kw", s["grid_import_limit_kw"],
          "电网→站内", "max(base_load_kw + charging_load_kw + storage_charge_kw - pv_used_kw - storage_discharge_kw, 0) <= grid_import_limit_kw", ["P-GRID"])
    limit("C-PCC-EXPORT", "PCC", BRANCHES, "net_power_balance", "grid_export_kw", s["grid_export_limit_kw"],
          "站内→电网", "grid_export_kw <= 0; surplus PV must be absorbed or curtailed", ["P-EXPORT"])
    limit("C-PCC-BALANCE", "PCC", BRANCHES, "signed_balance", "power_balance_residual_kw", 0,
          "供给减消耗", cfg["model"]["pcc_balance"], ["P-TOPOLOGY"], operator="=")
    limit("C-PV-TOTAL", "BR-PV", inverter_ids, "sum_leaves", "pv_used_kw", pv_total,
          "光伏→母线", "sum(inverter_ac_used_kw) <= pv.capacity_kw; do not add BR-PV again", ["P-PV"])
    for aid in inverter_ids:
        limit(f"C-{aid}", aid, [aid], "single_flow", "pv_inverter_ac_kw", eq["pv"]["inverter_rated_power_kw"],
              "逆变器→母线", "inverter_ac_used_kw <= inverter_rated_power_kw", ["P-PV"])
    limit("C-CHARGE-TOTAL", "BR-CHARGE", port_ids, "sum_leaves", "charging_load_kw", charge_total,
          "母线→充电", "sum(port_ac_equivalent_kw) <= terminal_count * terminal_rated_power_kw; PCC net limit also applies", ["P-CHARGER"])
    for i, port in enumerate(port_ids, 1):
        limit(f"C-CHG-{i:03d}", f"CHG-{i:03d}", [port], "single_flow", "port_ac_equivalent_kw", charge_power,
              "母线→终端/枪口共用流", "port_ac_equivalent_kw == terminal_ac_equivalent_kw; count only once", ["P-CHARGER"])
    for direction in ("charge", "discharge"):
        limit(f"C-PCS-{direction.upper()}", "PCS-001", ["PCS-001"], "single_flow", f"storage_{direction}_kw",
              eq["storage"][f"pcs_max_{direction}_kw"], "母线→PCS" if direction == "charge" else "PCS→母线",
              f"storage_{direction}_kw <= pcs_max_{direction}_kw; nameplate_only_not_SOC_feasible_dispatch", ["P-STORAGE"])
    return {"assets": assets, "constraints": constraints, "meters": meters, "provenance": provenance_rows(cfg, config_hash)}


def validate_tables(cfg, tables):
    assets, limits, meters, provenance = (tables[k] for k in ("assets", "constraints", "meters", "provenance"))
    for kind, records in tables.items():
        expected_fields = set(COMMON) | set(SCHEMAS[kind])
        for row in records:
            require(set(row) == expected_fields, "COLUMNS", f"{kind}列结构不符")
            require(row["station_id"] == cfg["station"]["station_id"]
                    and all(row[k] == cfg[k] for k in ("dataset_id", "schema_version", "config_version")),
                    "ROW_IDENTITY", f"{kind}身份不一致")
            require(cell(row["is_simulated"]) == "true" and row["source_type"] == "synthetic_registry", "SOURCE_TYPE", f"{kind}非模拟行")
            for field in ("effective_from", "effective_to", "available_at"):
                require(row[field] == cfg["station"][field], "ROW_TIME", f"{kind}.{field}不一致")
    ids = [a["asset_id"] for a in assets]
    require(len(ids) == len(set(ids)), "DUPLICATE_ASSET", "设备编号重复")
    lookup = {a["asset_id"]: a for a in assets}
    for a in assets:
        parent = a["parent_asset_id"]
        require(not parent or parent in lookup, "ORPHAN_ASSET", f"孤立设备: {a['asset_id']} → {parent}")
        seen, cursor = set(), a["asset_id"]
        while cursor:
            require(cursor not in seen, "TOPOLOGY_CYCLE", f"设备父子关系成环: {a['asset_id']}")
            seen.add(cursor)
            cursor = lookup[cursor]["parent_asset_id"]
        require("STATION" in seen, "ORPHAN_ASSET", f"{a['asset_id']}未接至站点根节点")
        require(a["availability"] == "available" and all(cell(a[k]) == "false" for k in ("fault", "protection_active", "manual_takeover", "frozen")),
                "ASSET_STATE", f"{a['asset_id']}基准状态不符")
        require(all(not a[k] for k in ("manufacturer", "model", "serial_number")), "REAL_IDENTITY", "本版没有真实厂商型号序列号")
        for field in ("rated_power_kw", "rated_capacity_kwh", "max_available_power_kw"):
            if a[field] not in (None, ""):
                number(a[field], f"{a['asset_id']}.{field}")
        if a["rated_power_kw"] not in (None, "") and a["max_available_power_kw"] not in (None, ""):
            require(number(a["max_available_power_kw"]) <= number(a["rated_power_kw"]), "ASSET_POWER_LIMIT", "局部可用功率超过额定功率")
    counts = Counter(a["asset_type"] for a in assets)
    charge, pv, storage = (cfg["equipment"][k] for k in ("charging", "pv", "storage"))
    ports = {a["port_id"]: a for a in assets if a["asset_type"] == "charging_port"}
    expected_ports = {f"{charge['port_id_prefix']}{i:03d}" for i in range(1, charge["port_count"] + 1)}
    require(set(ports) == expected_ports and counts["charging_port"] == charge["port_count"], "MISSING_PORT", "端口缺失、重复或有额外编号")
    expected_counts = {"station": 1, "grid_connection": 1, "ac_bus": 1, "branch": 4,
                       "charger": charge["terminal_count"], "charging_port": charge["port_count"],
                       "pv_inverter": pv["inverter_count"], "pcs": 1, "battery_system": 1, "base_load": 1, "meter": 5}
    require(dict(counts) == expected_counts, "ASSET_COUNT", f"设备分类数量不符: {dict(counts)}")
    for port_id, port in ports.items():
        parent = lookup[port["parent_asset_id"]]
        require(port["asset_id"] == port_id and parent["asset_type"] == "charger"
                and parent["asset_id"] == f"CHG-{int(port_id[len(charge['port_id_prefix']):]):03d}", "PORT_PARENT", "桩枪映射不符")
        require(parent["capacity_aggregation_role"] == "alias_of_child" and port["capacity_aggregation_role"] == "leaf_dispatch",
                "DOUBLE_COUNT", "终端与枪口须采用别名/叶子计量关系")
        require(number(port["rated_power_kw"]) == number(parent["rated_power_kw"]) == number(charge["port_rated_power_kw"]),
                "CHARGER_POWER", f"{port_id}额定功率不一致")
    inverter_sum = sum(number(a["rated_power_kw"]) for a in assets if a["asset_type"] == "pv_inverter")
    require(inverter_sum == number(lookup["BR-PV"]["rated_power_kw"]) == number(pv["capacity_kw"]), "PV_CAPACITY", "逐台光伏与总容量不一致")
    require(number(lookup["PCS-001"]["rated_power_kw"]) == number(storage["pcs_rated_power_kw"])
            and number(lookup["BESS-001"]["rated_capacity_kwh"]) == number(storage["rated_capacity_kwh"]),
            "STORAGE_CAPACITY", "储能或PCS容量不一致")
    require(not lookup["BESS-001"]["rated_power_kw"] and not lookup["PCS-001"]["rated_capacity_kwh"], "UNIT_BOUNDARY", "不得混淆PCS功率与电池能量")
    mids = [m["meter_id"] for m in meters]
    require(len(mids) == len(set(mids)) == 5 and set(mids) == set(METER_IDS), "METER_COUNT", "计量点缺失或重复")
    expected_covers = {"M-PCC": set(BRANCHES), "M-PV": {a["asset_id"] for a in assets if a["asset_type"] == "pv_inverter"},
                       "M-CHARGE": set(ports), "M-STORAGE": {"PCS-001"}, "M-BASE": {"LOAD-BASE"}}
    for m in meters:
        refs = array(m, "covers_asset_ids")
        require(m["measured_asset_id"] in lookup and all(ref in lookup for ref in refs), "METER_REFERENCE", "计量对象不存在")
        require(len(refs) == len(set(refs)) and set(refs) == expected_covers[m["meter_id"]], "METER_COVERAGE", "计量范围遗漏或重叠")
        require(cell(m["is_independent_measurement"]) == "false" and m["meter_status"] == "definition_only", "METER_SOURCE", "模拟定义不能变为独立实测")
    require(all(not a["meter_id"] or a["meter_id"] in mids for a in assets), "METER_REFERENCE", "设备关联了不存在的电表")
    cids = [c["constraint_id"] for c in limits]
    require(len(cids) == len(set(cids)), "DUPLICATE_CONSTRAINT", "约束编号重复")
    by_constraint = {c["constraint_id"]: c for c in limits}
    expected_cids = {"C-PCC-IMPORT", "C-PCC-EXPORT", "C-PCC-BALANCE", "C-PV-TOTAL", "C-CHARGE-TOTAL", "C-PCS-CHARGE", "C-PCS-DISCHARGE"}
    expected_cids.update(f"C-PV-INV-{i:03d}" for i in range(1, pv["inverter_count"] + 1))
    expected_cids.update(f"C-CHG-{i:03d}" for i in range(1, charge["port_count"] + 1))
    require(set(cids) == expected_cids, "MISSING_CONSTRAINT", "约束缺失或有额外约束")
    for c in limits:
        members = array(c, "member_asset_ids")
        require(c["scope_asset_id"] in lookup and all(m in lookup for m in members), "CONSTRAINT_REFERENCE", "约束引用不存在的资源")
        require(len(members) == len(set(members)), "DOUBLE_COUNT", "约束重复累计同一资源")
        require(c["unit"] == "kW" and c["operator"] in ("<=", "="), "CONSTRAINT_UNIT", "功率约束单位或操作符错误")
        number(c["limit_value"])
        for member in members:
            cursor = lookup[member]["parent_asset_id"]
            while cursor:
                require(cursor not in members, "DOUBLE_COUNT", f"约束{c['constraint_id']}同时累计父子设备")
                cursor = lookup[cursor]["parent_asset_id"]
    for cid, expected in [("C-PCC-IMPORT", cfg["station"]["grid_import_limit_kw"]), ("C-PCC-EXPORT", 0), ("C-PCC-BALANCE", 0),
                          ("C-PV-TOTAL", pv["capacity_kw"]), ("C-CHARGE-TOTAL", charge["port_count"] * number(charge["port_rated_power_kw"]))]:
        require(number(by_constraint[cid]["limit_value"]) == number(expected), "CONSTRAINT_LIMIT", f"{cid}上限不符")
    for i, port_id in enumerate(sorted(ports), 1):
        c = by_constraint[f"C-CHG-{i:03d}"]
        require(array(c, "member_asset_ids") == [port_id] and c["scope_asset_id"] == ports[port_id]["parent_asset_id"]
                and number(c["limit_value"]) == number(ports[port_id]["rated_power_kw"]), "CHARGER_CONSTRAINT", "终端/枪口共用约束不符")
    for direction in ("charge", "discharge"):
        c = by_constraint[f"C-PCS-{direction.upper()}"]
        require(number(c["limit_value"]) == number(storage[f"pcs_max_{direction}_kw"])
                and number(c["limit_value"]) <= number(lookup["PCS-001"]["rated_power_kw"]), "PCS_LIMIT", "PCS方向上限越界")
    pids = [r["provenance_id"] for r in provenance]
    require(len(pids) == len(set(pids)), "PROVENANCE_ID", "来源编号重复")
    require(all(ref in pids for records in tables.values() for r in records for ref in array(r, "provenance_refs")),
            "PROVENANCE_REFERENCE", "来源关联不存在")
    require(all(r["basis_type"] in ("设计资料参考", "既有模拟设定", "本次已选模拟参数", "新增模拟假设") for r in provenance),
            "PROVENANCE_TYPE", "来源分类错误")
    return {"asset_rows": len(assets), "asset_counts": dict(sorted(counts.items())), "constraint_rows": len(limits),
            "meter_rows": len(meters), "provenance_rows": len(provenance), "unique_ids_and_acyclic_topology": True,
            "all_references_resolve": True, "parent_child_power_not_double_counted": True,
            "pv_total_kw": float(inverter_sum), "charging_leaf_total_kw": float(charge["port_count"] * number(charge["port_rated_power_kw"])),
            "storage_pcs_kw": float(number(storage["pcs_rated_power_kw"])), "battery_nameplate_kwh": float(number(storage["rated_capacity_kwh"])),
            "storage_dispatch_ready": False}


def occupied_ports(sessions, as_of):
    """Historical truth at as_of; never interprets actual_end as a service promise."""
    return sorted(row["port_id"] for row in sessions if instant(row["start_at"]) <= as_of < instant(row["end_at"]))


def audit_history(cfg, project_root, tables):
    def read(source):
        return csv_records((project_root / cfg["sources"][source]["path"]).read_bytes())

    pv, load, sessions = read("pv_data"), read("load_data"), read("sessions")
    s, contract = cfg["station"], cfg["history_contract"]
    expected = contract["expected_interval_rows"]
    require(len(pv) == len(load) == expected, "HISTORY_ROWS", "年度主表行数不符")
    require(len(sessions) == contract["expected_session_rows"], "SESSION_ROWS", "会话数量不符")
    begin, end = instant(s["effective_from"]), instant(s["effective_to"])
    tol = number(contract["power_balance_tolerance_kw"])
    grid_peak = surplus_peak = grid_energy = surplus_energy = pv_energy = Decimal(0)
    surplus_count = exceed_count = 0
    for i, (p, l) in enumerate(zip(pv, load)):
        time = begin + timedelta(minutes=15 * i)
        require(instant(p["timestamp"]) == instant(l["timestamp"]) == time, "HISTORY_TIME", f"第{i+1}个时间戳重复、缺口或错位")
        for row in (p, l):
            require(row["station_id"] == s["station_id"] and row["is_simulated"].lower() == "true", "HISTORY_IDENTITY", "原数据身份或模拟属性不符")
            require(all(v not in (None, "") for v in row.values()), "HISTORY_MISSING", "年度原表存在缺失值")
        p0 = number(p["pv_power_kw"], "pv_power_kw")
        require(p0 <= number(cfg["equipment"]["pv"]["capacity_kw"]), "HISTORY_PV_LIMIT", "光伏超过设备总额定功率")
        p1 = number(pv[i + 1]["pv_power_kw"]) if i + 1 < expected else number(contract["pv_last_endpoint_kw"])
        pv_average = (p0 + p1) / 2
        base, charging, total = (number(l[k], k) for k in ("base_load_kw", "charging_load_kw", "total_load_kw"))
        require(base + charging == total, "HISTORY_LOAD_BALANCE", "基础与充电分量不守恒")
        require(number(p["temperature_c"], nonnegative=False) == number(l["temperature_c"], nonnegative=False), "HISTORY_TEMPERATURE", "共用温度不一致")
        net = total - pv_average
        grid, surplus = max(net, 0), max(-net, 0)
        used = pv_average - surplus
        require(abs(grid + used - total) <= tol, "HISTORY_BALANCE", "派生并网能量平衡不符")
        grid_peak, surplus_peak = max(grid_peak, grid), max(surplus_peak, surplus)
        grid_energy += grid * Decimal("0.25")
        surplus_energy += surplus * Decimal("0.25")
        pv_energy += pv_average * Decimal("0.25")
        surplus_count += int(surplus > 0)
        exceed_count += int(grid > number(s["grid_import_limit_kw"]) + tol)
    require(exceed_count == 0, "HISTORY_IMPORT_LIMIT", f"储能闲置的历史基线有{exceed_count}个15分钟区间超过购电上限")
    ports = {a["port_id"]: a for a in tables["assets"] if a["asset_type"] == "charging_port"}
    require({r["port_id"] for r in sessions} == set(ports), "SESSION_PORT_MAPPING", "历史端口与台账未完整一对一对应")
    require(len({r["session_id"] for r in sessions}) == len(sessions), "SESSION_DUPLICATE", "会话编号重复")
    groups = defaultdict(list)
    peaks = set()
    for row in sessions:
        require(row["station_id"] == s["station_id"] and row["is_simulated"].lower() == "true", "SESSION_IDENTITY", "会话身份错误")
        arrival, start, finish = (instant(row[k]) for k in ("arrival_at", "start_at", "end_at"))
        require(arrival <= start < finish and start < end and finish > begin, "SESSION_TIME", "会话时间顺序或年度交集错误")
        peak = number(row["peak_power_kw"])
        require(0 < peak <= number(ports[row["port_id"]]["max_available_power_kw"]), "SESSION_POWER", "历史会话峰值超过对应端口可用上限")
        peaks.add(float(peak))
        groups[row["port_id"]].append((start, finish))
    for port, spans in groups.items():
        spans.sort()
        require(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), "SESSION_OVERLAP", f"同端口会话重叠: {port}")
    source_pv_cfg = json.loads((project_root / cfg["sources"]["pv_config"]["path"]).read_text(encoding="utf-8-sig"))
    source_load_cfg = json.loads((project_root / cfg["sources"]["load_config"]["path"]).read_text(encoding="utf-8-sig"))
    require(number(source_pv_cfg["station"]["capacity_kw"]) == number(cfg["equipment"]["pv"]["capacity_kw"])
            and source_pv_cfg["station"]["inverter_count_reference"] == cfg["equipment"]["pv"]["inverter_count"]
            and source_load_cfg["reference_resources"]["charging_ports"] == len(ports), "SOURCE_CONFIG_SCALE", "原配置与新台账规模不一致")
    initial_ports = occupied_ports(sessions, begin)
    carry_out_ports = occupied_ports(sessions, end)
    return {"scope": "offline_full_year_audit_only_not_decision_input", "time_resolution": "15min_interval_average",
            "not_a_certification_of_intra_interval_peaks_or_real_equipment": True,
            "pv_rows": len(pv), "load_rows": len(load), "session_rows": len(sessions), "historical_ports": len(ports),
            "continuous_unique_aligned_timestamps": True, "missing_main_table_cells": 0,
            "load_components_exactly_conserved": True, "temperature_matches": True,
            "session_peak_values_kw": sorted(peaks), "all_session_peaks_within_port_limit": True, "per_port_sessions_do_not_overlap": True,
            "initial_occupied_port_count": len(initial_ports), "initial_occupied_ports": initial_ports,
            "initial_occupancy_is_historical_truth_not_forecast": True,
            "sessions_continuing_at_year_end": len(carry_out_ports),
            "pv_interval_rule": contract["pv_interval_rule"], "last_midnight_endpoint_kw": contract["pv_last_endpoint_kw"],
            "storage_charge_discharge_kw_for_this_check": [0, 0], "grid_import_peak_kw": float(grid_peak),
            "grid_import_limit_kw": s["grid_import_limit_kw"], "grid_import_exceedance_intervals": exceed_count,
            "pv_surplus_intervals_before_storage_or_curtailment": surplus_count,
            "pv_surplus_peak_kw": float(surplus_peak), "pv_surplus_energy_kwh": float(surplus_energy),
            "unconstrained_pv_energy_kwh": float(pv_energy), "baseline_grid_import_energy_kwh": float(grid_energy),
            "baseline_curtailment_if_storage_idle_kwh": float(surplus_energy), "baseline_export_kw": 0,
            "derived_meter_timeseries_written": False, "storage_soc_simulated": False}


def self_tests(cfg, tables):
    results = {}

    def rejection(name, expected_code, mutate):
        changed = deepcopy(tables)
        mutate(changed)
        try:
            validate_tables(cfg, changed)
        except RegistryError as exc:
            require(exc.code == expected_code, "SELF_TEST", f"{name}应报{expected_code}，实际{exc.code}")
            results[name] = {"status": "passed", "rejected_with": exc.code}
        else:
            raise RegistryError("SELF_TEST", f"{name}错误未被拒绝")

    rejection("duplicate_asset_id", "DUPLICATE_ASSET", lambda t: t["assets"].append(deepcopy(t["assets"][0])))
    rejection("orphan_asset", "ORPHAN_ASSET", lambda t: t["assets"][-1].update(parent_asset_id="MISSING"))
    rejection("topology_cycle", "TOPOLOGY_CYCLE", lambda t: t["assets"][0].update(parent_asset_id="PCC"))
    rejection("missing_port", "MISSING_PORT", lambda t: t["assets"].__setitem__(slice(None), [a for a in t["assets"] if a["asset_id"] != "SIM-G-128"]))
    rejection("inconsistent_pv_capacity", "PV_CAPACITY", lambda t: next(a for a in t["assets"] if a["asset_id"] == "PV-INV-001").update(rated_power_kw=499, max_available_power_kw=499))
    rejection("available_power_above_nameplate", "ASSET_POWER_LIMIT", lambda t: next(a for a in t["assets"] if a["asset_id"] == "SIM-G-001").update(max_available_power_kw=241))
    rejection("invalid_meter_reference", "METER_REFERENCE", lambda t: t["meters"][0].update(measured_asset_id="MISSING"))
    rejection("parent_child_power_double_count", "DOUBLE_COUNT", lambda t: next(c for c in t["constraints"] if c["constraint_id"] == "C-CHARGE-TOTAL")["member_asset_ids"].append("CHG-001"))
    rejection("pcs_power_above_nameplate", "PCS_LIMIT", lambda t: next(c for c in t["constraints"] if c["constraint_id"] == "C-PCS-CHARGE").update(limit_value=2001))
    sample = [{"port_id": "TEST", "start_at": "2024-12-31T23:55:00+08:00", "end_at": "2025-01-01T00:05:00+08:00"}]
    require(occupied_ports(sample, instant(sample[0]["start_at"])) == ["TEST"]
            and occupied_ports(sample, instant("2025-01-01T00:00:00+08:00")) == ["TEST"]
            and occupied_ports(sample, instant(sample[0]["end_at"])) == [], "SELF_TEST", "跨年或左闭右开占用规则错误")
    results["cross_year_occupancy_and_interval_end_release"] = {"status": "passed"}
    try:
        number("NaN")
    except RegistryError as exc:
        require(exc.code == "INVALID_NUMBER", "SELF_TEST", "NaN错误类型不符")
        results["non_finite_numeric_input"] = {"status": "passed", "rejected_with": exc.code}
    else:
        raise RegistryError("SELF_TEST", "NaN未被拒绝")
    return results


def notes(cfg, stats, history):
    station, eq = cfg["station"], cfg["equipment"]
    lines = [
        "# 黄花站统一模拟场站配置与设备台账", "",
        f"建档日期：{cfg['recorded_on']}。数据版本：`{cfg['dataset_id']}`；配置版本：`{cfg['config_version']}`。", "",
        "本包提供单站光储充模拟的统一设备、拓扑、功率约束和计量定义。全部为模拟；设计资料只支持设备类别和约束方向，不能证明实站铭牌、建设状态或安全认证。", "",
        "## 1. 场站配置", "",
        "| 项目 | 当前模拟值与边界 |", "| --- | --- |",
        f"| 站点 | `{station['station_id']}`，黄花充换电站 |",
        f"| 时区及有效期 | {station['timezone']}；[{station['effective_from']}, {station['effective_to']}) |",
        "| 时间含义 | 配置在模拟起点已知；实际建档日期另记，不能解释为2025年实站已投运 |",
        f"| 光伏 | {eq['pv']['capacity_kw']}kW，{eq['pv']['inverter_count']}台×{eq['pv']['inverter_rated_power_kw']}kW逆变器；沿用原聚合数据 |",
        f"| 充电 | {eq['charging']['terminal_count']}台独立单枪终端，{eq['charging']['port_count']}个端口，每条终端/枪口流上限{eq['charging']['port_rated_power_kw']}kW |",
        f"| 并网 | 购电≤{station['grid_import_limit_kw']}kW；禁止反送，售电上限0 |",
        f"| 储能 | 1套{eq['storage']['rated_capacity_kwh']}kWh电池系统、1台{eq['storage']['pcs_rated_power_kw']}kW PCS |",
        "| 储能资格 | 只有名义容量/PCS功率；效率、当前SOC、上下限、备用和末端目标为空，dispatch_ready=false |",
        "| 基准状态 | 全部设备available，无故障、降额、保护、人工接管或冻结；available不表示枪口空闲 |",
        "| 身份 | 模拟ID；真实厂家、型号、序列号未提供，CSV留空 |", "",
        f"台账共{stats['asset_rows']}行，包含设备、逻辑节点及电表；不能把行数当作物理设备总数。", "",
        "| 类型 | 行数 |", "| --- | --- |",
    ]
    lines.extend(f"| {kind} | {count} |" for kind, count in stats["asset_counts"].items())
    lines += ["", "## 2. 文件和读取约定", "",
              "`simulation_config.json`是唯一参数入口；四份CSV、本文和报告由脚本生成，不手工修改派生文件。", "",
              "| 文件 | 内容 |", "| --- | --- |",
              "| 设备台账.csv | 逐台设备、逐端口及父子关系，含逻辑节点和计量设备 |",
              "| 功率约束.csv | 并网净功率、叶子资源和终端/枪口共用限制 |",
              "| 计量关系.csv | 5个模拟计量点的边界、流向和原字段映射 |",
              "| 来源与假设.csv | 四类依据、原文位置和文件SHA-256 |",
              "| validation_report.json | 本次生成的实际校验、来源/产物摘要及历史兼容检查 |",
              "| generate_station_registry.py | 仅使用Python标准库的确定性生成和只读校验入口 |", "",
              "CSV使用UTF-8 BOM、英文逗号、LF换行，功率/电量保留6位小数。布尔值为true/false；空单元格表示未提供或不适用，不填0。数组和对象单元格使用JSON。", "",
              "## 3. 拓扑与计量", "", "```text", "STATION", "  PCC [M-PCC：并网总边界]", "    BUS-AC",
              "      BR-PV [M-PV] → PV-INV-001…010（AC输出，合计5MW）",
              "      BR-CHARGE [M-CHARGE] → CHG-001…128 → SIM-G-001…128（一对一）",
              "      BR-STORAGE [M-STORAGE] → PCS-001（AC）→ BESS-001（DC名义能量）",
              "      BR-BASE [M-BASE] → LOAD-BASE", "```", "",
              "父子关系是模拟分组和计量等效关系，不是现场施工接线图；meter节点挂在其计量位置，计量设备本身不参与功率累计。", "",
              "充电终端和枪口共用一条AC支路等效功率变量。128×240=30,720kW是装机叶子上限之和，不能据此承诺全站同时可用30.72MW；还要满足并网10MW净购电及其他资源约束。", "",
              "枪口连接类型设为DC，但额定功率在本包中沿用历史充电支路等效口径，不代表核实的DC输出铭牌或电池侧功率。历史会话120/180/240kW是会话功率，不把固定端口额定功率降成120或180kW。", "",
              "光伏支路与10台逆变器的总功率只计一次，原CSV保持站级聚合，不均分制造单机实测。PCS是AC侧功率边界；4MWh是电池名义能量，没有效率/SOC时不能保证2MW持续2小时。基础负荷设备的额定功率未知留空，不从历史峰值反推铭牌。", "",
              "五个计量点均为definition_only、is_independent_measurement=false。并网功率如按平衡推导，必须标记为派生；不能将上级总表和支路表再相加。M-PV的目标通道是实际利用出力，原光伏CSV仅提供不限发的参考出力。", "",
              "```text", cfg["model"]["pcc_balance"], cfg["model"]["pv_balance"], "interval_energy_kwh = interval_average_kw × 0.25h", "```", "",
              "本轮没有独立变压器/PCS/充电损耗模型，不再添加未定义损耗。后续引入效率时须统一AC/DC边界，不能把同一损耗重复计入。", "",
              "## 4. 原数据映射与时间纪律", "",
              "原始文件位置与SHA-256固定在配置sources中，路径相对项目根目录；字段映射在sources.mapping及计量关系表中。原文件、平台配置和模型均不修改。", "",
              "- `pv_power_kw`是瞬时值。已完成区间按相邻点梯形平均，电量再乘0.25；必须等右端点到达。末区间按原光伏说明追加2026年元旦00:00夜间零点。",
              "- `base_load_kw`、`charging_load_kw`、`total_load_kw`为区间平均值，区间结束后完整可用；电量直接乘0.25，不能再做梯形积分。",
              "- `base_load_kw`排除充电；`total_load_kw=base+charging`，尚未扣光伏/储能，不是并网净负荷。历史充电与可优化剩余需求应分开，避免再叠加整单新计划。",
              "- `port_id`一对一映射枪口。占用采用[start_at,end_at)，包含跨年会话；实际end_at仅用于历史占用复核，不能当作预计离站或服务承诺。",
              "- 本包未生成车辆需求、储能状态轨迹、电价或实时预测，也未接入平台。平台mock-station-001与本包尺度不同，未来需要显式适配，不能仅改站号。", "",
              "## 5. 本次实际校验", "",
              f"主表分别{history['pv_rows']:,}行，会话{history['session_rows']:,}条；时间连续、唯一、对齐，基础与充电分量严格守恒。历史会话覆盖全部128端口，峰值均不超过240kW，同端口无重叠。", "",
              f"2025年起点有{history['initial_occupied_port_count']}个端口承接跨年会话；年末有{history['sessions_continuing_at_year_end']}条会话延续到2026年。设备available不意味着这些端口空闲。完整年初占用端口见报告。", "",
              f"储能闲置、采用上述光伏积分假设时，15分钟平均购电峰值{history['grid_import_peak_kw']:.6f}kW，超过{station['grid_import_limit_kw']}kW的区间为{history['grid_import_exceedance_intervals']}个。", "",
              f"限发前有{history['pv_surplus_intervals_before_storage_or_curtailment']}个光伏富余区间，富余参考电量{history['pv_surplus_energy_kwh']:.6f}kWh。禁止反送且储能闲置时，这部分需限发；这是离线兼容对照，不是已实现的优化或实站收益。", "",
              "全年检查读取右端点及实际会话结束，是事后审计，禁止作为决策时输入。15分钟检查不证明分钟级峰值、潮流、电压或实站安全。", "",
              "校验还实际注入重复编号、孤立设备、拓扑环、遗漏端口、错误光伏容量、失效计量关联、父子重复计功率、PCS越额定功率及非有限数，确认明确拒绝；另检查跨年占用的起止边界。报告保存逐项结果。", "",
              "## 6. 使用与复现", "",
              "要求Python 3.10及以上，仅使用标准库。以下在本文件夹执行；--validate-only和--self-test均不写文件。", "", "```powershell",
              "python -B .\\generate_station_registry.py --validate-only",
              "python -B .\\generate_station_registry.py --self-test", "",
              "# 从相同配置重新生成到另一个目录；保留原始包",
              "python -B .\\generate_station_registry.py --output-dir ..\\场站台账复现检查", "```", "",
              "默认输出目录为配置所在目录，已有产物时拒绝覆盖；只有显式--overwrite才重建已知产物。--output-dir会复制配置和脚本，形成同样的8文件数据包。复制包到不同目录层级后，使用--project-root显式指定原项目根目录。", "",
              "生成前核对全部来源SHA-256，生成后再次核对；来源变化立即失败，不能悄悄换数据。每次生成均重建两次并比较字节；落盘后重新读取四份CSV校验结构/引用，再比对产物字节。--validate-only还核对保存的报告与产物摘要，且比较检查前后的全部8文件摘要。", "",
              "同一配置、脚本、输入文件和Python环境下，全部8文件可逐字节复现。报告的generation_runtime记录环境；recorded_on是固定建档日期，不冒充每次校验时间，也不在数据中混入不断变化的系统时间。", "",
              "## 7. 来源与未建模边界", "",
              "来源分类为设计资料参考、既有模拟设定、本次已选模拟参数、新增模拟假设。具体原文位置和完整摘要见来源与假设表，20条依据均可追溯。", "",
              "资料包括《场站智能体详细设计方案_合并版》的设备/容量边界、《场站功能运行机理说明》的状态与计量关系，以及《场站层7.08》的资源评估原则。它们没有提供黄花站完整逐台铭牌；10MW购电和2MW/4MWh储能来自本次用户选择。", "",
              "本轮没有建立变压器铭牌/阻抗、潮流和电压、无功、保护定值、真实设备控制、储能SOC/效率/备用、车辆电池模型或真实设备身份。储能运行模型补齐前，PCS上限只是设备边界，不是已认证可调能力。", "",
              "## 8. 字段字典", ""]
    for label, schema in [("四表共有字段", COMMON), ("设备台账", ASSET), ("功率约束", CONSTRAINT), ("计量关系", METER), ("来源与假设", PROVENANCE)]:
        lines += [f"### {label}", "", "| 字段 | 含义 |", "| --- | --- |"]
        lines.extend(f"| `{key}` | {text} |" for key, text in schema.items())
        lines.append("")
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def prepare(cfg, config_data, project_root):
    validate_config(cfg)
    sources = audit_sources(cfg, project_root)
    tables = build_tables(cfg, digest(config_data))
    stats = validate_tables(cfg, tables)
    history = audit_history(cfg, project_root, tables)
    tests = self_tests(cfg, tables)
    blobs = {cfg["output"][kind]: csv_bytes(list(COMMON) + list(SCHEMAS[kind]), records) for kind, records in tables.items()}
    decoded = {kind: csv_records(blobs[cfg["output"][kind]]) for kind in SCHEMAS}
    require(validate_tables(cfg, decoded) == stats, "CSV_ROUNDTRIP", "CSV序列化后校验结果变化")
    second = build_tables(cfg, digest(config_data))
    repeat = {cfg["output"][kind]: csv_bytes(list(COMMON) + list(SCHEMAS[kind]), records) for kind, records in second.items()}
    require(all(blobs[name] == repeat[name] for name in repeat), "REPRODUCIBILITY", "相同输入重复构造台账不一致")
    blobs[cfg["output"]["notes"]] = notes(cfg, stats, history)
    report = {"schema_version": cfg["schema_version"], "dataset_id": cfg["dataset_id"], "config_version": cfg["config_version"],
              "is_simulated": True, "status": "passed", "config_recorded_on": cfg["recorded_on"],
              "generation_runtime": {"python": platform.python_version(), "dependencies": "standard_library_only"},
              "checks": stats, "history_compatibility": history, "negative_and_boundary_tests": tests,
              "reproducibility": {"deterministic_table_bytes": True, "csv_roundtrip_verified": True,
                                  "written_artifacts_read_back": True, "report_self_hash_excluded": True},
              "source_files": sources, "scope": {"platform_connected": False, "original_data_rewritten": False,
              "models_trained_or_called": False, "storage_dispatch_implemented": False, "real_equipment_certified": False}}
    return blobs, report


def package_bytes(cfg, config_data, script_data, blobs, report):
    result = {"simulation_config.json": config_data, "generate_station_registry.py": script_data, **blobs}
    report["artifact_hashes_sha256"] = {name: digest(data) for name, data in sorted(result.items())}
    result[cfg["output"]["report"]] = json_bytes(report)
    return result


def validate_disk(cfg, target, expected):
    for name in expected:
        require((target / name).is_file(), "ARTIFACT_MISSING", f"产物缺失: {target / name}")
    # Report structural errors before a generic byte/hash mismatch.
    tables = {kind: csv_records((target / cfg["output"][kind]).read_bytes()) for kind in SCHEMAS}
    validate_tables(cfg, tables)
    for name, data in expected.items():
        path = target / name
        actual = path.read_bytes()
        if name == cfg["output"]["report"]:
            saved, current = json.loads(actual), json.loads(data)
            # Different Python patch versions do not invalidate the stored data.
            saved.pop("generation_runtime", None)
            current.pop("generation_runtime", None)
            require(saved == current, "REPORT_MISMATCH", "保存的报告与当前输入、产物或实算结果不符")
        else:
            require(actual == data, "ARTIFACT_MISMATCH", f"产物与配置/来源不一致: {name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().with_name("simulation_config.json"))
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true", help="只读校验，不刷新报告")
    mode.add_argument("--self-test", action="store_true", help="只读运行关系错误及边界检查")
    parser.add_argument("--overwrite", action="store_true", help="显式允许重建本包产物")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    root = (args.project_root or config_path.parents[3]).resolve()
    target = (args.output_dir or config_path.parent).resolve()
    config_data = config_path.read_bytes()
    script_data = Path(__file__).resolve().read_bytes()
    cfg = json.loads(config_data)
    before = {}
    if args.validate_only:
        require(target.is_dir(), "ARTIFACT_MISSING", f"数据包不存在: {target}")
        before = {p.name: digest(p.read_bytes()) for p in target.iterdir() if p.is_file()}
    blobs, report = prepare(cfg, config_data, root)
    expected = package_bytes(cfg, config_data, script_data, blobs, report)
    if args.validate_only:
        validate_disk(cfg, target, expected)
        after = {p.name: digest(p.read_bytes()) for p in target.iterdir() if p.is_file()}
        require(before == after, "READ_ONLY_VIOLATION", "只读校验改变了包文件")
    elif not args.self_test:
        protected = {(root / spec["path"]).resolve() for spec in cfg["sources"].values()}
        for name, data in expected.items():
            path = target / name
            require(path.resolve() not in protected, "PROTECTED_SOURCE", f"输出会覆盖冻结来源，已拒绝: {path}")
            if path.exists() and path.resolve() not in (config_path, Path(__file__).resolve()):
                require(args.overwrite, "OUTPUT_EXISTS", f"已有产物{path}；另选输出目录，或显式使用--overwrite")
            require(not path.is_symlink(), "OUTPUT_PATH", f"输出不能为符号链接: {path}")
        target.mkdir(parents=True, exist_ok=True)
        # Write/read back data first; only then publish a report claiming success.
        for name, data in expected.items():
            if name != cfg["output"]["report"]:
                (target / name).write_bytes(data)
        for name, data in expected.items():
            if name != cfg["output"]["report"]:
                require((target / name).read_bytes() == data, "WRITE_READBACK", f"落盘回读不一致: {name}")
        validate_tables(cfg, {kind: csv_records((target / cfg["output"][kind]).read_bytes()) for kind in SCHEMAS})
        audit_sources(cfg, root)
        (target / cfg["output"]["report"]).write_bytes(expected[cfg["output"]["report"]])
        validate_disk(cfg, target, expected)
    audit_sources(cfg, root)
    print(json.dumps({"status": "passed", "mode": "validate_only" if args.validate_only else "self_test" if args.self_test else "generate",
                      "output_directory": str(target), "file_count": len(expected), "asset_rows": report["checks"]["asset_rows"],
                      "negative_and_boundary_cases": len(report["negative_and_boundary_tests"]),
                      "initial_occupied_ports": report["history_compatibility"]["initial_occupied_port_count"],
                      "pv_surplus_intervals": report["history_compatibility"]["pv_surplus_intervals_before_storage_or_curtailment"],
                      "original_source_hashes_unchanged": True, "read_only_files_unchanged": True if args.validate_only else None}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    try:
        raise SystemExit(main())
    except (RegistryError, OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "failed", "code": getattr(exc, "code", "INPUT_OR_IO_ERROR"), "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
