"""Business-only input and an explicitly synthetic, replaceable station adapter."""
import csv
from copy import deepcopy
from datetime import timedelta
from decimal import Decimal, InvalidOperation
from uuid import uuid4

from .common import PocError, now, stamp


def validate_demand(demand):
    if not isinstance(demand, dict) or set(demand) != {"vehicles"}:
        raise PocError("INPUT_INVALID", "需求只包含 vehicles 车辆列表，不接受候选或场站边界")
    vehicles = demand["vehicles"]
    if not isinstance(vehicles, list) or not 1 <= len(vehicles) <= 12:
        raise PocError("INPUT_INVALID", "本版每次填写 1～12 辆车")
    ids = set()
    fields = {"vehicle_id", "requested_energy_kwh", "arrival_minutes", "departure_minutes", "priority"}
    for v in vehicles:
        if not isinstance(v, dict) or set(v) != fields:
            raise PocError("INPUT_INVALID", "每辆车需填写编号、补能量、到站分钟、离站分钟、优先级")
        if not isinstance(v["vehicle_id"], str) or not v["vehicle_id"].strip() or len(v["vehicle_id"]) > 60 or v["vehicle_id"] in ids:
            raise PocError("INPUT_INVALID", "车辆编号不能为空、重复或超过 60 字")
        ids.add(v["vehicle_id"])
        try:
            energy = Decimal(str(v["requested_energy_kwh"]))
            if isinstance(v["requested_energy_kwh"], bool) or not energy.is_finite() or not 0 < energy <= 1000 or energy * 1000 != (energy * 1000).to_integral_value():
                raise ValueError()
        except (InvalidOperation, ValueError, TypeError):
            raise PocError("INPUT_INVALID", "补能量须大于 0、不超过 1000 kWh，最多三位小数") from None
        a, d = v["arrival_minutes"], v["departure_minutes"]
        if type(a) is not int or type(d) is not int or a % 15 or d % 15 or not 0 <= a < d <= 60:
            raise PocError("INPUT_INVALID", "到离站时间须为 0～60 分钟内的 15 分钟整点，且离站晚于到站")
        if v["priority"] not in ("critical", "ordinary"):
            raise PocError("INPUT_INVALID", "优先级须为关键或普通")


def read_demand_csv(path):
    vehicles = []
    columns = ["车辆编号", "补能量kWh", "到站分钟", "离站分钟", "优先级"]
    try:
        with open(path, encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != columns:
                raise ValueError()
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError()
                vehicles.append({"vehicle_id": row[columns[0]].strip(), "requested_energy_kwh": float(row[columns[1]]),
                    "arrival_minutes": int(row[columns[2]]), "departure_minutes": int(row[columns[3]]),
                    "priority": {"关键": "critical", "普通": "ordinary"}[row[columns[4]].strip()]})
    except (ValueError, KeyError, UnicodeError, csv.Error):
        raise PocError("INPUT_INVALID", "需求表格式错误，请使用 needs.example.csv 的五列表头、数值和关键/普通优先级") from None
    demand = {"vehicles": vehicles}
    validate_demand(demand)
    return demand


class MockStationAdapter:
    """No hardware connection. Fleet capabilities below are declared test assumptions."""
    def __init__(self, site=None):
        self.site = deepcopy(site) if site is not None else {
            "equipment": [{"asset_id": f"charger-{i}", "max_power_kw": 200, "connector": "DC",
                "availability": "available", "occupied_slots": [], "data_quality": "A"} for i in range(1, 4)],
            "energy": {"pv_available_kw": [100, 100, 150, 150], "base_load_kw": [50]*4,
                "grid_import_limit_kw": 500, "reverse_export_allowed": False,
                "storage": {"capacity_kwh": 500, "soc_percent": 60, "reserve_soc_percent": 30,
                            "dispatch_mode": "idle", "data_quality": "A"}},
            "safety": {"protection_active": False, "manual_takeover": False, "frozen_assets": [], "limits_known": True},
            "rates_cny_per_kwh": [0.4, 0.4, 1.2, 1.2], "cost_tolerance_cny": 20}

    def capture(self, demand, at=None):
        validate_demand(demand)
        at = (at or now()).replace(microsecond=0)
        start, end = stamp(at), stamp(at + timedelta(hours=1))
        ids = {"request_id": "needs-" + uuid4().hex, "snapshot_id": "mock-state-" + uuid4().hex,
            "station_id": "mock-station-001", "rule_version": "mock-rule-v2", "strategy_version": "service-search-v1"}
        site = deepcopy(self.site)
        vehicles = [{"vehicle_id": v["vehicle_id"], "requested_energy_kwh": v["requested_energy_kwh"],
            "arrival_slot": v["arrival_minutes"] // 15, "deadline_slot": v["departure_minutes"] // 15,
            "priority": v["priority"], "max_charge_power_kw": 200, "connector": "DC", "soc_percent": 30,
            "data_quality": "B"} for v in demand["vehicles"]]
        snapshot = {**ids, "generated_at": start, "expires_at": end, "clock_source": "test_runner_utc",
            "operation_mode": "normal", "data_quality_summary": "B", "vehicles": vehicles,
            "equipment": site["equipment"], "energy": site["energy"], "safety": site["safety"],
            "economic_context": {"tariff_config_ref": "mock-tariff", "tariff_version": "mock-tariff-v1",
                "effective_from": start, "effective_to": end, "rates_cny_per_kwh": site["rates_cny_per_kwh"],
                "cost_tolerance_ref": "mock-cost-tolerance", "cost_tolerance_version": "mock-tolerance-v1",
                "cost_tolerance_cny": site["cost_tolerance_cny"], "joint_approval_ref": "mock-approval-not-real",
                "data_quality_summary": "A"}, "audit_refs": ["mock-station-adapter-v2"],
            "simulation_assumptions": ["所有车辆按模拟 DC 车型处理：受电上限 200 kW、SOC 30%、无充电损耗；不代表真实车辆事实",
                "场站状态、电价、光伏和容量由模拟适配器提供；占用时段禁止新分配，既有负荷须计入 base_load_kw",
                "未来一小时，15 分钟粒度；只支持单桩连续服务，储能 idle；未接真实充电曲线或换电"]}
        return {"request_id": ids["request_id"], "user_demand": deepcopy(demand),
            "station_state_snapshot": snapshot, "service_orchestration_plan": None,
            "poc_context": {"simulation": True, "evaluation_window": {"effective_from": start, "effective_to": end},
                "interval_minutes": 15, "green_data_available": True, "baseline": None,
                "capacity_result": {**ids, "result_id": "mock-capacity-result", "simulation": True, "expires_at": end,
                    "grid_import_limit_kw": site["energy"]["grid_import_limit_kw"],
                    "soc_reserve_percent": site["energy"]["storage"]["reserve_soc_percent"], "source": "mock-station-adapter-v2"}}}
