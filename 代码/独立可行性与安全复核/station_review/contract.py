"""Canonical transport only. No optimizer, platform, or optimizer validator imports."""
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import math

SCHEMA = "a03-frozen-v1"
STATE_SCHEMA = "a03-state-v1"
RULE = "a03-offline-governance-v1"
EPS = 1e-5  # absolute kW/kWh tolerance, independently fixed by this reviewer
MAX_STATE_AGE_SECONDS = 60


class ReviewError(ValueError):
    def __init__(self, code, field, actual=None, expected=None):
        self.code, self.field = code, field
        self.actual, self.expected = actual, expected
        super().__init__(f"{code}: {field}")


def need(condition, code, field, actual=None, expected=None):
    if not condition:
        raise ReviewError(code, field, actual, expected)


def _json_tree(value):
    if type(value) is dict:
        need(all(type(k) is str for k in value), "INPUT_INVALID", "JSON object keys")
        for v in value.values():
            _json_tree(v)
    elif type(value) is list:
        for v in value:
            _json_tree(v)
    else:
        need(value is None or type(value) in (str, bool, int, float), "INPUT_INVALID", "JSON value type")
        if type(value) in (int, float):
            need(math.isfinite(value), "NONFINITE_NUMBER", "JSON number")


def canonical(value):
    _json_tree(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def content_hash(plan):
    return digest({k: v for k, v in plan.items() if k != "fingerprint"})


def time_at(value, field="timestamp"):
    try:
        t = datetime.fromisoformat(value)
        need(t.utcoffset() == timedelta(hours=8), "TIMEZONE_INVALID", field, value, "explicit UTC+08:00")
        return t
    except ReviewError:
        raise
    except (TypeError, ValueError, AttributeError):
        raise ReviewError("TIME_INVALID", field, expected="ISO8601 with UTC+08:00") from None


def num(value, field, low=0, high=1e9):
    need(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
         "NUMBER_INVALID", field, expected=[low, high])
    return value


def integer(value, field, low, high):
    need(type(value) is int and low <= value <= high, "INTEGER_INVALID", field, value, [low, high])
    return value


def text_id(value, field):
    need(type(value) is str and 0 < len(value.strip()) <= 200, "IDENTITY_INVALID", field)
    return value


def sha(value, field):
    need(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
         "HASH_INVALID", field)
    return value


def vector(value, field, n, low=0):
    need(type(value) is list and len(value) == n, "LENGTH_INVALID", field, expected=n)
    for x in value:
        num(x, field, low)
    return value


def slot_list(value, field, n):
    need(type(value) is list, "INPUT_INVALID", field)
    for x in value:
        integer(x, field, 0, n-1)
    need(len(set(value)) == len(value), "DUPLICATE_SLOT", field)
    return value


def exact_map(value, keys, field):
    need(type(value) is dict and set(value) == set(keys), "MAPPING_INVALID", field, expected=sorted(keys))
    return value


def boolean(value, field):
    need(type(value) is bool, "BOOLEAN_INVALID", field)
    return value


@dataclass(frozen=True)
class FrozenPlan:
    """Own immutable bytes; all reads return newly decoded objects."""
    payload: bytes

    def __post_init__(self):
        need(type(self.payload) is bytes, "INPUT_INVALID", "FrozenPlan.payload")
        need(type(json.loads(self.payload)) is dict, "INPUT_INVALID", "frozen document")

    def to_dict(self):
        return json.loads(self.payload)

    @classmethod
    def from_dict(cls, value):
        return cls(canonical(value))


def freeze_plan(constraints, plan, *, producer, producer_version, publisher, plan_version,
                dependencies, issued_at, valid_until, execute_from_slot=0, executed_prefix=None):
    """Bind an optimizer's exact bytes/constraints to A03 publication metadata.

    This constructs a review request, NEVER an authorization or feasible result.
    Validation is intentionally performed by the independent review entry point.
    """
    prefix, event_hashes = [], {}
    for row in ([] if executed_prefix is None else executed_prefix):
        core = {key: value for key, value in row.items() if key != "received_at"}
        event_id = core["event_id"]
        h = digest(core)
        if event_id in event_hashes:
            need(h == event_hashes[event_id], "RECEIPT_ID_CONFLICT", event_id)
            continue
        event_hashes[event_id] = h
        prefix.append(core)
    prefix.sort(key=lambda row: row["slot"])
    body = {"schema_version": SCHEMA, "constraints": constraints, "plan": plan,
            "constraints_hash": digest(constraints), "plan_fingerprint": plan["fingerprint"],
            "producer": producer, "producer_version": producer_version, "publisher": publisher,
            "plan_version": plan_version, "dependencies": dependencies, "issued_at": issued_at,
            "valid_until": valid_until, "execute_from_slot": execute_from_slot,
            "executed_prefix_hash": digest(prefix)}
    body["fingerprint"] = digest(body)
    return FrozenPlan.from_dict(body)


def constraints_shape(d):
    """Independent input contract: A02 four slots or native A03 1..96 slots."""
    need(d["schema_version"] in {"a02-joint-v1", "a03-constraints-v1"}, "SCHEMA_UNSUPPORTED", "constraints.schema_version")
    start, end = (time_at(d[k], k) for k in ("window_start", "window_end"))
    need(type(d["interval_minutes"]) is int and d["interval_minutes"] == 15, "GRID_INVALID", "interval_minutes")
    n_float = (end-start).total_seconds()/900
    need(n_float.is_integer() and 1 <= n_float <= 96, "GRID_INVALID", "window length")
    n = int(n_float)
    if d["schema_version"] == "a02-joint-v1":
        need(n == 4 and time_at(d["as_of"]) <= start, "GRID_INVALID", "A02 window")
    need(start.minute % 15 == 0 and start.second == 0 and start.microsecond == 0, "GRID_INVALID", "window_start")
    for key in ("station_id", "task_id"):
        text_id(d[key], key)
    integer(d["task_version"], "task_version", 1, 2**31-1)
    need(d["meter_boundary"] == "station_ac", "METER_BOUNDARY_INVALID", "meter_boundary")
    need(d["commitment_status"] == "complete", "COMMITMENT_UNKNOWN", "commitment_status")
    need(type(d["source_refs"]) is dict and bool(d["source_refs"]), "SOURCE_MISSING", "source_refs")
    for key, value in d["source_refs"].items():
        text_id(key, "source_ref")
        sha(value, key)
    need(type(d["assumptions"]) is list and all(type(x) is str for x in d["assumptions"]), "INPUT_INVALID", "assumptions")
    for key in ("base_load_kw", "pv_available_kw", "grid_import_limit_kw"):
        vector(d[key], key, n)
    need(type(d["grid_export_limit_kw"]) in (int, float) and d["grid_export_limit_kw"] == 0,
         "EXPORT_NOT_SUPPORTED", "grid_export_limit_kw")
    if d["new_charge_limit_kw"] is not None:
        vector(d["new_charge_limit_kw"], "new_charge_limit_kw", n)
    need(type(d["ports"]) is list and 1 <= len(d["ports"]) <= 128, "MAPPING_INVALID", "ports")
    ports = {}
    for p in d["ports"]:
        pid = text_id(p["port_id"], "port_id")
        need(pid not in ports, "DUPLICATE_ID", "port_id", pid)
        ports[pid] = p
        text_id(p["connector"], "connector")
        vector(p["max_power_kw"], "port.max_power_kw", n)
        slot_list(p["occupied_slots"], "occupied_slots", n)
    exact_map(d["existing_charge_by_port_kw"], ports, "existing_charge_by_port_kw")
    for pid, values in d["existing_charge_by_port_kw"].items():
        vector(values, "existing_charge_by_port_kw", n)
        need(all(x == 0 or t in ports[pid]["occupied_slots"] for t, x in enumerate(values)),
             "COMMITMENT_OCCUPANCY_MISSING", pid)
    need(type(d["vehicles"]) is list and len(d["vehicles"]) <= 128, "MAPPING_INVALID", "vehicles")
    vehicles = {}
    for v in d["vehicles"]:
        vid = text_id(v["vehicle_id"], "vehicle_id")
        need(vid not in vehicles, "DUPLICATE_ID", "vehicle_id", vid)
        vehicles[vid] = v
        text_id(v["connector"], "vehicle.connector")
        a = integer(v["arrival_slot"], "arrival_slot", 0, n-1)
        integer(v["deadline_slot"], "deadline_slot", a+1, n)
        num(v["required_energy_kwh"], "required_energy_kwh", 1e-4)
        num(v["max_power_kw"], "vehicle.max_power_kw", 1e-4)
        allowed = v["allowed_port_ids"]
        need(type(allowed) is list and all(type(x) is str for x in allowed) and bool(allowed)
             and len(set(allowed)) == len(allowed) and set(allowed) <= set(ports), "MAPPING_INVALID", "allowed_port_ids")
    s = d["storage"]
    text_id(s["storage_id"], "storage_id")
    cap = num(s["capacity_kwh"], "capacity_kwh", 1e-4)
    for key in ("initial_energy_kwh", "terminal_energy_kwh"):
        num(s[key], key, 0, cap)
    for key in ("min_soc", "max_soc", "reserve_soc"):
        num(s[key], key, 0, 1)
    need(s["min_soc"] <= s["max_soc"] and s["reserve_soc"] <= s["max_soc"], "SOC_BOUNDS_INVALID", "storage")
    for key in ("charge_efficiency", "discharge_efficiency"):
        num(s[key], key, 1e-4, 1)
    for key in ("max_charge_kw", "max_discharge_kw"):
        vector(s[key], key, n)
    need(s["terminal_rule"] == "hard_equality_at_window_end", "TERMINAL_RULE_UNSUPPORTED", "storage.terminal_rule")
    return n, start, end, ports, vehicles
