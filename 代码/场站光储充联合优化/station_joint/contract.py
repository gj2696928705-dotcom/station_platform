"""JSON contract and canonical hashes; no optimization or feasibility formulas."""
from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
import math

SCHEMA = "a02-joint-v1"
N = 4
DT = 0.25
EPS = 1e-5  # absolute kW / kWh / CNY; never a percentage of station capacity


class InputError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def fingerprint(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def require(condition, message, code="INPUT_INVALID"):
    if not condition:
        raise InputError(code, message)


def number(value, name, low=0, high=1e7):
    require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
            f"{name}: expected finite number in [{low}, {high}]")
    return value


def curve(value, name, low=0, high=1e7, size=N):
    require(type(value) is list and len(value) == size, f"{name}: expected {size} explicit values")
    for v in value:
        number(v, name, low, high)
    return value


def identifier(value, name):
    require(type(value) is str and bool(value.strip()) and len(value) <= 200, f"{name}: invalid identifier")
    return value


def time_at(value):
    try:
        t = datetime.fromisoformat(value)
        require(t.utcoffset() == timedelta(hours=8), "times must explicitly use UTC+08:00")
        return t
    except (TypeError, ValueError, AttributeError):
        raise InputError("TIME_INVALID", "invalid UTC+08:00 timestamp") from None


def slots(value, name):
    require(type(value) is list and all(type(t) is int and 0 <= t < N for t in value)
            and len(set(value)) == len(value), f"{name}: invalid slots")


def validate_input(data):
    """Reject unknown/missing inputs; no defaults for resources, commitments or SOC."""
    try:
        require(type(data) is dict, "input must be an object")
        d = deepcopy(data)
        require(d["schema_version"] == SCHEMA, "unsupported schema")
        require(d["simulation"] is True and d["execution_mode"] == "offline_replay", "only explicit offline simulation is supported")
        require(d["meter_boundary"] == "station_ac", "AC station meter boundary required")
        for k in ("task_id", "station_id"):
            identifier(d[k], k)
        require(type(d["task_version"]) is int and d["task_version"] > 0, "invalid task version")
        start, end, at = (time_at(d[k]) for k in ("window_start", "window_end", "as_of"))
        require(end - start == timedelta(hours=1) and at <= start, "expected one-hour future/replay window")
        require(d["interval_minutes"] == 15 and type(d["interval_minutes"]) is int, "only four 15-minute intervals supported")
        require(start.minute % 15 == 0 and start.second == 0 and start.microsecond == 0, "start must be on 15-minute grid")
        require(type(d["source_refs"]) is dict and bool(d["source_refs"]), "source references required")
        for key, ref in d["source_refs"].items():
            identifier(key, "source ref")
            require(type(ref) is str and len(ref) == 64 and all(c in "0123456789abcdef" for c in ref), "source ref must be SHA-256")
        require(type(d["assumptions"]) is list and all(type(a) is str for a in d["assumptions"]), "assumptions must be explicit")
        require(d["commitment_status"] == "complete", "existing commitments are unknown", "COMMITMENT_UNKNOWN")
        for k in ("base_load_kw", "pv_available_kw", "grid_import_limit_kw"):
            curve(d[k], k)
        curve(d["rates_cny_per_kwh"], "rates", -100, 100)
        require(d["grid_export_limit_kw"] == 0 and type(d["grid_export_limit_kw"]) in (int, float), "export prohibited")
        if d["new_charge_limit_kw"] is not None:
            curve(d["new_charge_limit_kw"], "new charging budget")
        ports = d["ports"]
        require(type(ports) is list and 1 <= len(ports) <= 128, "expected 1..128 independent ports")
        ids = []
        for p in ports:
            ids.append(identifier(p["port_id"], "port_id"))
            identifier(p["connector"], "connector")
            curve(p["max_power_kw"], "port power")
            slots(p["occupied_slots"], "occupied_slots")
        require(len(set(ids)) == len(ids), "duplicate ports")
        require(type(d["existing_charge_by_port_kw"]) is dict and set(d["existing_charge_by_port_kw"]) == set(ids), "existing charge profiles must cover all ports explicitly")
        for p in ports:
            values = curve(d["existing_charge_by_port_kw"][p["port_id"]], "existing charge")
            require(all(v == 0 or t in p["occupied_slots"] for t, v in enumerate(values)), "existing charging must reserve its port")
        fleet = d["vehicles"]
        require(type(fleet) is list and len(fleet) <= 128, "expected up to 128 vehicles")
        vids = []
        for v in fleet:
            vids.append(identifier(v["vehicle_id"], "vehicle_id"))
            identifier(v["connector"], "vehicle connector")
            a, b = v["arrival_slot"], v["deadline_slot"]
            require(type(a) is int and type(b) is int and 0 <= a < b <= N, "invalid vehicle window")
            number(v["required_energy_kwh"], "vehicle energy", 1e-4, 1e5)
            number(v["max_power_kw"], "vehicle power", 1e-4, 1e5)
            allowed = v["allowed_port_ids"]
            require(type(allowed) is list and bool(allowed) and all(type(p) is str for p in allowed)
                    and len(set(allowed)) == len(allowed) and set(allowed) <= set(ids), "invalid allowed ports")
        require(len(set(vids)) == len(vids), "duplicate vehicles")
        s = d["storage"]
        identifier(s["storage_id"], "storage_id")
        cap = number(s["capacity_kwh"], "storage capacity", 1e-4)
        for k in ("initial_energy_kwh", "terminal_energy_kwh"):
            number(s[k], k, 0, cap)
        for k in ("min_soc", "max_soc", "reserve_soc"):
            number(s[k], k, 0, 1)
        require(s["min_soc"] <= s["max_soc"] and s["reserve_soc"] <= s["max_soc"], "invalid SOC bounds")
        for k in ("charge_efficiency", "discharge_efficiency"):
            number(s[k], k, 1e-4, 1)
        for k in ("max_charge_kw", "max_discharge_kw"):
            curve(s[k], k)
        require(s["terminal_rule"] == "hard_equality_at_window_end", "explicit window terminal equality required")
        for k in ("throughput_cost_cny_per_kwh", "curtailment_cost_cny_per_kwh"):
            number(d["objective"][k], k, 0, 100)
        return d
    except InputError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as e:
        raise InputError("INPUT_INVALID", f"missing or malformed input: {e}") from None


def freeze(data):
    d = validate_input(data)
    return {"input": d, "input_hash": fingerprint(d)}
