"""Independent AC power/energy and latest-state review. Standard library only."""
from datetime import timedelta

from .contract import (EPS, MAX_STATE_AGE_SECONDS, SCHEMA, STATE_SCHEMA, ReviewError,
                       boolean, constraints_shape, content_hash, digest, exact_map, integer,
                       need, num, sha, slot_list, text_id, time_at, vector)


def identity(b):
    need(b["schema_version"] == SCHEMA, "SCHEMA_UNSUPPORTED", "frozen.schema_version")
    sha(b["fingerprint"], "frozen.fingerprint")
    need(content_hash(b) == b["fingerprint"], "FROZEN_HASH_MISMATCH", "frozen.fingerprint")
    p, d = b["plan"], b["constraints"]
    need(digest(d) == b["constraints_hash"], "CONSTRAINT_HASH_MISMATCH", "constraints_hash")
    need(p["input_hash"] == b["constraints_hash"], "OPTIMIZER_INPUT_MISMATCH", "plan.input_hash")
    need(content_hash(p) == p["fingerprint"] == b["plan_fingerprint"], "PLAN_HASH_MISMATCH", "plan.fingerprint")
    expected_schema = "a02-joint-v1" if d["schema_version"] == "a02-joint-v1" else "a03-plan-v1"
    need(p["schema_version"] == expected_schema, "SCHEMA_UNSUPPORTED", "plan.schema_version")
    integer(b["plan_version"], "plan_version", 1, 2**31-1)
    return d, p


def dependencies(b, state, at, start, end):
    refs = b["constraints"]["source_refs"]
    deps = b["dependencies"]
    need(type(deps) is list and bool(deps), "DEPENDENCY_MISSING", "dependencies")
    kinds, names = set(), set()
    for x in deps:
        key = text_id(x["ref"], "dependency.ref")
        kind = x["kind"]
        need(kind in {"forecast_load", "forecast_pv", "commitments"}, "DEPENDENCY_KIND_INVALID", "dependency.kind")
        need((key, kind) not in names, "DUPLICATE_ID", "dependency.ref_and_kind", key)
        names.add((key, kind))
        kinds.add(kind)
        need(x["status"] == "usable", "DEPENDENCY_UNAVAILABLE", key)
        if kind.startswith("forecast_"):
            need(x["native_grid"] == "15min_interval_average", "FORECAST_GRID_INVALID", key)
        sha(x["sha256"], "dependency.sha256")
        need(refs.get(key) == x["sha256"] == state["dependency_hashes"].get(key),
             "DEPENDENCY_CHANGED", key)
        issued, available, valid = (time_at(x[k], "dependency."+k) for k in ("issued_at", "available_at", "valid_until"))
        need(issued <= available <= time_at(b["issued_at"]) <= at < valid,
             "DEPENDENCY_EXPIRED_OR_FUTURE", key)
        need(time_at(x["window_start"]) <= start and time_at(x["window_end"]) >= end,
             "FORECAST_COVERAGE_GAP", key)
    need(kinds == {"forecast_load", "forecast_pv", "commitments"}, "DEPENDENCY_MISSING", "dependency kinds")


def state_shape(s, d, at, n, ports):
    need(s["schema_version"] == STATE_SCHEMA, "SCHEMA_UNSUPPORTED", "state.schema_version")
    text_id(s["state_id"], "state_id")
    integer(s["revision"], "state.revision", 1, 2**31-1)
    for key in ("station_id", "task_id", "task_version", "meter_boundary"):
        need(type(s[key]) is type(d[key]) and s[key] == d[key], "STATE_CONTEXT_MISMATCH", "state."+key, s[key], d[key])
    measured, available, valid = (time_at(s[k], "state."+k) for k in ("measured_at", "available_at", "valid_until"))
    need(measured <= available <= at < valid, "STATE_EXPIRED_OR_FUTURE", "state timestamps")
    age = (at-measured).total_seconds()
    need(age <= MAX_STATE_AGE_SECONDS, "STATE_TOO_OLD", "state age seconds", age, MAX_STATE_AGE_SECONDS)
    need(s["quality"] == "good" and s["measurement"]["quality"] == "good", "MEASUREMENT_QUALITY", "quality")
    need(s["power_semantics"] == "ac_dispatch_snapshot", "MEASUREMENT_SEMANTICS", "power_semantics")
    for key in ("base_load_kw", "pv_available_kw", "storage_energy_kwh", "storage_soc"):
        num(s["measurement"][key], "measurement."+key)
    num(s["measurement"]["net_import_kw"], "measurement.net_import_kw", -1e9)
    num(s["grid_import_limit_kw"], "state.grid_import_limit_kw")
    num(s["pv_max_power_kw"], "state.pv_max_power_kw")
    exact_map(s["dependency_hashes"], d["source_refs"], "dependency_hashes")
    for key, value in d["source_refs"].items():
        need(s["dependency_hashes"][key] == value, "DEPENDENCY_CHANGED", key)
    if s["new_charge_limit_kw"] is not None:
        vector(s["new_charge_limit_kw"], "state.new_charge_limit_kw", n)
    for key in ("online", "protection_active", "manual_takeover", "frozen"):
        boolean(s[key], "state."+key)
    need(s["online"], "STATION_OFFLINE", "state.online", False, True)
    for key in ("protection_active", "manual_takeover", "frozen"):
        need(not s[key], "STATION_"+key.upper(), "state."+key, True, False)
    exact_map(s["ports"], ports, "state.ports")
    asset_ids = set()
    for pid, p in s["ports"].items():
        aid = text_id(p["asset_id"], "port.asset_id")
        need(aid not in asset_ids, "DUPLICATE_ID", "state.port.asset_id", aid)
        asset_ids.add(aid)
        need(p["connector"] == ports[pid]["connector"], "DEVICE_MAPPING_CHANGED", pid)
        num(p["max_power_kw"], "state.port.max_power_kw")
        slot_list(p["occupied_slots"], "state.port.occupied_slots", n)
        device_flags(p, "port."+pid)
    st = s["storage"]
    need(st["storage_id"] == d["storage"]["storage_id"], "DEVICE_MAPPING_CHANGED", "state.storage_id")
    need(st["storage_id"] not in asset_ids, "DUPLICATE_ID", "state.storage_id")
    for key in ("max_charge_kw", "max_discharge_kw"):
        num(st[key], "state.storage."+key)
    device_flags(st, "storage")
    return measured, valid


def device_flags(x, field):
    for key in ("online", "fault", "protection_active", "manual_takeover", "frozen"):
        boolean(x[key], field+"."+key)


def blocked(x):
    return not x["online"] or any(x[k] for k in ("fault", "protection_active", "manual_takeover", "frozen"))


def execution_lineage(execution, n):
    """Trusted accepted-plan ledger partitions slots; stopped plans cannot own new steps."""
    history = execution["accepted_plan_history"]
    need(type(history) is list, "EXECUTION_HISTORY_INVALID", "accepted_plan_history")
    previous_version, end_slot, owners = 0, 0, {}
    for i, entry in enumerate(history):
        version = integer(entry["plan_version"], "history.plan_version", previous_version+1, 2**31-1)
        fp = sha(entry["plan_fingerprint"], "history.plan_fingerprint")
        start = integer(entry["execute_from_slot"], "history.execute_from_slot", 0, n-1)
        need(start == end_slot, "EXECUTION_HISTORY_GAP", "history.execute_from_slot", start, end_slot)
        stop = entry["stop_after_slot"]
        if stop is None:
            need(i == len(history)-1, "EXECUTION_HISTORY_INVALID", "active plan must be last")
            end_slot = n
        else:
            end_slot = integer(stop, "history.stop_after_slot", start, n)
        owners[version] = (fp, start, end_slot)
        previous_version = version
    accepted = execution["accepted_plan_version"]
    need((not history and accepted == 0) or (bool(history) and previous_version == accepted
         and history[-1]["plan_fingerprint"] == execution["accepted_plan_fingerprint"]),
         "EXECUTION_HISTORY_INVALID", "history latest accepted identity")
    if history:
        need((execution["cancel_status"] == "acknowledged" and history[-1]["stop_after_slot"] == execution["stopped_after_slot"])
             or (execution["cancel_status"] != "acknowledged" and history[-1]["stop_after_slot"] is None),
             "EXECUTION_HISTORY_INVALID", "history cancellation boundary")
    return owners


def receipts(b, state, d, at, n, start, vehicles, ports, fail):
    """Recompute an immutable, contiguous actual execution prefix from raw receipts."""
    execution = state["execution"]
    rows = execution["receipts"]
    need(type(rows) is list, "RECEIPT_INVALID", "execution.receipts")
    previous_version = integer(execution["accepted_plan_version"], "accepted_plan_version", 0, 2**31-1)
    previous_fp = execution["accepted_plan_fingerprint"]
    need((previous_version == 0 and previous_fp is None) or
         (previous_version > 0 and sha(previous_fp, "accepted_plan_fingerprint")),
         "EXECUTION_IDENTITY_INVALID", "accepted plan")
    cancel = execution["cancel_status"]
    need(cancel in {"none", "requested", "acknowledged"}, "CANCEL_STATUS_INVALID", "cancel_status")
    owners = execution_lineage(execution, n)
    canonical_rows, seen_ids, seen_slots = [], {}, set()
    for r in rows:
        event_id = text_id(r["event_id"], "receipt.event_id")
        # Transport receive times may differ; business content must be identical.
        core = {k: v for k, v in r.items() if k != "received_at"}
        h = digest(core)
        if event_id in seen_ids:
            need(seen_ids[event_id] == h, "RECEIPT_ID_CONFLICT", event_id)
            continue
        seen_ids[event_id] = h
        for key in ("station_id", "task_id", "task_version"):
            need(r[key] == d[key], "RECEIPT_CONTEXT_MISMATCH", key)
        integer(r["plan_version"], "receipt.plan_version", 1, 2**31-1)
        integer(r["task_version"], "receipt.task_version", 1, 2**31-1)
        t = integer(r["slot"], "receipt.slot", 0, n-1)
        owner = owners.get(r["plan_version"])
        need(owner is not None and r["plan_fingerprint"] == owner[0] and owner[1] <= t < owner[2],
             "RECEIPT_OLD_VERSION", event_id)
        need(t not in seen_slots, "RECEIPT_SLOT_CONFLICT", "receipt.slot", t)
        seen_slots.add(t)
        measured, available = (time_at(r[k], "receipt."+k) for k in ("measured_at", "available_at"))
        need(start+timedelta(minutes=15*(t+1)) <= measured <= available <= at,
             "RECEIPT_TIME_INVALID", event_id)
        if "received_at" in r:
            need(available <= time_at(r["received_at"]) <= at, "RECEIPT_TIME_INVALID", "received_at")
        need(r["quality"] == "good", "RECEIPT_QUALITY", event_id)
        actual = r["actual"]
        connections = actual["connected_vehicle_port_ids"]
        need(type(connections) is dict and set(connections) <= set(vehicles)
             and all(type(pid) is str and pid in ports for pid in connections.values()),
             "RECEIPT_MAPPING_INVALID", "connected_vehicle_port_ids")
        need(len(set(connections.values())) == len(connections), "RECEIPT_PORT_CONFLICT", "connected_vehicle_port_ids")
        for key in ("vehicle_energy_kwh", "existing_by_port_energy_kwh"):
            exact_map(actual[key], vehicles if key == "vehicle_energy_kwh" else ports, "receipt."+key)
            for value in actual[key].values():
                num(value, "receipt."+key)
        need(all(value <= EPS or vid in connections for vid, value in actual["vehicle_energy_kwh"].items()),
             "RECEIPT_MAPPING_INVALID", "nonzero vehicle energy requires connection")
        for key in ("base_energy_kwh", "pv_used_energy_kwh", "storage_charge_energy_kwh",
                    "storage_discharge_energy_kwh", "storage_energy_before_kwh", "storage_energy_after_kwh", "grid_import_energy_kwh"):
            num(actual[key], "receipt."+key)
        canonical_rows.append(core)
    canonical_rows.sort(key=lambda r: r["slot"])
    need([r["slot"] for r in canonical_rows] == list(range(len(canonical_rows))), "EXECUTED_PREFIX_GAP", "receipt slots")
    k = integer(b["execute_from_slot"], "execute_from_slot", 0, n-1)
    need(k == len(canonical_rows), "EXECUTED_PREFIX_MISMATCH", "execute_from_slot", k, len(canonical_rows))
    need(digest(canonical_rows) == b["executed_prefix_hash"], "EXECUTED_PREFIX_HASH_MISMATCH", "executed_prefix_hash")
    if previous_version:
        same = b["plan_version"] == previous_version and b["plan_fingerprint"] == previous_fp
        need(same or b["plan_version"] > previous_version, "PLAN_VERSION_ROLLBACK", "plan_version")
        if not same:
            need(cancel == "acknowledged", "CANCEL_NOT_CONFIRMED", "cancel_status", cancel, "acknowledged")
            stopped = time_at(execution["stopped_at"])
            need(stopped <= at and (not canonical_rows or time_at(canonical_rows[-1]["measured_at"]) <= stopped),
                 "CANCEL_TIME_INVALID", "stopped_at")
            stop_slot = integer(execution["stopped_after_slot"], "stopped_after_slot", 0, n)
            need(stop_slot == k, "CANCEL_PREFIX_MISMATCH", "stopped_after_slot")
        else:
            need(cancel == "none", "PLAN_CANCELLED_OR_CANCELLING", "cancel_status")
    else:
        need(not rows and cancel == "none", "EXECUTION_IDENTITY_INVALID", "execution")
    need(at <= start+timedelta(minutes=15*k), "INTERVAL_ALREADY_STARTED", "execute_from_slot", k)
    storage = d["storage"]
    energy = storage["initial_energy_kwh"]
    for r in canonical_rows:
        t, a = r["slot"], r["actual"]
        before, after = a["storage_energy_before_kwh"], a["storage_energy_after_kwh"]
        if abs(before-energy) > EPS:
            fail("RECEIPT_ENERGY_CHAIN", field="receipt.storage_energy_before_kwh", slot=t, actual=before, expected=energy)
        pc, pd = a["storage_charge_energy_kwh"], a["storage_discharge_energy_kwh"]
        predicted = before+storage["charge_efficiency"]*pc-pd/storage["discharge_efficiency"]
        if abs(after-predicted) > EPS:
            fail("RECEIPT_STORAGE_DYNAMICS", field="receipt.storage_energy_after_kwh", slot=t, actual=after, expected=predicted)
        if pc > EPS and pd > EPS:
            fail("RECEIPT_SIMULTANEOUS", field="receipt.storage_energy", slot=t, actual=[pc, pd], expected="one direction")
        expected_grid = (a["base_energy_kwh"]+sum(a["vehicle_energy_kwh"].values())
                         +sum(a["existing_by_port_energy_kwh"].values())+pc-pd-a["pv_used_energy_kwh"])
        if abs(a["grid_import_energy_kwh"]-expected_grid) > EPS:
            fail("RECEIPT_POWER_BALANCE", field="receipt.grid_import_energy_kwh", slot=t,
                 actual=a["grid_import_energy_kwh"], expected=expected_grid)
        energy = after
    return k, canonical_rows


def verify_physical(b, state, at):
    errors = []

    def fail(code, *, field, actual=None, expected=None, **context):
        errors.append({"code": code, "field": field, "actual": actual, "expected": expected, **context})

    result = {"passed": False, "violations": errors, "rule_version": "a03-physical-v1", "metrics": None}
    try:
        d, p = identity(b)
        n, start, end, ports, vehicles = constraints_shape(d)
        measured, _ = state_shape(state, d, at, n, ports)
        issued, valid = (time_at(b[k], k) for k in ("issued_at", "valid_until"))
        need(time_at(d["as_of"]) <= issued <= at < valid <= end, "PLAN_EXPIRED_OR_FUTURE", "frozen timestamps")
        k, prefix = receipts(b, state, d, at, n, start, vehicles, ports, fail)
        dependencies(b, state, at, start+timedelta(minutes=15*k), end)
        need(measured >= start+timedelta(minutes=15*k) if k else True,
             "MEASUREMENT_BEFORE_PREFIX", "state.measured_at")
        for key, keys in (("charge_by_port_kw", ports), ("charge_by_vehicle_kw", vehicles)):
            exact_map(p[key], keys, key)
            for values in p[key].values():
                vector(values, key, n, -EPS)
        for key in ("pv_used_kw", "storage_charge_kw", "storage_discharge_kw"):
            vector(p[key], key, n, -EPS)
        vector(p["net_import_kw"], "net_import_kw", n, -1e9)
        vector(p["storage_energy_kwh"], "storage_energy_kwh", n+1, -EPS)
        need(type(p["assignments"]) is list, "ASSIGNMENT_INVALID", "assignments")
        by_port = {pid: [0.0]*n for pid in ports}
        by_vehicle, seen, occupied = {}, set(), set()
        for a in p["assignments"]:
            vid, pid = a["vehicle_id"], a["port_id"]
            need(vid in vehicles and pid in ports and vid not in seen, "ASSIGNMENT_ID_INVALID", "assignment")
            seen.add(vid)
            v, port = vehicles[vid], ports[pid]
            values = vector(a["power_kw"], "assignment.power_kw", n, -EPS)
            connected = slot_list(a["connected_slots"], "assignment.connected_slots", n)
            need(bool(connected) and connected == list(range(min(connected), max(connected)+1)),
                 "NONCONTIGUOUS_CONNECTION", vid)
            if pid not in v["allowed_port_ids"] or port["connector"] != v["connector"]:
                fail("CONNECTOR_OR_PORT_MISMATCH", field=vid, actual=pid, expected=v["allowed_port_ids"])
            for t in connected:
                if not v["arrival_slot"] <= t < v["deadline_slot"]:
                    fail("VEHICLE_WINDOW", field=vid, slot=t, actual=t, expected=[v["arrival_slot"], v["deadline_slot"]])
                if (pid, t) in occupied or t in port["occupied_slots"] or (t >= k and t in state["ports"][pid]["occupied_slots"]):
                    fail("PORT_CONFLICT", field=pid, slot=t, actual="occupied", expected="exclusive")
                occupied.add((pid, t))
            for t, value in enumerate(values):
                limit = min(v["max_power_kw"], port["max_power_kw"][t]) if t in connected else 0
                if value > limit+EPS:
                    fail("VEHICLE_POWER", field=vid, slot=t, actual=value, expected={"maximum": limit})
                if t >= k and t in connected and blocked(state["ports"][pid]):
                    fail("PORT_BLOCKED", field=pid, slot=t, actual="blocked", expected="operable")
                if abs(value-p["charge_by_vehicle_kw"][vid][t]) > EPS:
                    fail("VEHICLE_CURVE_MISMATCH", field=vid, slot=t, actual=p["charge_by_vehicle_kw"][vid][t], expected=value)
                by_port[pid][t] += value
            by_vehicle[vid] = values
        need(seen == set(vehicles), "MISSING_VEHICLE", "assignments")
        delivered, remaining = {}, {}
        for vid, v in vehicles.items():
            actual = sum(r["actual"]["vehicle_energy_kwh"][vid] for r in prefix)
            future = sum(by_vehicle[vid][k:])*0.25
            delivered[vid], remaining[vid] = actual+future, v["required_energy_kwh"]-actual
            if abs(actual+future-v["required_energy_kwh"]) > EPS:
                fail("REMAINING_COMMITMENT", field=vid, actual=actual+future, expected=v["required_energy_kwh"])
        for pid, port in ports.items():
            for t in range(n):
                total = by_port[pid][t]
                if abs(total-p["charge_by_port_kw"][pid][t]) > EPS:
                    fail("PORT_CURVE_MISMATCH", field=pid, slot=t, actual=p["charge_by_port_kw"][pid][t], expected=total)
                combined = total+d["existing_charge_by_port_kw"][pid][t]
                limit = port["max_power_kw"][t]
                if t >= k:
                    live = state["ports"][pid]
                    limit = min(limit, 0 if blocked(live) else live["max_power_kw"])
                if combined > limit+EPS:
                    fail("PORT_POWER_LIMIT", field=pid, slot=t, actual=combined, expected={"maximum": limit})
        s, m = d["storage"], state["measurement"]
        energy = p["storage_energy_kwh"]
        measured_energy = m["storage_energy_kwh"]
        if abs(m["storage_soc"]*s["capacity_kwh"]-measured_energy) > EPS:
            fail("MEASUREMENT_SOC_MISMATCH", field="storage_soc", actual=m["storage_soc"], expected=measured_energy/s["capacity_kwh"])
        if abs(energy[0]-s["initial_energy_kwh"]) > EPS:
            fail("STORAGE_INITIAL", field="storage_energy_kwh[0]", actual=energy[0], expected=s["initial_energy_kwh"])
        if abs(energy[k]-measured_energy) > EPS:
            fail("STORAGE_LIVE_INITIAL", field="storage_energy_kwh", slot=k, actual=energy[k], expected=measured_energy)
        if prefix and measured == time_at(prefix[-1]["measured_at"]) and abs(measured_energy-prefix[-1]["actual"]["storage_energy_after_kwh"]) > EPS:
            fail("MEASUREMENT_RECEIPT_CONFLICT", field="storage_energy_kwh", actual=measured_energy,
                 expected=prefix[-1]["actual"]["storage_energy_after_kwh"])
        floor, ceiling = max(s["min_soc"], s["reserve_soc"])*s["capacity_kwh"], s["max_soc"]*s["capacity_kwh"]
        if not floor-EPS <= measured_energy <= ceiling+EPS:
            fail("STORAGE_LIVE_SOC", field="measurement.storage_energy_kwh", actual=measured_energy, expected=[floor, ceiling])
        for t, value in enumerate(energy):
            if not floor-EPS <= value <= ceiling+EPS:
                fail("STORAGE_SOC_RESERVE", field="storage_energy_kwh", slot=t, actual=value, expected=[floor, ceiling])
        if abs(energy[-1]-s["terminal_energy_kwh"]) > EPS:
            fail("STORAGE_TERMINAL", field="storage_energy_kwh[-1]", actual=energy[-1], expected=s["terminal_energy_kwh"])
        # Actual historical values are immutable and never revalidated against today's derating.
        for r in prefix:
            t, a = r["slot"], r["actual"]
            expected_connections = {a["vehicle_id"]: a["port_id"] for a in p["assignments"] if t in a["connected_slots"]}
            if expected_connections != a["connected_vehicle_port_ids"]:
                fail("EXECUTED_CONNECTION_CHANGED", field="connected_vehicle_port_ids", slot=t,
                     actual=expected_connections, expected=a["connected_vehicle_port_ids"])
            pairs = [("storage_charge_kw", a["storage_charge_energy_kwh"]/0.25),
                     ("storage_discharge_kw", a["storage_discharge_energy_kwh"]/0.25),
                     ("pv_used_kw", a["pv_used_energy_kwh"]/0.25),
                     ("net_import_kw", a["grid_import_energy_kwh"]/0.25)]
            for field, actual in pairs:
                if abs(p[field][t]-actual) > EPS:
                    fail("EXECUTED_PREFIX_CHANGED", field=field, slot=t, actual=p[field][t], expected=actual)
            for boundary, actual in ((t, a["storage_energy_before_kwh"]), (t+1, a["storage_energy_after_kwh"])):
                if abs(energy[boundary]-actual) > EPS:
                    fail("EXECUTED_PREFIX_CHANGED", field="storage_energy_kwh", slot=boundary, actual=energy[boundary], expected=actual)
            for vid in vehicles:
                actual = a["vehicle_energy_kwh"][vid]/0.25
                if abs(by_vehicle[vid][t]-actual) > EPS:
                    fail("EXECUTED_PREFIX_CHANGED", field=vid, slot=t, actual=by_vehicle[vid][t], expected=actual)
            for pid in ports:
                actual = a["existing_by_port_energy_kwh"][pid]/0.25
                if abs(d["existing_charge_by_port_kw"][pid][t]-actual) > EPS:
                    fail("EXISTING_PREFIX_CHANGED", field=pid, slot=t, actual=d["existing_charge_by_port_kw"][pid][t], expected=actual)
        new_charge = [sum(values[t] for values in by_port.values()) for t in range(n)]
        existing = [sum(values[t] for values in d["existing_charge_by_port_kw"].values()) for t in range(n)]
        reconstructed_grid, reconstructed_energy = [], [measured_energy]
        for t in range(k, n):
            pc, pd, pv, grid = (p[key][t] for key in ("storage_charge_kw", "storage_discharge_kw", "pv_used_kw", "net_import_kw"))
            live_storage = state["storage"]
            limits = [("STORAGE_CHARGE_POWER", pc, min(s["max_charge_kw"][t], 0 if blocked(live_storage) else live_storage["max_charge_kw"])),
                      ("STORAGE_DISCHARGE_POWER", pd, min(s["max_discharge_kw"][t], 0 if blocked(live_storage) else live_storage["max_discharge_kw"])),
                      ("PV_LIMIT", pv, min(d["pv_available_kw"][t], m["pv_available_kw"], state["pv_max_power_kw"]))]
            for code, actual, limit in limits:
                if actual < -EPS or actual > limit+EPS:
                    fail(code, field=code.lower(), slot=t, actual=actual, expected={"minimum": 0, "maximum": limit})
            if pc > EPS and pd > EPS:
                fail("STORAGE_SIMULTANEOUS", field="storage power", slot=t, actual=[pc, pd], expected="one direction")
            next_energy = reconstructed_energy[-1]+s["charge_efficiency"]*pc*0.25-pd*0.25/s["discharge_efficiency"]
            reconstructed_energy.append(next_energy)
            if abs(energy[t+1]-next_energy) > EPS:
                fail("STORAGE_DYNAMICS", field="storage_energy_kwh", slot=t+1, actual=energy[t+1], expected=next_energy)
            if not floor-EPS <= next_energy <= ceiling+EPS:
                fail("STORAGE_RECOMPUTED_SOC", field="recomputed_storage_energy_kwh", slot=t+1, actual=next_energy, expected=[floor, ceiling])
            expected_grid = d["base_load_kw"][t]+existing[t]+new_charge[t]+pc-pd-pv
            reconstructed_grid.append(expected_grid)
            if abs(grid-expected_grid) > EPS:
                fail("POWER_BALANCE", field="net_import_kw", slot=t, actual=grid, expected=expected_grid)
            limit = min(d["grid_import_limit_kw"][t], state["grid_import_limit_kw"])
            # Check calculated flow too: a forged reported grid cannot conceal export/overload.
            for field, value in (("net_import_kw", grid), ("recomputed_net_import_kw", expected_grid)):
                if value < -EPS or value > limit+EPS:
                    fail("GRID_IMPORT_EXPORT", field=field, slot=t, actual=value, expected=[0, limit])
            for budget in (d["new_charge_limit_kw"], state["new_charge_limit_kw"]):
                if budget is not None and new_charge[t] > budget[t]+EPS:
                    fail("NEW_CHARGE_BUDGET", field="new_charge_kw", slot=t, actual=new_charge[t], expected={"maximum": budget[t]})
        head_grid = m["base_load_kw"]+existing[k]+new_charge[k]+p["storage_charge_kw"][k]-p["storage_discharge_kw"][k]-p["pv_used_kw"][k]
        limit = min(d["grid_import_limit_kw"][k], state["grid_import_limit_kw"])
        if head_grid < -EPS or head_grid > limit+EPS:
            fail("LIVE_GRID_IMPORT_EXPORT", field="live_head_grid_kw", slot=k, actual=head_grid, expected=[0, limit])
        if abs(head_grid-p["net_import_kw"][k]) > EPS:
            fail("LIVE_POWER_BALANCE", field="net_import_kw", slot=k, actual=p["net_import_kw"][k], expected=head_grid)
        if m["net_import_kw"] < -EPS or m["net_import_kw"] > state["grid_import_limit_kw"]+EPS:
            fail("MEASURED_GRID_IMPORT_EXPORT", field="measurement.net_import_kw", actual=m["net_import_kw"], expected=[0, state["grid_import_limit_kw"]])
        result["metrics"] = {"execute_from_slot": k, "interval_count": n,
            "reconstructed_charge_by_port_kw": by_port, "reconstructed_new_charge_kw": new_charge,
            "reconstructed_remaining_grid_kw": reconstructed_grid, "reconstructed_remaining_energy_kwh": reconstructed_energy,
            "delivered_plus_planned_by_vehicle_kwh": delivered, "remaining_commitment_by_vehicle_kwh": remaining,
            "live_head_grid_kw": head_grid, "pv_cap_rule": "min_frozen_forecast_latest_snapshot_and_rating_over_suffix"}
    except ReviewError as e:
        fail(e.code, field=e.field, actual=e.actual, expected=e.expected)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, ArithmeticError) as e:
        fail("INPUT_MISSING_OR_MALFORMED", field="physical input", actual=type(e).__name__, expected="complete review contract")
    result["passed"] = not errors
    return result
