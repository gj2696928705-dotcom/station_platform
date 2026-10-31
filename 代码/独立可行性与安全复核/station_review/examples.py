"""Explicit synthetic fixtures, never defaults for missing station measurements."""
from datetime import timedelta

from .contract import RULE, STATE_SCHEMA, content_hash, digest, freeze_plan, time_at


def resign(plan):
    plan["fingerprint"] = content_hash(plan)
    return plan


def fixture():
    """Hand oracle: 1 kWh vehicle; battery 5 -> 5.5 -> 5.5 -> 5 -> 5."""
    at, end = "2025-06-01T10:00:00+08:00", "2025-06-01T11:00:00+08:00"
    refs = {key: digest({"synthetic_fixture": key, "version": 1}) for key in ("load", "pv", "commitments")}
    d = {"schema_version": "a02-joint-v1", "simulation": True, "execution_mode": "offline_replay",
         "task_id": "A03-HAND", "task_version": 1, "station_id": "synthetic-test-station",
         "meter_boundary": "station_ac", "as_of": at, "window_start": at, "window_end": end,
         "interval_minutes": 15, "source_refs": refs, "assumptions": [], "commitment_status": "complete",
         "base_load_kw": [4]*4, "pv_available_kw": [2]*4, "grid_import_limit_kw": [20]*4,
         "grid_export_limit_kw": 0, "rates_cny_per_kwh": [1]*4, "new_charge_limit_kw": [8]*4,
         "existing_charge_by_port_kw": {pid: [0]*4 for pid in ("P1", "P2")},
         "ports": [{"port_id": pid, "connector": "DC", "max_power_kw": [4]*4, "occupied_slots": []}
                   for pid in ("P1", "P2")],
         "vehicles": [{"vehicle_id": "V1", "connector": "DC", "allowed_port_ids": ["P1", "P2"],
                       "arrival_slot": 0, "deadline_slot": 4, "required_energy_kwh": 1, "max_power_kw": 4}],
         "storage": {"storage_id": "B1", "capacity_kwh": 10, "initial_energy_kwh": 5,
                     "min_soc": 0.1, "max_soc": 0.9, "reserve_soc": 0.3,
                     "charge_efficiency": 0.5, "discharge_efficiency": 1,
                     "max_charge_kw": [4]*4, "max_discharge_kw": [4]*4,
                     "terminal_energy_kwh": 5, "terminal_rule": "hard_equality_at_window_end"},
         "objective": {"throughput_cost_cny_per_kwh": 0, "curtailment_cost_cny_per_kwh": 0}}
    p = resign({"schema_version": "a02-joint-v1", "simulation": True, "input_hash": digest(d),
                "assignments": [{"vehicle_id": "V1", "port_id": "P1", "power_kw": [0, 4, 0, 0], "connected_slots": [1]}],
                "charge_by_vehicle_kw": {"V1": [0, 4, 0, 0]},
                "charge_by_port_kw": {"P1": [0, 4, 0, 0], "P2": [0]*4},
                "pv_used_kw": [1]*4, "storage_charge_kw": [4, 0, 0, 0],
                "storage_discharge_kw": [0, 0, 2, 0], "storage_energy_kwh": [5, 5.5, 5.5, 5, 5],
                "net_import_kw": [7, 7, 1, 3]})
    deps = [{"ref": key, "sha256": value, "status": "usable", "native_grid": "15min_interval_average",
             "kind": {"load": "forecast_load", "pv": "forecast_pv", "commitments": "commitments"}[key],
             "issued_at": at, "available_at": at, "valid_until": end, "window_start": at, "window_end": end}
            for key, value in refs.items()]
    b = freeze_plan(d, p, producer="A02", producer_version="a02-joint-v1", publisher="O1-SIM",
                    plan_version=1, dependencies=deps, issued_at=at, valid_until=end)
    s = synthetic_state(d, at=at, pv_cap=2, initial_energy=5)
    policy = synthetic_policy(b, s, at)
    return b, s, policy


def synthetic_state(d, *, at, pv_cap, initial_energy, measurement_base=None):
    """ALL readings/flags here are assumptions for a named offline demonstration."""
    flags = {"online": True, "fault": False, "protection_active": False, "manual_takeover": False, "frozen": False}
    return {"schema_version": STATE_SCHEMA, "state_id": "explicit-synthetic-snapshot", "revision": 1,
            "station_id": d["station_id"], "task_id": d["task_id"], "task_version": d["task_version"],
            "meter_boundary": "station_ac", "measured_at": at, "available_at": at,
            "valid_until": (time_at(at)+timedelta(seconds=60)).isoformat(), "quality": "good",
            "power_semantics": "ac_dispatch_snapshot", "is_simulated": True,
            "source_note": "Explicit A03 fixture assumptions; not D01/D02 independent realtime telemetry.",
            "online": True, "protection_active": False, "manual_takeover": False, "frozen": False,
            "measurement": {"quality": "good", "base_load_kw": d["base_load_kw"][0] if measurement_base is None else measurement_base,
                            "pv_available_kw": pv_cap, "net_import_kw": 0, "storage_energy_kwh": initial_energy,
                            "storage_soc": initial_energy/d["storage"]["capacity_kwh"]},
            "grid_import_limit_kw": max(d["grid_import_limit_kw"]), "pv_max_power_kw": pv_cap,
            "new_charge_limit_kw": d["new_charge_limit_kw"],
            "dependency_hashes": {key: value for key, value in d["source_refs"].items()},
            "ports": {p["port_id"]: {"asset_id": p["port_id"], "connector": p["connector"],
                                     "max_power_kw": max(p["max_power_kw"]), "occupied_slots": [], **flags}
                      for p in d["ports"]},
            "storage": {"storage_id": d["storage"]["storage_id"], "max_charge_kw": max(d["storage"]["max_charge_kw"]),
                        "max_discharge_kw": max(d["storage"]["max_discharge_kw"]), **flags},
            "execution": {"accepted_plan_version": 0, "accepted_plan_fingerprint": None,
                          "accepted_plan_history": [],
                          "cancel_status": "none", "stopped_at": None, "stopped_after_slot": None, "receipts": []}}


def synthetic_policy(b, state, at, *, allow_assumptions=False):
    """Explicit fixture approval. Production callers must supply their own trusted policy."""
    b = b.to_dict() if hasattr(b, "to_dict") else b
    d = b["constraints"]
    return {"rule_version": RULE, "policy_id": "synthetic-offline-approval-v1", "simulation_only": True,
            "authorized_publishers": [b["publisher"]], "allowed_producers": {b["producer"]: [b["producer_version"]]},
            "allowed_sources": dict(d["source_refs"]), "allow_assumptions": allow_assumptions,
            "approval": {"frozen_fingerprint": b["fingerprint"], "plan_fingerprint": b["plan_fingerprint"],
                         "state_hash": digest(state), "state_id": state["state_id"], "state_revision": state["revision"],
                         "task_id": d["task_id"], "task_version": d["task_version"], "publisher": b["publisher"],
                         "simulation_only": True, "released": True, "approved_at": at,
                         "valid_until": state["valid_until"]}}


def partial_fixture():
    """One physically completed step, acknowledged stop, and suffix proposal v2."""
    frozen, state, _ = fixture()
    old = frozen.to_dict()
    at = "2025-06-01T10:15:00+08:00"
    receipt = {"event_id": "receipt-0", "station_id": state["station_id"], "task_id": state["task_id"], "task_version": 1,
               "plan_version": 1, "plan_fingerprint": old["plan_fingerprint"], "slot": 0,
               "measured_at": at, "available_at": at, "quality": "good",
               "actual": {"vehicle_energy_kwh": {"V1": 0}, "connected_vehicle_port_ids": {},
                          "existing_by_port_energy_kwh": {"P1": 0, "P2": 0},
                          "base_energy_kwh": 1, "pv_used_energy_kwh": 0.25, "storage_charge_energy_kwh": 1,
                          "storage_discharge_energy_kwh": 0, "storage_energy_before_kwh": 5,
                          "storage_energy_after_kwh": 5.5, "grid_import_energy_kwh": 1.75}}
    state.update(measured_at=at, available_at=at, valid_until="2025-06-01T10:16:00+08:00")
    state["measurement"].update(storage_energy_kwh=5.5, storage_soc=0.55)
    state["execution"].update(accepted_plan_version=1, accepted_plan_fingerprint=old["plan_fingerprint"],
                              accepted_plan_history=[{"plan_version": 1, "plan_fingerprint": old["plan_fingerprint"],
                                                      "execute_from_slot": 0, "stop_after_slot": 1}],
                              cancel_status="acknowledged", stopped_at=at, stopped_after_slot=1, receipts=[receipt])
    new = freeze_plan(old["constraints"], old["plan"], producer="A02", producer_version="a02-joint-v1",
                      publisher="O1-SIM", plan_version=2, dependencies=old["dependencies"], issued_at=at,
                      valid_until=old["valid_until"], execute_from_slot=1, executed_prefix=[receipt])
    return new, state, old
