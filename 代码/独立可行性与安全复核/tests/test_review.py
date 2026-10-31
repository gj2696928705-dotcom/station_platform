"""Hand oracles and adversarial review of independently supplied JSON artifacts."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from datetime import timedelta
import json
from pathlib import Path

import pytest

from station_review import FrozenPlan, digest, freeze_plan, review_plan
from station_review.contract import content_hash, time_at
from station_review.examples import fixture, partial_fixture, resign, synthetic_policy, synthetic_state


def seal(b, *, bind_input=True):
    if bind_input:
        b["constraints_hash"] = digest(b["constraints"])
        b["plan"]["input_hash"] = b["constraints_hash"]
    resign(b["plan"])
    b["plan_fingerprint"] = b["plan"]["fingerprint"]
    b["fingerprint"] = content_hash(b)
    return b


def codes(result):
    return {v["code"] for v in result["violations"]}


def run(b, s, *, gov=True, at=None):
    b = b.to_dict() if isinstance(b, FrozenPlan) else b
    at = at or s["available_at"]
    p = synthetic_policy(b, s, at)
    p["approval"]["released"] = gov
    return review_plan(b, s, p, as_of=at)


def partial():
    b, s, old = partial_fixture()
    return b.to_dict(), s, old


def test_hand_oracle_deterministic_and_independent():
    b, s, policy = fixture()
    first = review_plan(b, s, policy, as_of=s["available_at"])
    assert first == review_plan(b, s, policy, as_of=s["available_at"])
    assert first["passed"] and first["status"] == "SIMULATION_ALLOWED"
    m = first["physical"]["metrics"]
    # Hand calculation at AC boundary: 4 + vehicle + charge - discharge - 1.
    assert m["reconstructed_remaining_grid_kw"] == [7, 7, 1, 3]
    assert m["reconstructed_remaining_energy_kwh"] == [5, 5.5, 5.5, 5, 5]
    assert m["delivered_plus_planned_by_vehicle_kwh"] == {"V1": 1}
    assert sum(m["reconstructed_remaining_grid_kw"])*0.25 == 4.5
    assert not first["device_dispatch_authorized"] and not first["certified_safety_service"]
    assert first["physical"]["state_hash"] == first["governance"]["state_hash"] == digest(s)
    assert first["physical"]["frozen_fingerprint"] == first["governance"]["frozen_fingerprint"] == b.to_dict()["fingerprint"]


def test_freeze_owns_immutable_bytes():
    b, s, p = fixture()
    original = b.payload
    x = b.to_dict()
    x["plan"]["pv_used_kw"][0] = 999
    assert b.payload == original
    with pytest.raises(FrozenInstanceError):
        b.payload = b"{}"
    assert review_plan(b, s, p, as_of=s["available_at"])["passed"]


@pytest.mark.parametrize("physical,governance", [(True, True), (True, False), (False, True), (False, False)])
def test_only_double_pass_releases(physical, governance):
    b, s, _ = fixture()
    b = b.to_dict()
    if not physical:
        b["plan"]["net_import_kw"][0] = 999
        seal(b)
    r = run(b, s, gov=governance)
    assert r["physical"]["passed"] is physical
    assert r["governance"]["passed"] is governance
    assert r["eligible_for_offline_simulation"] is (physical and governance)


@pytest.mark.parametrize("key", ["pv_used_kw", "storage_charge_kw", "storage_discharge_kw", "storage_energy_kwh", "net_import_kw"])
def test_unsigned_plan_tamper_is_blocked(key):
    b, s, _ = fixture()
    b = b.to_dict()
    b["plan"][key][0] += 1
    assert "FROZEN_HASH_MISMATCH" in codes(run(b, s))


@pytest.mark.parametrize("change,expected", [
    (lambda b: b["plan"]["net_import_kw"].__setitem__(0, 0), "POWER_BALANCE"),
    (lambda b: b["plan"]["storage_energy_kwh"].__setitem__(1, 5), "STORAGE_DYNAMICS"),
    (lambda b: b["plan"]["storage_charge_kw"].__setitem__(2, 1), "STORAGE_SIMULTANEOUS"),
    (lambda b: b["plan"]["pv_used_kw"].__setitem__(1, 5), "PV_LIMIT"),
    (lambda b: b["plan"]["charge_by_port_kw"]["P1"].__setitem__(1, 0), "PORT_CURVE_MISMATCH"),
    (lambda b: b["plan"]["charge_by_vehicle_kw"]["V1"].__setitem__(1, 0), "VEHICLE_CURVE_MISMATCH"),
    (lambda b: b["plan"]["assignments"][0].update(connected_slots=[0, 2]), "NONCONTIGUOUS_CONNECTION"),
    (lambda b: b["plan"]["assignments"][0].update(port_id="unknown"), "ASSIGNMENT_ID_INVALID"),
    (lambda b: b["plan"]["assignments"].clear(), "MISSING_VEHICLE"),
    (lambda b: b["constraints"]["vehicles"][0].update(required_energy_kwh=2), "REMAINING_COMMITMENT"),
    (lambda b: b["constraints"]["vehicles"][0].update(deadline_slot=1), "VEHICLE_WINDOW"),
    (lambda b: b["constraints"]["vehicles"][0].update(connector="AC"), "CONNECTOR_OR_PORT_MISMATCH"),
    (lambda b: b["constraints"]["ports"][0].update(occupied_slots=[1]), "PORT_CONFLICT"),
    (lambda b: b["constraints"]["storage"].update(terminal_energy_kwh=6), "STORAGE_TERMINAL"),
    (lambda b: b["constraints"]["storage"].update(reserve_soc=0.6), "STORAGE_LIVE_SOC"),
    (lambda b: b["constraints"].update(new_charge_limit_kw=[0]*4), "NEW_CHARGE_BUDGET"),
    (lambda b: b["constraints"].update(commitment_status="unknown"), "COMMITMENT_UNKNOWN"),
])
def test_resigned_physical_counterexamples(change, expected):
    b, s, _ = fixture()
    b = b.to_dict()
    change(b)
    seal(b)
    r = run(b, s)
    assert not r["passed"] and expected in codes(r)


@pytest.mark.parametrize("change,expected", [
    (lambda s: s["measurement"].update(pv_available_kw=0.5), "PV_LIMIT"),
    (lambda s: s["measurement"].update(storage_energy_kwh=2, storage_soc=0.2), "STORAGE_LIVE_SOC"),
    (lambda s: s["measurement"].update(storage_energy_kwh=4, storage_soc=0.4), "STORAGE_LIVE_INITIAL"),
    (lambda s: s["measurement"].update(storage_soc=0.9), "MEASUREMENT_SOC_MISMATCH"),
    (lambda s: s.update(grid_import_limit_kw=6), "GRID_IMPORT_EXPORT"),
    (lambda s: s["ports"]["P1"].update(fault=True), "PORT_BLOCKED"),
    (lambda s: s["ports"]["P1"].update(max_power_kw=2), "PORT_POWER_LIMIT"),
    (lambda s: s["ports"]["P1"].update(occupied_slots=[1]), "PORT_CONFLICT"),
    (lambda s: s["storage"].update(manual_takeover=True), "STORAGE_CHARGE_POWER"),
    (lambda s: s.update(protection_active=True), "STATION_PROTECTION_ACTIVE"),
    (lambda s: s.update(manual_takeover=True), "STATION_MANUAL_TAKEOVER"),
    (lambda s: s.update(frozen=True), "STATION_FROZEN"),
    (lambda s: s.update(online=False), "STATION_OFFLINE"),
    (lambda s: s.update(quality="unknown"), "MEASUREMENT_QUALITY"),
    (lambda s: s["measurement"].update(base_load_kw=9), "LIVE_POWER_BALANCE"),
    (lambda s: s["measurement"].update(base_load_kw=0), "LIVE_POWER_BALANCE"),
    (lambda s: s["measurement"].update(net_import_kw=-1), "MEASURED_GRID_IMPORT_EXPORT"),
    (lambda s: s.update(new_charge_limit_kw=[1]*4), "NEW_CHARGE_BUDGET"),
    (lambda s: s.update(task_version=2), "STATE_CONTEXT_MISMATCH"),
    (lambda s: s.update(station_id="other"), "STATE_CONTEXT_MISMATCH"),
    (lambda s: s["dependency_hashes"].update(pv="f"*64), "DEPENDENCY_CHANGED"),
    (lambda s: s["ports"]["P1"].update(connector="AC"), "DEVICE_MAPPING_CHANGED"),
    (lambda s: s["storage"].update(storage_id="other"), "DEVICE_MAPPING_CHANGED"),
    (lambda s: s["ports"].pop("P2"), "MAPPING_INVALID"),
])
def test_latest_state_rechecks_with_new_approval(change, expected):
    b, s, _ = fixture()
    change(s)
    r = run(b, s)
    assert not r["passed"] and expected in codes(r)


@pytest.mark.parametrize("change,expected", [
    (lambda b, s: s.update(valid_until=s["available_at"]), "STATE_EXPIRED_OR_FUTURE"),
    (lambda b, s: s.update(measured_at="2025-06-01T09:58:00+08:00"), "STATE_TOO_OLD"),
    (lambda b, s: s.update(measured_at="2025-06-01T10:00:01+08:00"), "STATE_EXPIRED_OR_FUTURE"),
    (lambda b, s: s.update(measured_at="2025-06-01T10:00:00"), "TIMEZONE_INVALID"),
    (lambda b, s: s.update(measured_at="2025-06-01T02:00:00+00:00"), "TIMEZONE_INVALID"),
    (lambda b, s: b.update(valid_until=b["issued_at"]), "PLAN_EXPIRED_OR_FUTURE"),
    (lambda b, s: b["dependencies"][0].update(valid_until=b["issued_at"]), "DEPENDENCY_EXPIRED_OR_FUTURE"),
    (lambda b, s: b["dependencies"][0].update(window_end="2025-06-01T10:45:00+08:00"), "FORECAST_COVERAGE_GAP"),
    (lambda b, s: b["dependencies"][0].update(status="unknown"), "DEPENDENCY_UNAVAILABLE"),
    (lambda b, s: b["dependencies"][0].update(native_grid="instantaneous_sample"), "FORECAST_GRID_INVALID"),
    (lambda b, s: b["dependencies"].pop(), "DEPENDENCY_MISSING"),
])
def test_clock_freshness_and_dependency_boundaries(change, expected):
    b, s, _ = fixture()
    b = b.to_dict()
    change(b, s)
    seal(b)
    assert expected in codes(run(b, s))


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -float("inf"), "4", True])
def test_bad_measurements_never_default_to_zero(value):
    b, s, _ = fixture()
    s["measurement"]["base_load_kw"] = value
    r = review_plan(b, s, {}, as_of=s["available_at"])
    assert not r["passed"] and r["violations"]
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize("value", [[], [1]*3, [1]*5, None, [1, 1, 1, True]])
def test_wrong_curve_shape(value):
    b, s, _ = fixture()
    b = b.to_dict()
    b["plan"]["pv_used_kw"] = value
    seal(b)
    assert not run(b, s)["passed"]


@pytest.mark.parametrize("field", ["measurement", "execution", "storage", "ports", "measured_at", "dependency_hashes"])
def test_missing_live_fields_fail_closed(field):
    b, s, _ = fixture()
    s.pop(field)
    r = review_plan(b, s, {}, as_of=b.to_dict()["issued_at"])
    assert not r["passed"] and r["violations"]


@pytest.mark.parametrize("change,expected", [
    (lambda p: p.update(rule_version="unknown"), "GOVERNANCE_RULE_UNSUPPORTED"),
    (lambda p: p.update(simulation_only=False), "SIMULATION_MODE_REQUIRED"),
    (lambda p: p.update(authorized_publishers=["other"]), "PUBLISHER_UNAUTHORIZED"),
    (lambda p: p.update(allowed_producers={}), "PRODUCER_UNAUTHORIZED"),
    (lambda p: p["allowed_sources"].update(pv="f"*64), "SOURCE_UNAPPROVED"),
    (lambda p: p["approval"].update(plan_fingerprint="f"*64), "APPROVAL_BINDING_MISMATCH"),
    (lambda p: p["approval"].update(state_hash="f"*64), "APPROVAL_BINDING_MISMATCH"),
    (lambda p: p["approval"].update(released="true"), "SIMULATION_RELEASE_MISSING"),
    (lambda p: p.pop("approval"), "GOVERNANCE_INPUT_INVALID"),
])
def test_versioned_governance(change, expected):
    b, s, p = fixture()
    change(p)
    r = review_plan(b, s, p, as_of=s["available_at"])
    assert r["physical"]["passed"] and expected in codes(r) and not r["passed"]


def test_assumptions_and_real_mode_cannot_be_silently_approved():
    b, s, _ = fixture()
    b = b.to_dict()
    b["constraints"]["assumptions"] = ["synthetic service curve"]
    seal(b)
    assert "ASSUMPTIONS_NOT_ACCEPTED" in codes(run(b, s))
    p = synthetic_policy(b, s, s["available_at"], allow_assumptions=True)
    assert review_plan(b, s, p, as_of=s["available_at"])["passed"]
    b["constraints"]["execution_mode"] = "live"
    seal(b)
    assert "SIMULATION_MODE_REQUIRED" in codes(run(b, s))


def test_new_state_invalidates_old_approval_and_previous_review():
    b, s, p = fixture()
    old_hash = digest(s)
    assert review_plan(b, s, p, as_of=s["available_at"])["passed"]
    s["revision"] += 1
    r = review_plan(b, s, p, as_of=s["available_at"], expected_state_hash=old_hash)
    assert {"STATE_CHANGED_SINCE_REVIEW", "APPROVAL_BINDING_MISMATCH"} <= codes(r)
    assert not r["passed"]
    assert run(b, s)["passed"]


def test_storage_soc_recomputed_from_measurement_not_reported_energy_chain():
    b, s, _ = fixture()
    s["measurement"].update(storage_energy_kwh=3, storage_soc=0.3)
    r = run(b, s)
    assert "STORAGE_LIVE_INITIAL" in codes(r)
    assert r["physical"]["metrics"]["reconstructed_remaining_energy_kwh"] == [3, 3.5, 3.5, 3, 3]


def test_forged_grid_cannot_hide_export():
    b, s, _ = fixture()
    b = b.to_dict()
    b["plan"]["pv_used_kw"][3] = 5
    b["constraints"]["pv_available_kw"] = [6]*4
    s["measurement"]["pv_available_kw"] = s["pv_max_power_kw"] = 6
    b["plan"]["net_import_kw"][3] = 0
    seal(b)
    assert {"GRID_IMPORT_EXPORT", "POWER_BALANCE"} <= codes(run(b, s))


def test_self_reported_feasibility_is_ignored():
    b, s, _ = fixture()
    b = b.to_dict()
    b["plan"].update(feasible=True, verification={"passed": True})
    b["plan"]["net_import_kw"][0] = 100
    seal(b)
    assert not run(b, s)["passed"]


def test_partial_execution_uses_actual_prefix_and_remaining_service():
    b, s, _ = partial()
    r = run(b, s)
    assert r["passed"], r["violations"]
    m = r["physical"]["metrics"]
    assert m["execute_from_slot"] == 1
    assert m["reconstructed_remaining_energy_kwh"] == [5.5, 5.5, 5, 5]
    assert m["remaining_commitment_by_vehicle_kwh"] == {"V1": 1}


def test_old_full_plan_resend_after_partial_is_blocked():
    _, s, old = partial()
    assert "EXECUTED_PREFIX_MISMATCH" in codes(run(old, s))


@pytest.mark.parametrize("change,expected", [
    (lambda b, s: s["execution"].update(cancel_status="requested"), "EXECUTION_HISTORY_INVALID"),
    (lambda b, s: s["execution"].update(stopped_after_slot=0), "EXECUTION_HISTORY_INVALID"),
    (lambda b, s: s["execution"]["receipts"][0].update(plan_version=2), "RECEIPT_OLD_VERSION"),
    (lambda b, s: s["execution"]["receipts"][0].update(slot=1), "RECEIPT_OLD_VERSION"),
    (lambda b, s: s["execution"]["receipts"][0].update(quality="unknown"), "RECEIPT_QUALITY"),
    (lambda b, s: b.update(executed_prefix_hash="f"*64), "EXECUTED_PREFIX_HASH_MISMATCH"),
    (lambda b, s: b["plan"]["storage_charge_kw"].__setitem__(0, 0), "EXECUTED_PREFIX_CHANGED"),
    (lambda b, s: b.update(plan_version=1), "PLAN_CANCELLED_OR_CANCELLING"),
])
def test_partial_rejection_cases(change, expected):
    b, s, _ = partial()
    change(b, s)
    seal(b)
    r = run(b, s)
    assert not r["passed"] and expected in codes(r)


def test_receipt_repeats_are_idempotent_conflicts_rejected():
    b, s, _ = partial()
    r = deepcopy(s["execution"]["receipts"][0])
    r["received_at"] = r["available_at"]
    s["execution"]["receipts"].append(r)
    assert run(b, s)["passed"]
    r["actual"]["grid_import_energy_kwh"] += 1
    assert "RECEIPT_ID_CONFLICT" in codes(run(b, s))


def test_bad_actual_arithmetic_not_saved_by_matching_prefix_hash():
    b, s, _ = partial()
    s["execution"]["receipts"][0]["actual"]["grid_import_energy_kwh"] += 1
    b["executed_prefix_hash"] = digest(s["execution"]["receipts"])
    seal(b)
    assert "RECEIPT_POWER_BALANCE" in codes(run(b, s))


def test_next_interval_already_started_is_blocked():
    b, s, _ = partial()
    assert "INTERVAL_ALREADY_STARTED" in codes(run(b, s, at="2025-06-01T10:15:01+08:00"))


def test_no_mutation_on_rejected_review():
    b, s, p = fixture()
    original = deepcopy((b.to_dict(), s, p))
    p["approval"]["released"] = False
    saved = deepcopy(p)
    assert not review_plan(b, s, p, as_of=s["available_at"])["passed"]
    assert b.to_dict() == original[0] and s == original[1] and p == saved


def test_native_horizon_not_hardcoded_to_four():
    b, s, _ = fixture()
    b = b.to_dict()
    d, p = b["constraints"], b["plan"]
    d["schema_version"], p["schema_version"] = "a03-constraints-v1", "a03-plan-v1"
    end = "2025-06-01T12:00:00+08:00"
    d["window_end"] = b["valid_until"] = end
    for dep in b["dependencies"]:
        dep["window_end"] = dep["valid_until"] = end
    for key in ("base_load_kw", "pv_available_kw", "grid_import_limit_kw", "new_charge_limit_kw", "rates_cny_per_kwh"):
        d[key] += [d[key][-1]]*4
    for values in d["existing_charge_by_port_kw"].values():
        values += [0]*4
    for port in d["ports"]:
        port["max_power_kw"] += [4]*4
    for key in ("max_charge_kw", "max_discharge_kw"):
        d["storage"][key] += [4]*4
    for key in ("pv_used_kw", "storage_charge_kw", "storage_discharge_kw", "storage_energy_kwh", "net_import_kw"):
        p[key] += [p[key][-1]]*4
    for key in ("charge_by_port_kw", "charge_by_vehicle_kw"):
        for values in p[key].values():
            values += [0]*4
    p["assignments"][0]["power_kw"] += [0]*4
    s["new_charge_limit_kw"] += [8]*4
    seal(b)
    assert run(b, s)["passed"]


def test_saved_a02_plan_compatibility_without_importing_optimizer():
    path = Path(__file__).resolve().parents[2]/"场站光储充联合优化"/"开发验证"/"hand_result.json"
    x = json.loads(path.read_text(encoding="utf-8"))
    d, p = x["input"], x["joint"]["plan"]
    key, value = next(iter(d["source_refs"].items()))
    # A02 fixture has one source manifest containing all three frozen kinds.
    # One immutable manifest can cover three kinds; do not alter optimizer inputs
    # to manufacture extra reference aliases or silently change its input hash.
    deps = [{"ref": key, "sha256": value, "kind": kind, "status": "usable", "native_grid": "15min_interval_average",
             "issued_at": d["as_of"], "available_at": d["as_of"],
             "valid_until": d["window_end"], "window_start": d["window_start"], "window_end": d["window_end"]}
            for kind in ("forecast_load", "forecast_pv", "commitments")]
    b = freeze_plan(d, p, producer="A02", producer_version="a02-joint-v1", publisher="O1-SIM", plan_version=1,
                    dependencies=deps, issued_at=d["as_of"], valid_until=d["window_end"])
    s = synthetic_state(d, at=d["as_of"], pv_cap=0, initial_energy=5)
    assert run(b, s)["passed"]


@pytest.mark.parametrize("value", [None, [], {"fingerprint": float("nan")}, 3])
def test_malformed_top_level_returns_rejection(value):
    r = review_plan(value, {}, {}, as_of="2025-06-01T10:00:00+08:00")
    assert not r["passed"] and r["violations"]
    json.dumps(r, allow_nan=False)
