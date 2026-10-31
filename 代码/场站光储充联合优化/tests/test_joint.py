"""A02 numerical and adversarial acceptance; independent hand/exhaustive oracles."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path

import pytest

from station_joint import InputError, freeze, solve, verify_plan, simulation_gate
from station_joint.contract import fingerprint
from station_joint.replay import LinkedReplay, compare_idle
from station_joint.solver import classify_status


DATA = Path(__file__).resolve().parents[2] / "数据"


def sample():
    return {"schema_version": "a02-joint-v1", "simulation": True, "execution_mode": "offline_replay",
        "task_id": "hand-example", "task_version": 1, "station_id": "test-station", "meter_boundary": "station_ac",
        "as_of": "2025-06-01T10:00:00+08:00", "window_start": "2025-06-01T10:00:00+08:00",
        "window_end": "2025-06-01T11:00:00+08:00", "interval_minutes": 15,
        "source_refs": {"synthetic_test": fingerprint({"fixture_version": 1})}, "assumptions": [],
        "commitment_status": "complete", "base_load_kw": [4]*4, "pv_available_kw": [0]*4,
        "existing_charge_by_port_kw": {"P1": [0]*4, "P2": [0]*4},
        "grid_import_limit_kw": [20]*4, "grid_export_limit_kw": 0, "rates_cny_per_kwh": [1, 1, 3, 3],
        "new_charge_limit_kw": None,
        "ports": [{"port_id": p, "connector": "DC", "max_power_kw": [4]*4, "occupied_slots": []} for p in ("P1", "P2")],
        "vehicles": [{"vehicle_id": "V1", "connector": "DC", "allowed_port_ids": ["P1", "P2"], "arrival_slot": 0,
                      "deadline_slot": 4, "required_energy_kwh": 1, "max_power_kw": 4}],
        "storage": {"storage_id": "B1", "capacity_kwh": 10, "initial_energy_kwh": 5,
                    "min_soc": 0.1, "max_soc": 0.9, "reserve_soc": 0.3, "max_charge_kw": [4]*4,
                    "max_discharge_kw": [4]*4, "charge_efficiency": 1, "discharge_efficiency": 1,
                    "terminal_rule": "hard_equality_at_window_end", "terminal_energy_kwh": 5},
        "objective": {"throughput_cost_cny_per_kwh": 0, "curtailment_cost_cny_per_kwh": 0}}


def resign(plan):
    plan["fingerprint"] = fingerprint({k: v for k, v in plan.items() if k != "fingerprint"})
    return plan


def codes(check):
    return {v["code"] for v in check["violations"]}


def test_hand_calculation_and_idle_same_terminal():
    d = sample()
    r = compare_idle(d)
    # Base import = 8 CNY; vehicle = 1 CNY; move 2 kWh from price 3 to 1 => save 4.
    assert r["joint"]["status"] == r["idle"]["status"] == "OPTIMAL"
    assert r["idle"]["objective"]["total_objective_cny"] == pytest.approx(9)
    assert r["joint"]["objective"]["total_objective_cny"] == pytest.approx(5)
    assert r["joint"]["solver"]["best_bound_cny"] == pytest.approx(5)
    assert r["joint"]["solver"]["absolute_gap_cny"] <= 1e-6
    assert r["delta_joint_minus_idle"]["storage_terminal_kwh"] == pytest.approx(0)
    assert r["gate"]["eligible_for_offline_simulation"]
    assert not r["gate"]["device_dispatch_authorized"]


def test_tiny_exhaustive_oracle_with_efficiency():
    d = sample()
    d["vehicles"] = []
    d["storage"].update(capacity_kwh=8, initial_energy_kwh=4, terminal_energy_kwh=4,
                        reserve_soc=0, min_soc=0, max_soc=1, max_charge_kw=[4]*4,
                        max_discharge_kw=[4]*4, charge_efficiency=0.5, discharge_efficiency=1)
    # Powers in steps of 2 kW exactly include every optimum corner of this fixture.
    best = float("inf")
    feasible_count = 0
    for dispatch in product((-4, -2, 0, 2, 4), repeat=4):
        energy, cost = 4, 0
        for t, p in enumerate(dispatch):
            energy += (0.5*p if p >= 0 else p)*0.25
            if not 0 <= energy <= 8:
                break
            cost += (4+p)*d["rates_cny_per_kwh"][t]*0.25
        else:
            if abs(energy-4) < 1e-10:
                feasible_count += 1
                best = min(best, cost)
    assert feasible_count > 1
    r = solve(d)
    assert r["status"] == "OPTIMAL"
    assert r["objective"]["total_objective_cny"] == pytest.approx(best)
    assert best == pytest.approx(7)


@pytest.mark.parametrize("change, expected", [
    (lambda d: d["vehicles"][0].update(required_energy_kwh=5), "INFEASIBLE"),
    (lambda d: d["vehicles"][0].update(connector="AC"), "INFEASIBLE"),
    (lambda d: d["storage"].update(initial_energy_kwh=2), "INFEASIBLE"),
    (lambda d: d["storage"].update(terminal_energy_kwh=9, max_charge_kw=[1]*4), "INFEASIBLE"),
    (lambda d: d.update(base_load_kw=[50]*4), "INFEASIBLE"),
    (lambda d: d.update(commitment_status="unknown"), "INVALID_INPUT"),
    (lambda d: d.update(pv_available_kw=[float("nan")]*4), "INVALID_INPUT"),
    (lambda d: d["storage"].update(charge_efficiency=0), "INVALID_INPUT"),
    (lambda d: d.update(base_load_kw=[1, 2]), "INVALID_INPUT"),
    (lambda d: d.update(window_start="2025-06-01T10:00:00"), "INVALID_INPUT"),
    (lambda d: d.update(grid_export_limit_kw=1), "INVALID_INPUT"),
    (lambda d: d["vehicles"].append(deepcopy(d["vehicles"][0])), "INVALID_INPUT"),
])
def test_infeasible_or_invalid_never_publishes(change, expected):
    d = sample()
    change(d)
    r = solve(d)
    assert r["status"] == expected
    assert r["plan"] is None
    if expected == "INFEASIBLE":
        assert r["reasons"][0]["code"] == "MODEL_PROVEN_INFEASIBLE"


def test_low_soc_full_pcs_with_efficiency():
    d = sample()
    d["storage"].update(initial_energy_kwh=3, terminal_energy_kwh=3, charge_efficiency=0.95, discharge_efficiency=0.95)
    r = solve(d)
    assert r["verification"]["passed"]
    p = r["plan"]
    assert min(p["storage_energy_kwh"]) >= 3-1e-6
    assert sum(p["storage_charge_kw"][:2])*0.25 == pytest.approx(2)
    assert sum(p["storage_discharge_kw"][2:])*0.25 == pytest.approx(2*0.95**2)
    assert r["objective"]["storage_loss_kwh"] == pytest.approx(2*(1-0.95**2))


def test_surplus_pv_no_export_storage_top():
    d = sample()
    d["vehicles"] = []
    d.update(pv_available_kw=[100]*4)
    d["storage"].update(initial_energy_kwh=9, terminal_energy_kwh=9)
    r = solve(d)
    assert r["verification"]["passed"]
    assert r["plan"]["net_import_kw"] == pytest.approx([0]*4)
    assert r["objective"]["pv_curtailed_kwh"] == pytest.approx(96)


def test_port_exclusivity_near_departure_and_existing_not_duplicated():
    d = sample()
    d["vehicles"][0].update(arrival_slot=3, deadline_slot=4, allowed_port_ids=["P1"])
    d["ports"][1]["occupied_slots"] = [3]
    d["existing_charge_by_port_kw"]["P2"] = [0, 0, 0, 4]
    d["storage"].update(max_charge_kw=[0]*4, max_discharge_kw=[0]*4)
    r = solve(d)
    assert r["plan"]["charge_by_vehicle_kw"]["V1"] == pytest.approx([0, 0, 0, 4])
    assert r["plan"]["net_import_kw"] == pytest.approx([4, 4, 4, 12])
    v2 = {**d["vehicles"][0], "vehicle_id": "V2"}
    d["vehicles"].append(v2)
    assert solve(d)["status"] == "INFEASIBLE"


def test_prices_cannot_override_reserve_budget_or_mutual_exclusion():
    d = sample()
    d["rates_cny_per_kwh"] = [-100, -100, 100, 100]
    d["new_charge_limit_kw"] = [0, 0, 2, 2]
    r = solve(d)
    assert r["verification"]["passed"]
    assert r["plan"]["charge_by_vehicle_kw"]["V1"] == pytest.approx([0, 0, 2, 2])
    assert all(c < 1e-6 or p < 1e-6 for c, p in zip(r["plan"]["storage_charge_kw"], r["plan"]["storage_discharge_kw"]))
    d["new_charge_limit_kw"] = [0]*4
    assert solve(d)["status"] == "INFEASIBLE"


@pytest.mark.parametrize("field", ["pv_used_kw", "net_import_kw", "storage_charge_kw", "storage_discharge_kw", "storage_energy_kwh"])
def test_tamper_any_curve_rejected(field):
    d = sample()
    p = deepcopy(solve(d)["plan"])
    p[field][0] += 1
    assert "PLAN_HASH_MISMATCH" in codes(verify_plan(d, p))
    resign(p)
    assert not verify_plan(d, p)["passed"]  # hash recomputation does not bypass physics


def test_recomputed_hash_cannot_hide_simultaneous_storage_or_bad_energy():
    d = sample()
    p = deepcopy(solve(d)["plan"])
    p["storage_charge_kw"][0] = p["storage_discharge_kw"][0] = 1
    check = verify_plan(d, resign(p))
    assert "STORAGE_SIMULTANEOUS" in codes(check)


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(net_import_kw=[float("inf")]*4),
    lambda p: p.update(net_import_kw=[1]*3),
    lambda p: p.update(assignments=None),
    lambda p: p["assignments"][0].update(power_kw=[True]*4),
    lambda p: p["assignments"][0].update(connected_slots=[-1]),
    lambda p: p["charge_by_port_kw"].pop("P1"),
])
def test_malformed_plans_fail_closed(mutation):
    d = sample()
    p = deepcopy(solve(d)["plan"])
    mutation(p)
    assert not verify_plan(d, p)["passed"]


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(grid_import_limit_kw=[0]*4),
    lambda d: d["storage"].update(initial_energy_kwh=3),
    lambda d: [p.update(max_power_kw=[0]*4) for p in d["ports"]],
])
def test_new_state_rechecked_without_mutating_plan(mutation):
    d = sample()
    p = solve(d)["plan"]
    original_hash = p["fingerprint"]
    live = deepcopy(d)
    mutation(live)
    check = verify_plan(d, p, live_input=live)
    assert not check["passed"]
    assert check["checked_input_hash"] != check["input_hash"]
    assert p["fingerprint"] == original_hash


def test_live_pv_drop_and_assignment_curve_reconstruction():
    d = sample()
    d["pv_available_kw"] = [4]*4
    p = solve(d)["plan"]
    live = deepcopy(d)
    live["pv_available_kw"] = [0]*4
    assert "PV_LIMIT" in codes(verify_plan(d, p, live_input=live))
    p["charge_by_vehicle_kw"]["V1"][0] += 1
    assert "VEHICLE_CURVE_MISMATCH" in codes(verify_plan(d, resign(p)))


@pytest.mark.parametrize("numeric_ok,governance_ok", list(product([False, True], repeat=2)))
def test_governance_and_numeric_double_gate(numeric_ok, governance_ok):
    d = sample()
    d["assumptions"] = ["explicit synthetic assumption"]
    p = solve(d)["plan"]
    if not numeric_ok:
        p["net_import_kw"][0] += 1
        resign(p)
    gate = simulation_gate(d, p, allow_assumptions=governance_ok)
    assert gate["numeric"]["passed"] == numeric_ok
    assert gate["governance"]["passed"] == governance_ok
    assert gate["eligible_for_offline_simulation"] == (numeric_ok and governance_ok)


def test_status_classification_distinguishes_unknown_feasible_infeasible():
    assert [classify_status(n) for n in (0, 1, 2, 4, 6)] == ["OPTIMAL", "FEASIBLE", "INFEASIBLE", "ERROR", "UNKNOWN"]


def test_forced_unknown_does_not_extract_or_release_plan(monkeypatch):
    from ortools.linear_solver import pywraplp
    create = pywraplp.Solver.CreateSolver

    class Interrupted:
        def __init__(self):
            self.real = create("SCIP")
        def __getattr__(self, name):
            return getattr(self.real, name)
        def Solve(self):
            return 6  # controlled backend termination, not a physical infeasibility
        def Objective(self):
            return self
        def BestBound(self):
            return float("-inf")
        def nodes(self):
            return 0

    monkeypatch.setattr(pywraplp.Solver, "CreateSolver", lambda name: Interrupted())
    r = solve(sample())
    assert r["status"] == "UNKNOWN" and r["plan"] is None
    assert not r["solver"]["optimality_proven"]
    assert r["reasons"][0]["code"] == "NO_INCUMBENT_FEASIBILITY_UNKNOWN"


def test_feasible_incumbent_never_claims_optimal(monkeypatch):
    from ortools.linear_solver import pywraplp
    create = pywraplp.Solver.CreateSolver

    class Incumbent:
        def __init__(self):
            self.real = create("SCIP")
        def __getattr__(self, name):
            return getattr(self.real, name)
        def Solve(self):
            assert self.real.Solve() == 0
            return 1  # controlled status-path test; real numeric incumbent

    monkeypatch.setattr(pywraplp.Solver, "CreateSolver", lambda name: Incumbent())
    r = solve(sample())
    assert r["status"] == "FEASIBLE" and r["plan"] is not None
    assert r["verification"]["passed"] and not r["solver"]["optimality_proven"]


def test_2mw_4mwh_is_not_two_hours_at_initial_soc():
    d = sample()
    d["vehicles"] = []
    d.update(base_load_kw=[3000]*4, grid_import_limit_kw=[10000]*4)
    d["storage"].update(capacity_kwh=4000, initial_energy_kwh=2400, terminal_energy_kwh=1200,
                        max_charge_kw=[2000]*4, max_discharge_kw=[2000]*4,
                        charge_efficiency=0.95, discharge_efficiency=0.95)
    r = solve(d)
    assert r["verification"]["passed"]
    # max(10%,30%)=30%, never 10%+30%; delivered AC energy is 1200*0.95.
    assert sum(r["plan"]["storage_discharge_kw"])*0.25 == pytest.approx(1140)
    assert r["objective"]["storage_terminal_kwh"] == pytest.approx(1200)
    assert compare_idle(d)["idle"]["status"] == "INFEASIBLE"


def test_infeasibility_reports_numeric_shortfall():
    d = sample()
    d["vehicles"][0]["required_energy_kwh"] = 5
    r = solve(d)
    conflict = next(e for e in r["reasons"] if e["code"] == "VEHICLE_WINDOW_CAPACITY_SHORTFALL")
    assert conflict["required_kwh"] == 5 and conflict["relaxed_maximum_kwh"] == 4


def test_connection_session_conflict_rejected_even_without_power_overlap():
    d = sample()
    p = solve(d)["plan"]
    assigned_port = p["assignments"][0]["port_id"]
    live = deepcopy(d)
    index = next(i for i, port in enumerate(live["ports"]) if port["port_id"] == assigned_port)
    live["ports"][index]["occupied_slots"] = p["assignments"][0]["connected_slots"]
    assert "PORT_CONFLICT" in codes(verify_plan(d, p, live_input=live))


def test_freeze_copies_and_binds_all_fields():
    d = sample()
    frozen = freeze(d)
    d["storage"]["terminal_energy_kwh"] = 7
    assert frozen["input"]["storage"]["terminal_energy_kwh"] == 5
    assert frozen["input_hash"] != freeze(d)["input_hash"]


@pytest.fixture(scope="module")
def replay():
    return LinkedReplay(DATA)


def test_d01_default_fails_closed_and_never_fills_unknown(replay):
    for sid, expected in (("S01_sunny", "COMMITMENT_PROFILE_MISSING"), ("S10_no_booking", "COMMITMENT_UNKNOWN"),
                          ("S11_pv_gap", "FORECAST_UNAVAILABLE"), ("S12_late_weather", "FORECAST_UNAVAILABLE")):
        with pytest.raises(InputError) as e:
            replay.load(sid)
        assert e.value.code == expected
    with pytest.raises(InputError):
        replay.load("S10_no_booking", demo_assumptions=True)


def test_d01_all_valid_demo_scenarios_and_input_provenance(replay):
    for sid in replay.scenario_ids()[:9]:
        d = replay.load(sid, demo_assumptions=True)
        assert d["source_site_dispatch_ready"] is False
        assert not any("truth" in ref for ref in d["source_refs"])
        assert d["storage"]["terminal_energy_kwh"] == d["storage"]["initial_energy_kwh"]
        r = compare_idle(d, time_limit_seconds=5, allow_assumptions=True)
        assert r["joint"]["status"] in {"OPTIMAL", "FEASIBLE", "INFEASIBLE"}, r
        if r["joint"]["plan"]:
            assert r["gate"]["eligible_for_offline_simulation"]
        if r["idle"]["status"] == r["joint"]["status"] == "OPTIMAL":
            assert r["delta_joint_minus_idle"]["total_objective_cny"] <= 1e-5


def test_source_hash_tampering_rejected(tmp_path):
    linked = tmp_path / "场站联动模拟数据"
    linked.mkdir()
    report = {"status": "passed", "output_hashes_sha256": {"scenario_inputs.csv": "0"*64}}
    (linked / "validation_report.json").write_text(json.dumps(report), encoding="utf-8")
    (linked / "scenario_inputs.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(InputError) as e:
        LinkedReplay(tmp_path)
    assert e.value.code == "SOURCE_HASH_MISMATCH"
