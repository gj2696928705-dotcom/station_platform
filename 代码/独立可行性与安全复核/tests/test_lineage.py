from copy import deepcopy

from station_review import freeze_plan, review_plan
from station_review.examples import fixture, partial_fixture, synthetic_policy


def check(b, s):
    return review_plan(b, s, synthetic_policy(b, s, s["available_at"]), as_of=s["available_at"])


def second_partial():
    b, s, _ = partial_fixture()
    previous = b.to_dict()
    at = "2025-06-01T10:30:00+08:00"
    r = deepcopy(s["execution"]["receipts"][0])
    r.update(event_id="receipt-1", slot=1, plan_version=2, plan_fingerprint=previous["plan_fingerprint"], measured_at=at, available_at=at)
    r["actual"].update(vehicle_energy_kwh={"V1": 1}, connected_vehicle_port_ids={"V1": "P1"}, storage_charge_energy_kwh=0,
                       storage_energy_before_kwh=5.5, storage_energy_after_kwh=5.5)
    s["execution"]["receipts"].append(r)
    s["execution"]["accepted_plan_history"].append({"plan_version": 2, "plan_fingerprint": previous["plan_fingerprint"],
                                                      "execute_from_slot": 1, "stop_after_slot": 2})
    s["execution"].update(accepted_plan_version=2, accepted_plan_fingerprint=previous["plan_fingerprint"], stopped_at=at, stopped_after_slot=2)
    s.update(measured_at=at, available_at=at, valid_until="2025-06-01T10:31:00+08:00")
    b = freeze_plan(previous["constraints"], previous["plan"], producer="A02", producer_version="a02-joint-v1",
                    publisher="O1-SIM", plan_version=3, dependencies=previous["dependencies"], issued_at=at,
                    valid_until=previous["valid_until"], execute_from_slot=2, executed_prefix=s["execution"]["receipts"])
    return b, s


def test_two_replans_accept_historical_receipts_with_their_own_versions():
    b, s = second_partial()
    r = check(b, s)
    assert r["passed"], r["violations"]
    assert r["physical"]["metrics"]["remaining_commitment_by_vehicle_kwh"] == {"V1": 0}
    assert r["physical"]["metrics"]["reconstructed_remaining_energy_kwh"] == [5.5, 5, 5]


def test_old_version_cannot_claim_a_step_after_its_stop_boundary():
    b, s = second_partial()
    s["execution"]["receipts"][1]["plan_version"] = 1
    r = check(b, s)
    assert not r["passed"] and "RECEIPT_OLD_VERSION" in {x["code"] for x in r["violations"]}


def test_cancellation_request_does_not_release_old_reservation():
    b, s, _ = partial_fixture()
    s["execution"].update(cancel_status="requested", stopped_at=None, stopped_after_slot=None)
    s["execution"]["accepted_plan_history"][-1]["stop_after_slot"] = None
    r = check(b, s)
    assert not r["passed"] and "CANCEL_NOT_CONFIRMED" in {x["code"] for x in r["violations"]}


def test_history_gap_is_rejected_even_with_correct_receipt_hash():
    b, s = second_partial()
    s["execution"]["accepted_plan_history"][1]["execute_from_slot"] = 2
    r = check(b, s)
    assert not r["passed"] and "EXECUTION_HISTORY_GAP" in {x["code"] for x in r["violations"]}


def test_initial_accepted_same_plan_can_be_rechecked_without_cancellation():
    b, s, _ = fixture()
    raw = b.to_dict()
    s["execution"].update(accepted_plan_version=1, accepted_plan_fingerprint=raw["plan_fingerprint"],
                          accepted_plan_history=[{"plan_version": 1, "plan_fingerprint": raw["plan_fingerprint"],
                                                  "execute_from_slot": 0, "stop_after_slot": None}])
    assert check(b, s)["passed"]


def test_freeze_normalizes_transport_only_repeats():
    b, s, _ = partial_fixture()
    raw = b.to_dict()
    r = deepcopy(s["execution"]["receipts"][0])
    r["received_at"] = r["available_at"]
    normalized = freeze_plan(raw["constraints"], raw["plan"], producer=raw["producer"], producer_version=raw["producer_version"],
                             publisher=raw["publisher"], plan_version=raw["plan_version"], dependencies=raw["dependencies"],
                             issued_at=raw["issued_at"], valid_until=raw["valid_until"], execute_from_slot=1,
                             executed_prefix=[r, deepcopy(r)])
    assert normalized.to_dict()["executed_prefix_hash"] == raw["executed_prefix_hash"]
    assert check(normalized, s)["passed"]


def test_new_forecast_only_needs_to_cover_remaining_suffix():
    b, s, _ = partial_fixture()
    raw = b.to_dict()
    raw["constraints"]["schema_version"] = "a03-constraints-v1"
    raw["constraints"]["as_of"] = s["available_at"]
    raw["plan"]["schema_version"] = "a03-plan-v1"
    from station_review.contract import content_hash, digest
    raw["constraints_hash"] = digest(raw["constraints"])
    raw["plan"]["input_hash"] = raw["constraints_hash"]
    raw["plan"]["fingerprint"] = content_hash(raw["plan"])
    raw["plan_fingerprint"] = raw["plan"]["fingerprint"]
    for dep in raw["dependencies"]:
        dep.update(issued_at=s["available_at"], available_at=s["available_at"], window_start=s["available_at"])
    raw["fingerprint"] = content_hash(raw)
    r = check(raw, s)
    assert r["passed"], r["violations"]


def test_missing_dependency_quality_fails_closed():
    b, s, _ = fixture()
    raw = b.to_dict()
    raw["dependencies"][0].pop("status")
    from station_review.contract import content_hash
    raw["fingerprint"] = content_hash(raw)
    assert not check(raw, s)["passed"]
