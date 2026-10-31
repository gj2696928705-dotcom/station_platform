"""A single fail-closed gate evaluates both layers on the same owned snapshot."""
import json
from datetime import timedelta

from .contract import (MAX_STATE_AGE_SECONDS, RULE, FrozenPlan, ReviewError,
                       canonical, digest, time_at)
from .governance import verify_governance
from .physical import verify_physical


def review_plan(frozen, state, policy, *, as_of, expected_state_hash=None):
    """Review an explicit business clock. Always recompute; no caller-supplied verdicts.

    state and policy are trusted station/authorization inputs, not optimizer fields.
    expected_state_hash can be supplied at publication to reject a changed snapshot.
    The output is offline evidence, never a device dispatch command or certificate.
    """
    result = {"schema_version": "a03-review-v1", "passed": False, "eligible_for_offline_simulation": False,
              "status": "REJECTED", "simulation": True, "device_dispatch_authorized": False,
              "certified_safety_service": False, "rule_version": RULE, "checked_at": as_of if type(as_of) is str else None,
              "state_hash": None, "policy_hash": None, "frozen_fingerprint": None,
              "plan_fingerprint": None, "plan_version": None, "task_id": None, "task_version": None,
              "constraints_hash": None, "state_reference": None, "valid_until": None,
              "violations": [], "physical": None, "governance": None,
              "on_failure": "reject_and_request_new_plan_or_state"}
    try:
        at = time_at(as_of, "as_of")
        b = json.loads(canonical(frozen.to_dict() if isinstance(frozen, FrozenPlan) else frozen))
        snapshot = json.loads(canonical(state))
        rules = json.loads(canonical(policy))
        # Own bytes prevent either checker mutating the other checker's inputs.
        state_hash, policy_hash = digest(snapshot), digest(rules)
        result.update(state_hash=state_hash, policy_hash=policy_hash,
                      frozen_fingerprint=b.get("fingerprint"), plan_fingerprint=b.get("plan_fingerprint"),
                      plan_version=b.get("plan_version"), constraints_hash=b.get("constraints_hash"))
        d = b.get("constraints", {})
        result.update(task_id=d.get("task_id"), task_version=d.get("task_version"),
                      state_reference={key: snapshot.get(key) for key in
                                       ("state_id", "revision", "measured_at", "available_at", "valid_until")})
        if expected_state_hash is not None and expected_state_hash != state_hash:
            result["violations"].append({"code": "STATE_CHANGED_SINCE_REVIEW", "field": "state_hash",
                                         "actual": state_hash, "expected": expected_state_hash})
        physical = verify_physical(b, snapshot, at)
        governance = verify_governance(b, snapshot, rules, at)
        binding = {"plan_fingerprint": result["plan_fingerprint"], "frozen_fingerprint": result["frozen_fingerprint"],
                   "plan_version": result["plan_version"], "state_hash": state_hash,
                   "state_reference": result["state_reference"], "checked_at": as_of, "policy_hash": policy_hash}
        for name, check in (("physical", physical), ("governance", governance)):
            check.update(binding)
            result[name] = check
            result["violations"].extend({"layer": name, **v} for v in check["violations"])
        if physical["passed"] is True and governance["passed"] is True and not result["violations"]:
            expiries = [time_at(b["valid_until"]), time_at(snapshot["valid_until"]),
                        time_at(snapshot["measured_at"])+timedelta(seconds=MAX_STATE_AGE_SECONDS),
                        time_at(rules["approval"]["valid_until"])]
            expiries.extend(time_at(x["valid_until"]) for x in b["dependencies"])
            result.update(passed=True, status="SIMULATION_ALLOWED", eligible_for_offline_simulation=True,
                          valid_until=min(expiries).isoformat())
    except (ReviewError, KeyError, TypeError, ValueError, AttributeError, IndexError, ArithmeticError, RecursionError) as e:
        code = e.code if isinstance(e, ReviewError) else "REVIEW_INPUT_INVALID"
        result["violations"].append({"code": code, "field": getattr(e, "field", "review input"),
                                     "actual": type(e).__name__, "expected": "complete finite JSON review input"})
    result["review_fingerprint"] = digest(result)
    return result
