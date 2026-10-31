"""Version-pinned offline publication rules; no physical pass flag is accepted."""
from .contract import RULE, ReviewError, digest, need, sha, text_id, time_at
from .physical import identity


def verify_governance(b, state, policy, at):
    errors = []

    def fail(code, field, actual=None, expected=None):
        errors.append({"code": code, "field": field, "actual": actual, "expected": expected})

    result = {"passed": False, "rule_version": RULE, "violations": errors}
    try:
        d, p = identity(b)
        text_id(policy["policy_id"], "policy.policy_id")
        need(type(policy["authorized_publishers"]) is list and bool(policy["authorized_publishers"])
             and all(type(x) is str and bool(x.strip()) for x in policy["authorized_publishers"]),
             "GOVERNANCE_INPUT_INVALID", "authorized_publishers")
        need(type(policy["allowed_producers"]) is dict and all(type(v) is list and bool(v)
             and all(type(x) is str and bool(x.strip()) for x in v) for v in policy["allowed_producers"].values()),
             "GOVERNANCE_INPUT_INVALID", "allowed_producers")
        need(type(policy["allowed_sources"]) is dict and bool(policy["allowed_sources"]),
             "GOVERNANCE_INPUT_INVALID", "allowed_sources")
        for key, value in policy["allowed_sources"].items():
            text_id(key, "allowed_sources.ref")
            sha(value, "allowed_sources.sha256")
        need(policy["rule_version"] == RULE, "GOVERNANCE_RULE_UNSUPPORTED", "policy.rule_version")
        need(policy["simulation_only"] is True and d["simulation"] is True and p["simulation"] is True
             and state["is_simulated"] is True
             and d["execution_mode"] == "offline_replay", "SIMULATION_MODE_REQUIRED", "execution_mode")
        need(b["publisher"] in policy["authorized_publishers"], "PUBLISHER_UNAUTHORIZED", "publisher", b["publisher"])
        need(b["producer_version"] in policy["allowed_producers"].get(b["producer"], []),
             "PRODUCER_UNAUTHORIZED", "producer_version", b["producer_version"])
        for key, value in d["source_refs"].items():
            need(policy["allowed_sources"].get(key) == value, "SOURCE_UNAPPROVED", "source_refs."+key)
        need(not d["assumptions"] or policy["allow_assumptions"] is True,
             "ASSUMPTIONS_NOT_ACCEPTED", "assumptions")
        need(state["station_id"] == d["station_id"] and state["task_id"] == d["task_id"]
             and state["task_version"] == d["task_version"], "GOVERNANCE_CONTEXT_MISMATCH", "state context")
        approval = policy["approval"]
        expected = {"frozen_fingerprint": b["fingerprint"], "plan_fingerprint": b["plan_fingerprint"],
                    "state_hash": digest(state), "state_id": state["state_id"], "state_revision": state["revision"],
                    "task_id": d["task_id"], "task_version": d["task_version"], "publisher": b["publisher"]}
        for key, value in expected.items():
            need(type(approval[key]) is type(value) and approval[key] == value,
                 "APPROVAL_BINDING_MISMATCH", "approval."+key, approval[key], value)
        need(approval["simulation_only"] is True and approval["released"] is True,
             "SIMULATION_RELEASE_MISSING", "approval.released")
        approved, valid = (time_at(approval[k], "approval."+k) for k in ("approved_at", "valid_until"))
        need(time_at(b["issued_at"]) <= approved <= at < valid
             and valid <= min(time_at(b["valid_until"]), time_at(state["valid_until"])),
             "APPROVAL_EXPIRED_OR_FUTURE", "approval timestamps")
    except ReviewError as e:
        fail(e.code, e.field, e.actual, e.expected)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, ArithmeticError) as e:
        fail("GOVERNANCE_INPUT_INVALID", "governance input", type(e).__name__, "complete versioned policy and approval")
    result["passed"] = not errors
    return result
