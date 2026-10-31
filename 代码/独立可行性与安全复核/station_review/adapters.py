"""Read-only archive/source adapters. Never fill unknown live fields from truth."""
import csv
import hashlib
import json
import os
from pathlib import Path

from .contract import ReviewError, digest, freeze_plan, need, time_at
from .examples import synthetic_policy, synthetic_state
from .review import review_plan


def load_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "DUPLICATE_JSON_KEY", key)
            result[key] = value
        return result

    def bad_constant(value):
        raise ReviewError("NONFINITE_NUMBER", value)

    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=pairs, parse_constant=bad_constant)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def absolute_path(path):
    # Normalize '..' before asking Windows to resolve reparse points. This also
    # works under redirected/sandboxed working directories.
    return Path(os.path.abspath(path)).resolve()


def source_path(root, ref):
    root = absolute_path(root)
    relative = "场站联动模拟数据/validation_report.json" if ref == "linked_validation_report" else ref
    p = (root/relative if "/" in relative or "\\" in relative else root/"场站联动模拟数据"/relative).resolve()
    need(p.is_relative_to(root), "SOURCE_PATH_INVALID", "source ref")
    return p


def check_sources(d, root):
    consumed = {}
    for ref, expected in d["source_refs"].items():
        actual = file_hash(source_path(root, ref))
        need(actual == expected, "SOURCE_FILE_CHANGED", ref, actual, expected)
        consumed[ref] = actual
    return consumed


def a02_dependencies(d, data_root):
    """Use actual D01 vintage timestamps, without rewriting optimizer constraints."""
    linked = Path(data_root)/"场站联动模拟数据"
    with (linked/"scenario_inputs.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["scenario_id"] == d["task_id"]]
    need(len(rows) == 1, "SOURCE_CONTEXT_MISSING", "scenario_inputs")
    scenario = rows[0]
    with (linked/"forecast_vintages.csv").open(encoding="utf-8-sig", newline="") as f:
        vintages = list(csv.DictReader(f))
    deps = []
    for resource in ("load", "pv"):
        selected = [v for v in vintages if v["vintage_id"] == scenario[resource+"_vintage_id"]]
        need(len(selected) == 1 and selected[0]["status"] == "usable", "FORECAST_UNAVAILABLE", resource)
        v = selected[0]
        need(time_at(v["issued_at"]) <= time_at(d["as_of"]), "FORECAST_FUTURE", resource)
        deps.append({"ref": "forecast_values.csv", "kind": "forecast_"+resource,
                     "status": v["status"], "native_grid": v["native_grid"],
                     "sha256": d["source_refs"]["forecast_values.csv"], "vintage_id": v["vintage_id"],
                     "issued_at": v["issued_at"], "available_at": v["issued_at"],
                     "valid_until": v["window_end"], "window_start": v["window_start"], "window_end": v["window_end"],
                     "validity_rule": "explicit offline window-only rule, not a live forecast TTL"})
    deps.append({"ref": "booking_events.csv", "kind": "commitments", "status": "usable", "sha256": d["source_refs"]["booking_events.csv"],
                 "issued_at": d["as_of"], "available_at": d["as_of"], "valid_until": d["window_end"],
                 "window_start": d["window_start"], "window_end": d["window_end"],
                 "validity_rule": "A02 assumed complete profiles frozen at its as_of; not validated future booking state"})
    return deps


def review_a02_archive(archive_path, data_root, *, demo_assumptions=False):
    """Archive compatibility evidence. Explicit demo snapshots are synthetic only."""
    archive = load_json(archive_path)
    out = {"simulation": True, "source": str(absolute_path(archive_path)), "archive_sha256": file_hash(archive_path),
           "demo_assumptions": demo_assumptions, "scenarios": [], "device_dispatch_authorized": False}
    consumed = {}
    for row in archive["scenarios"]:
        item = {"scenario_id": row["scenario_id"], "passed": False}
        if row.get("input") is None or row.get("joint", {}).get("plan") is None:
            item["adapter_violations"] = [{"code": "NO_OPTIMIZER_PLAN", "field": "A02 archive"}]
            out["scenarios"].append(item)
            continue
        d, p = row["input"], row["joint"]["plan"]
        try:
            consumed.update(check_sources(d, data_root))
            b = freeze_plan(d, p, producer="A02", producer_version="a02-joint-v1", publisher="O1-SIM",
                            plan_version=1, dependencies=a02_dependencies(d, data_root),
                            issued_at=d["as_of"], valid_until=d["window_end"])
            state, policy = {}, {}
            if demo_assumptions:
                # The entire snapshot is explicitly hypothetical. It is NOT inferred
                # from future station_interval_truth or execution_truth.
                state = synthetic_state(d, at=d["as_of"], pv_cap=5000, initial_energy=d["storage"]["initial_energy_kwh"])
                policy = synthetic_policy(b, state, d["as_of"], allow_assumptions=True)
            request = {"frozen_plan": b.to_dict(), "latest_state": state, "governance_policy": policy, "as_of": d["as_of"]}
            check = review_plan(b, state, policy, as_of=d["as_of"])
            item.update(passed=check["passed"], request=request, review=check,
                        snapshot_basis="explicit A03 demo assumptions" if demo_assumptions else "missing live state and approval; fail closed")
        except (ReviewError, OSError, KeyError, TypeError, ValueError) as e:
            item["adapter_violations"] = [{"code": getattr(e, "code", "ARCHIVE_INPUT_INVALID"),
                                          "field": getattr(e, "field", "archive"), "detail": str(e)}]
        out["scenarios"].append(item)
    # Recheck every consumed source after processing to expose concurrent replacement.
    for ref, expected in consumed.items():
        need(file_hash(source_path(data_root, ref)) == expected, "SOURCE_FILE_CHANGED", ref)
    out["consumed_sources_sha256"] = consumed
    out["passed_count"] = sum(x["passed"] for x in out["scenarios"])
    return out


def audit_d02(data_root):
    """Inspect raw visible workflow fixtures; no promotion into an A03 candidate.

    D02 lacks port mapping, complete commitments, and rated/live device state.
    Audit refusals occur at adaptation, before physical/governance gate evaluation.
    """
    root = absolute_path(data_root)
    folder = root/"滚动事件模拟数据"
    config = load_json(folder/"simulation_config.json")
    report = load_json(folder/"validation_report.json")
    hashes = {}
    for ref, expected in config["source_hashes_sha256"].items():
        p = (root/ref).resolve()
        need(p.is_relative_to(root.resolve()), "SOURCE_PATH_INVALID", ref)
        actual = file_hash(p)
        need(actual == expected, "SOURCE_FILE_CHANGED", ref, actual, expected)
        hashes[ref] = actual
    for name in ("events.jsonl", "scenario_manifest.json"):
        actual = file_hash(folder/name)
        need(actual == report["output_hashes_sha256"][name], "SOURCE_FILE_CHANGED", name)
        hashes["滚动事件模拟数据/"+name] = actual
    hashes["滚动事件模拟数据/simulation_config.json"] = file_hash(folder/"simulation_config.json")
    events = [json.loads(line) for line in (folder/"events.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    manifests = load_json(folder/"scenario_manifest.json")
    rows = []
    for m in manifests:
        stream = sorted([x for x in events if x["scenario_id"] == m["scenario_id"]],
                        key=lambda e: (time_at(e["received_at"]), e["ordinal"]))
        attempts = []
        for event in stream:
            if event["event_type"] != "plan_sent":
                continue
            at = time_at(event["received_at"])
            visible = [x["event_id"] for x in stream if time_at(x["available_at"]) <= at and time_at(x["received_at"]) <= at]
            plan = event["payload"]["plan"]
            need(digest(plan) == event["plan_fingerprint"], "PLAN_HASH_MISMATCH", event["event_id"])
            attempts.append({"event_id": event["event_id"], "checked_at": event["received_at"], "visible_event_ids": visible,
                             "plan_fingerprint": event["plan_fingerprint"], "passed": False, "stage": "adapter",
                             "violations": [{"code": code} for code in ("D02_WORKFLOW_PLAN_UNSUPPORTED", "DEVICE_MAPPING_MISSING",
                                                                         "COMMITMENT_PROFILE_MISSING", "COMPLETE_LIVE_STATE_MISSING",
                                                                         "VERSIONED_APPROVAL_MISSING")]})
        rows.append({"scenario_id": m["scenario_id"], "attempts": attempts, "release_count": 0})
    return {"simulation": True, "audit_kind": "strict_D02_adapter_refusal", "event_count": len(events),
            "scenario_count": len(rows), "scenarios": rows, "source_hashes_sha256": hashes,
            "device_dispatch_authorized": False,
            "limitation": "Workflow fixture acceptance is not evidence of A03 safety or algorithm feasibility; no truth values used to fill live inputs."}
