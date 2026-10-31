"""Reproducible offline review evidence. All original data/optimizer files read-only."""
from copy import deepcopy
import json
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from station_review import digest, review_plan
from station_review.adapters import audit_d02, check_sources, file_hash, load_json, review_a02_archive, source_path
from station_review.contract import content_hash
from station_review.examples import fixture, partial_fixture, synthetic_policy

OUT = ROOT/"开发验证"
DATA = ROOT.parent/"数据"
ARCHIVE = ROOT.parent/"场站光储充联合优化"/"开发验证"/"replay_demo.json"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)+"\n", encoding="utf-8")


def seal(b):
    b["plan"]["fingerprint"] = content_hash(b["plan"])
    b["plan_fingerprint"] = b["plan"]["fingerprint"]
    b["fingerprint"] = content_hash(b)


def request(b, state, policy=None):
    raw = b.to_dict() if hasattr(b, "to_dict") else b
    return {"frozen_plan": raw, "latest_state": state,
            "governance_policy": synthetic_policy(raw, state, state["available_at"]) if policy is None else policy,
            "as_of": state["available_at"]}


def scenarios():
    cases = []

    def add(name, b, s, expected, code=None, policy=None):
        cases.append({"scenario_id": name, "expected_passed": expected, "required_violation": code,
                      "request": request(b, s, policy)})

    b, s, p = fixture()
    add("normal_hand_plan", b, s, True, policy=p)
    b, s, _ = partial_fixture()
    add("partial_actual_prefix", b, s, True)

    def original():
        b, s, _ = fixture()
        return b.to_dict(), s

    b, s = original()
    b["plan"]["pv_used_kw"][0] += 1
    add("curve_tampered_without_hash", b, s, False, "FROZEN_HASH_MISMATCH")
    b, s = original()
    b["plan"]["net_import_kw"][0] = 0
    seal(b)
    add("forged_grid_with_new_hash", b, s, False, "POWER_BALANCE")
    b, s = original()
    s["measurement"]["pv_available_kw"] = 0.5
    add("latest_pv_drop", b, s, False, "PV_LIMIT")
    b, s = original()
    s["measurement"].update(storage_energy_kwh=2, storage_soc=0.2)
    add("latest_storage_below_reserve", b, s, False, "STORAGE_LIVE_SOC")
    b, s = original()
    s["ports"]["P1"]["fault"] = True
    add("assigned_port_fault", b, s, False, "PORT_BLOCKED")
    b, s = original()
    s["grid_import_limit_kw"] = 6
    add("latest_grid_limit_smaller", b, s, False, "GRID_IMPORT_EXPORT")
    b, s = original()
    b["dependencies"][0]["valid_until"] = b["issued_at"]
    seal(b)
    add("forecast_expired", b, s, False, "DEPENDENCY_EXPIRED_OR_FUTURE")
    b, s, _ = partial_fixture()
    s["execution"]["receipts"][0]["plan_version"] = 2
    add("late_wrong_version_receipt", b, s, False, "RECEIPT_OLD_VERSION")
    _, s, old = partial_fixture()
    add("resend_old_full_plan_after_execution", old, s, False, "EXECUTED_PREFIX_MISMATCH")
    b, s, _ = partial_fixture()
    s["execution"].update(cancel_status="requested", stopped_at=None, stopped_after_slot=None)
    s["execution"]["accepted_plan_history"][-1]["stop_after_slot"] = None
    add("new_plan_before_stop_confirmation", b, s, False, "CANCEL_NOT_CONFIRMED")
    b, s, _ = fixture()
    add("missing_governance_approval", b, s, False, "GOVERNANCE_INPUT_INVALID", policy={})
    b, s, policy = fixture()
    s["revision"] += 1
    add("new_state_old_approval", b, s, False, "APPROVAL_BINDING_MISMATCH", policy=policy)
    return cases


def main():
    archive = load_json(ARCHIVE)
    consumed = {ARCHIVE: file_hash(ARCHIVE)}
    for row in archive["scenarios"]:
        if row.get("input"):
            check_sources(row["input"], DATA)
            for ref in row["input"]["source_refs"]:
                path = source_path(DATA, ref)
                consumed[path] = file_hash(path)
    for name in ("events.jsonl", "scenario_manifest.json", "simulation_config.json", "validation_report.json"):
        path = DATA/"滚动事件模拟数据"/name
        consumed[path] = file_hash(path)
    d02_config = load_json(DATA/"滚动事件模拟数据"/"simulation_config.json")
    for ref in d02_config["source_hashes_sha256"]:
        path = DATA/ref
        consumed[path] = file_hash(path)

    records, artifacts = [], []
    b, state, policy = fixture()
    path = OUT/"hand_demo.json"
    write(path, {"request": request(b, state, policy),
                 "review": review_plan(b, state, policy, as_of=state["available_at"])})
    artifacts.append(path)
    for case in scenarios():
        name, req = case["scenario_id"], case["request"]
        inp, out = OUT/"requests"/(name+".json"), OUT/"results"/(name+".json")
        write(inp, req)
        proc = subprocess.run([sys.executable, "-X", "utf8", "-m", "station_review", "review",
                               "--request", str(inp), "--output", str(out)], cwd=ROOT,
                              capture_output=True, text=True, encoding="utf-8", timeout=20)
        actual = load_json(out)
        assert proc.returncode == (0 if case["expected_passed"] else 2), (name, proc.stderr)
        assert actual["passed"] == case["expected_passed"], name
        assert not actual["device_dispatch_authorized"]
        if case["required_violation"]:
            assert case["required_violation"] in {v["code"] for v in actual["violations"]}, name
        again = review_plan(req["frozen_plan"], req["latest_state"], req["governance_policy"], as_of=req["as_of"])
        assert again == actual, name
        records.append({"scenario_id": name, "expected_passed": case["expected_passed"], "cli_exit_code": proc.returncode,
                        "request_path": str(inp.relative_to(ROOT)), "result_path": str(out.relative_to(ROOT)),
                        "required_violation": case["required_violation"], "review": actual, "replayed_identically": True})
        artifacts.extend([inp, out])
    path = OUT/"simulation_e2e.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)+"\n" for row in records), encoding="utf-8")
    artifacts.append(path)

    combos = []
    for physical in (True, False):
        for governance in (True, False):
            b, state, _ = fixture()
            b = b.to_dict()
            if not physical:
                b["plan"]["net_import_kw"][0] = 100
                seal(b)
            policy = synthetic_policy(b, state, state["available_at"])
            policy["approval"]["released"] = governance
            check = review_plan(b, state, policy, as_of=state["available_at"])
            assert check["physical"]["passed"] == physical and check["governance"]["passed"] == governance
            assert check["passed"] == (physical and governance)
            combos.append({"expected_physical": physical, "expected_governance": governance, "review": check})
    path = OUT/"double_gate.json"
    write(path, combos)
    artifacts.append(path)

    for name, value in (("a02_compatibility.json", review_a02_archive(ARCHIVE, DATA, demo_assumptions=True)),
                        ("a02_strict.json", review_a02_archive(ARCHIVE, DATA)),
                        ("d02_strict_audit.json", audit_d02(DATA))):
        path = OUT/name
        write(path, value)
        artifacts.append(path)
    assert load_json(OUT/"a02_compatibility.json")["passed_count"] == 9
    assert load_json(OUT/"a02_strict.json")["passed_count"] == 0
    assert audit_d02(DATA) == load_json(OUT/"d02_strict_audit.json")
    after = {path: file_hash(path) for path in consumed}
    assert consumed == after, "An original data/archive file changed"

    xml = OUT/"tests.xml"
    suite = ET.parse(xml).getroot()
    totals = {key: sum(int(x.attrib.get(key, 0)) for x in suite.iter("testsuite"))
              for key in ("tests", "failures", "errors", "skipped")}
    assert totals["tests"] > 0 and totals["failures"] == totals["errors"] == 0
    source_files = sorted(ROOT.glob("station_review/*.py"))+sorted(ROOT.glob("tests/*.py"))
    source_files += [Path(__file__), ROOT/"pyproject.toml", ROOT/".gitignore", ROOT/"README.md", OUT/"验收记录.md"]
    manifest = {"recorded_on": "2026-10-29", "business_clock": "explicit 2025 offline replay",
                "python": platform.python_version(), "tests": totals,
                "simulation_e2e": {"cases": len(records), "allowed": sum(r["expected_passed"] for r in records),
                                   "rejected": sum(not r["expected_passed"] for r in records), "all_replayed_identically": True},
                "a02_archive": {"demo_passed": 9, "strict_passed": 0, "no_plan": 3},
                "d02": {"scenarios": 22, "events": 152, "release_count": 0},
                "source_files_unchanged": True,
                "consumed_source_hashes_before_and_after": {str(path): value for path, value in consumed.items()},
                "implementation_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in source_files},
                "evidence_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in artifacts+[xml]},
                "device_dispatch_authorized": False, "certified_safety_service": False}
    write(OUT/"verification_manifest.json", manifest)
    print(json.dumps({"tests": totals, "e2e_cases": len(records), "sources_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
