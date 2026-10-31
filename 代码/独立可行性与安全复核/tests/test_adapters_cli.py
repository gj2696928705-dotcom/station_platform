import json
from pathlib import Path
import subprocess
import sys

import pytest

from station_review import ReviewError, digest, review_plan
from station_review.adapters import audit_d02, check_sources, load_json, review_a02_archive, source_path
from station_review.examples import fixture, partial_fixture, synthetic_policy

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent/"数据"
ARCHIVE = ROOT.parent/"场站光储充联合优化"/"开发验证"/"replay_demo.json"


def test_actual_a02_archive_with_explicit_synthetic_snapshot():
    r = review_a02_archive(ARCHIVE, DATA, demo_assumptions=True)
    assert r["passed_count"] == 9
    assert [x["passed"] for x in r["scenarios"]] == [True]*9+[False]*3
    assert r["consumed_sources_sha256"]
    assert all(x["review"]["physical"]["passed"] and x["review"]["governance"]["passed"]
               for x in r["scenarios"][:9])
    assert all(x["request"]["latest_state"]["is_simulated"] for x in r["scenarios"][:9])


def test_archive_without_state_and_approval_cannot_release():
    r = review_a02_archive(ARCHIVE, DATA)
    assert r["passed_count"] == 0
    assert all(not x["review"]["passed"] for x in r["scenarios"][:9])


def test_d02_workflow_plans_rejected_at_adapter_and_replay_is_deterministic():
    r = audit_d02(DATA)
    assert r == audit_d02(DATA)
    assert r["event_count"] == 152 and r["scenario_count"] == 22
    assert sum(len(x["attempts"]) for x in r["scenarios"]) >= 22
    assert all(x["release_count"] == 0 for x in r["scenarios"])
    assert all(not attempt["passed"] and attempt["stage"] == "adapter"
               for row in r["scenarios"] for attempt in row["attempts"])


def test_sources_and_escaped_paths_fail_closed(tmp_path):
    with pytest.raises(ReviewError, match="SOURCE_PATH_INVALID"):
        source_path(DATA, "../outside-file")
    folder = tmp_path/"场站联动模拟数据"
    folder.mkdir()
    (folder/"forecast.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ReviewError, match="SOURCE_FILE_CHANGED"):
        check_sources({"source_refs": {"forecast.csv": "0"*64}}, tmp_path)


@pytest.mark.parametrize("raw,code", [('{"a":1,"a":2}', "DUPLICATE_JSON_KEY"), ('{"a":NaN}', "NONFINITE_NUMBER")])
def test_strict_json_transport(tmp_path, raw, code):
    path = tmp_path/"request.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ReviewError, match=code):
        load_json(path)


def test_cli_json_pass_reject_and_parse_error(tmp_path):
    b, s, p = fixture()
    request = {"frozen_plan": b.to_dict(), "latest_state": s, "governance_policy": p, "as_of": s["available_at"]}
    inp, out = tmp_path/"request.json", tmp_path/"result.json"

    def invoke(expected):
        r = subprocess.run([sys.executable, "-X", "utf8", "-m", "station_review", "review", "--request", str(inp), "--output", str(out)],
                           cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=20)
        assert r.returncode == expected, r.stderr
        return load_json(out)

    inp.write_text(json.dumps(request), encoding="utf-8")
    assert invoke(0)["passed"]
    s["measurement"]["pv_available_kw"] = 0
    inp.write_text(json.dumps(request), encoding="utf-8")
    rejected = invoke(2)
    assert not rejected["passed"]
    assert {x["code"] for x in rejected["violations"]} >= {"PV_LIMIT", "APPROVAL_BINDING_MISMATCH"}
    inp.write_text('{"frozen_plan":NaN}', encoding="utf-8")
    assert not invoke(1)["passed"]


def test_reviewer_imports_do_not_load_optimizer_or_platform():
    script = "import sys,station_review; print(any(k.split('.')[0] in {'ortools','station_joint','station_platform','numpy'} for k in sys.modules))"
    r = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert r.returncode == 0 and r.stdout.strip() == "False"


def test_executed_zero_power_connection_mapping_cannot_be_rewritten():
    b, s, _ = partial_fixture()
    b = b.to_dict()
    # Even a zero-power plugged interval represents an occupied physical port.
    b["plan"]["assignments"][0]["connected_slots"] = [0, 1]
    from station_review.contract import content_hash
    b["plan"]["fingerprint"] = content_hash(b["plan"])
    b["plan_fingerprint"] = b["plan"]["fingerprint"]
    b["fingerprint"] = content_hash(b)
    p = synthetic_policy(b, s, s["available_at"])
    r = review_plan(b, s, p, as_of=s["available_at"])
    assert "EXECUTED_CONNECTION_CHANGED" in {x["code"] for x in r["violations"]}


def test_snapshot_hash_change_even_if_numeric_capabilities_identical():
    b, s, policy = fixture()
    original = digest(s)
    s["state_id"] = "new-state-same-values"
    r = review_plan(b, s, policy, as_of=s["available_at"], expected_state_hash=original)
    assert not r["passed"]
    assert "STATE_CHANGED_SINCE_REVIEW" in {x["code"] for x in r["violations"]}


@pytest.mark.parametrize("malformed", ["O1-SIM", True, {"O1-SIM": True}])
def test_malformed_publisher_policy_never_grants_access(malformed):
    b, s, p = fixture()
    p["authorized_publishers"] = malformed
    assert not review_plan(b, s, p, as_of=s["available_at"])["passed"]
