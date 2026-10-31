"""Run from 代码/场站光储充联合优化: python -X utf8 开发验证/collect_evidence.py."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from station_joint import solve, verify_plan, simulation_gate
from station_joint.contract import fingerprint
import ortools

OUT = Path(__file__).resolve().parent


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8-sig"))


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def main():
    d = read("hand_input.json")
    p = read("hand_result.json")["joint"]["plan"]
    rejection = []
    for field in ("pv_used_kw", "net_import_kw", "storage_charge_kw", "storage_discharge_kw", "storage_energy_kwh"):
        tampered = deepcopy(p)
        tampered[field][0] += 1
        rejection.append({"case": f"tamper_{field}", "check": verify_plan(d, tampered)})
        tampered["fingerprint"] = fingerprint({k: v for k, v in tampered.items() if k != "fingerprint"})
        rejection.append({"case": f"resigned_tamper_{field}", "check": verify_plan(d, tampered)})
    live = deepcopy(d)
    live["grid_import_limit_kw"] = [0]*4
    rejection.append({"case": "live_grid_limit_drop", "check": verify_plan(d, p, live_input=live)})
    pv_input = deepcopy(d)
    pv_input["pv_available_kw"] = [4]*4
    pv_plan = solve(pv_input)["plan"]
    live_pv = deepcopy(pv_input)
    live_pv["pv_available_kw"] = [0]*4
    rejection.append({"case": "live_pv_drop", "check": verify_plan(pv_input, pv_plan, live_input=live_pv)})
    assert all(not item["check"]["passed"] for item in rejection)
    write("rejections.json", rejection)
    governed = deepcopy(d)
    governed["assumptions"] = ["acceptance-only hypothetical assumption"]
    governed_plan = solve(governed)["plan"]
    gates = []
    for numeric_ok in (False, True):
        for governance_ok in (False, True):
            proposed = deepcopy(governed_plan)
            if not numeric_ok:
                proposed["net_import_kw"][0] += 1
                proposed["fingerprint"] = fingerprint({k: v for k, v in proposed.items() if k != "fingerprint"})
            gate = simulation_gate(governed, proposed, allow_assumptions=governance_ok)
            assert gate["eligible_for_offline_simulation"] == (numeric_ok and governance_ok)
            gates.append({"numeric_expected": numeric_ok, "governance_expected": governance_ok, "gate": gate})
    write("double_gate.json", gates)
    replay = read("replay_demo.json")
    summaries = []
    for row in replay["scenarios"]:
        joint, idle = row.get("joint", {}), row.get("idle", {})
        delta = row.get("delta_joint_minus_idle") or {}
        summaries.append({"scenario_id": row["scenario_id"], "joint_status": row.get("status", joint.get("status")),
            "idle_status": idle.get("status"), "reason": row.get("reason"),
            "joint_objective_cny": joint.get("objective", {}).get("total_objective_cny"),
            "idle_objective_cny": idle.get("objective", {}).get("total_objective_cny"),
            **{f"delta_{k}": delta.get(k) for k in ("purchase_cost_cny", "total_objective_cny", "peak_import_kw",
                "pv_curtailed_kwh", "storage_throughput_kwh", "storage_loss_kwh", "storage_terminal_kwh")}})
    with (OUT / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    tests = {}
    for name in ("tests.xml", "regression.xml"):
        xml = ET.parse(OUT/name).getroot()
        suites = [xml] if xml.tag == "testsuite" else list(xml)
        tests[name] = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    paths = [*sorted((ROOT / "station_joint").glob("*.py")),
             ROOT/"tests/test_joint.py", ROOT/"README.md", ROOT/"运行.py",
             ROOT/"pyproject.toml", ROOT/"requirements.txt", ROOT/"requirements-dev.txt",
             *[OUT/name for name in ("collect_evidence.py", "验收记录.md", "hand_input.json", "hand_result.json", "replay_strict.json",
                 "replay_demo.json", "timed_stress.json", "rejections.json", "double_gate.json", "comparison.csv", "tests.xml", "regression.xml")]]
    manifest = {"date": "2026-10-29", "simulation": True, "python": sys.version,
        "algorithm_directory": "代码/场站光储充联合优化", "package": "station_joint",
        "test_evidence_scope": {"tests.xml": "standalone_algorithm", "regression.xml": "combined_platform_regression; run stage recorded in 验收记录.md"},
        "ortools": ortools.__version__, "tests": tests, "replay_source_refs": replay["source_refs"],
        "replay_optimal_count": sum(r.get("joint", {}).get("status") == "OPTIMAL" for r in replay["scenarios"]),
        "replay_input_unavailable_count": sum(r.get("status") == "INPUT_UNAVAILABLE" for r in replay["scenarios"]),
        "timed_stress_status": read("timed_stress.json")["result"]["status"],
        "sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    write("verification_manifest.json", manifest)
    print(json.dumps({"tests": tests, "replay_optimal_count": manifest["replay_optimal_count"],
                      "rejections": len(rejection), "stress_status": manifest["timed_stress_status"]}))


if __name__ == "__main__":
    main()
