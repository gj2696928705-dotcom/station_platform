"""Save current evidence after all tests, CLI and frozen checks actually pass."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import re
import sys

out = Path(__file__).resolve().parent
pv = out.parents[1]
parent = pv / "运行结果" / "20260929_000009_045346"
child = pv / "运行结果" / "20261009_night_zero_lower_01"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def installed_version(name):
    for distribution in ((name, "tensorflow-cpu", "tensorflow_cpu") if name == "tensorflow" else (name,)):
        try:
            return {"distribution": distribution, "version": importlib.metadata.version(distribution)}
        except importlib.metadata.PackageNotFoundError:
            continue
    return {"distribution": None, "version": None}


raw = (out / "unit_tests.log").read_bytes()
log = raw.decode("utf-16" if raw.startswith(b"\xff\xfe") else "utf-8-sig")
match = re.search(r"Ran (\d+) tests in ([\d.]+)s\s+OK\s*$", log)
assert match and int(match[1]) == 48, "all 48 full regression tests must pass"
tests = {"total_passed": int(match[1]), "seconds": float(match[2]), "failed": 0,
    "command": "python -X utf8 -m unittest discover -s tests -v",
    "log_ref": "开发验证/20261009_night_support/unit_tests.log", "log_sha256": digest(out / "unit_tests.log")}
for evidence in (parent / "verification" / "artifact_audit.json",
                 parent / "verification" / "final_cli" / "verification.json",
                 child / "verification" / "verification.json",
                 out / "backtest_cli_verification.json", out / "frozen_evaluation_audit.json"):
    assert read(evidence)["status"] == "passed", evidence
protocol = read(out / "frozen_2026_01" / "protocol.json")
assert protocol["status"] == "frozen_before_generation"
assert read(out / "frozen_2026_01" / "completion.json")["status"] == "completed"
assert all(digest(pv / name) == expected for name, expected in protocol["source_code_sha256"].items())
data_dir = pv.parent / "数据" / "光伏数据"
assert digest(data_dir / "generate_huanghua_pv.py") == protocol["generator_sha256"]
assert digest(data_dir / "黄花站_2025年_15分钟光伏模拟数据.csv") == protocol["original_data_sha256"]
for bundle, hashes in protocol["bundle_sha256"].items():
    assert all(digest(pv / "运行结果" / bundle / name) == expected for name, expected in hashes.items())
source_files = [*sorted((pv / "pv_forecast").glob("*.py")), *sorted((pv / "tests").glob("*.py"))]
verification = {"status": "passed", "verified_on": "2026-10-09",
    "scope": "PV implementation, simulated candidate artifacts and frozen new random-path evaluation",
    "unit_tests": tests,
    "source_file_sha256": {str(p.relative_to(pv)): digest(p) for p in source_files},
    "frozen_source_matches_current": True,
    "original_training_artifacts_ref": "运行结果/20260929_000009_045346/verification/artifact_audit.json",
    "original_cli_ref": "运行结果/20260929_000009_045346/verification/final_cli/verification.json",
    "night_support_cli_ref": "运行结果/20261009_night_zero_lower_01/verification/verification.json",
    "backtest_cli_ref": "backtest_cli_verification.json",
    "frozen_evaluation_ref": "frozen_2026_01/backtest.json",
    "frozen_protocol_ref": "frozen_2026_01/protocol.json",
    "frozen_audit_ref": "frozen_evaluation_audit.json",
    "full_year_windows": 34657, "full_year_scored_labels": 3327072,
    "visual_review": {"status": "reviewed", "contact_sheet": "night_support_png_review.png",
                      "png_count": 4, "metric_charts_added": False},
    "original_data_and_generator_unchanged": True,
    "original_bundles_weights_scalers_unchanged": True,
    "model_acceptance": {"status": "candidate_only", "production_accepted": False,
        "remaining": ["real-site data and issued meteorological forecast validation",
                      "daylight point forecast error and weather-dependent bias",
                      "daylight interval undercoverage on the new simulated path",
                      "new unused evaluation data for subsequent algorithm changes"],
        "night_zero_coverage_scope": "all-zero nighttime truth of the known simulator only"}}
assert (out / verification["visual_review"]["contact_sheet"]).is_file()
write(out / "verification.json", verification)
write(out / "runtime_environment.json", {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
    "versions": {name: installed_version(name) for name in
                 ("numpy", "pandas", "scikit-learn", "tensorflow", "keras", "matplotlib", "joblib", "jsonschema", "pillow")}})
implementation_path = parent / "implementation_verification.json"
backup = parent / "implementation_verification_pre_20261009_night_support.json"
if not backup.exists():
    backup.write_bytes(implementation_path.read_bytes())
implementation = read(implementation_path)
implementation.setdefault("verification_history", []).append({
    "unit_tests": implementation["unit_tests"],
    "current_source_verification_ref": implementation.get("current_source_verification_ref"),
    "backup_ref": backup.name})
implementation.update(unit_tests=tests, full_run_status="completed", verified_on="2026-10-09",
    scope=verification["scope"], model_acceptance=verification["model_acceptance"],
    current_source_verification_ref="../../开发验证/20261009_night_support/verification.json")
changes = implementation.setdefault("source_changes_after_training_started", [])
note = "Optional night zero lower interval support and frozen full-window backtest; original point predictions unchanged"
if note not in changes:
    changes.append(note)
write(implementation_path, implementation)
print(json.dumps({"status": "passed", "unit_tests": tests, "production_accepted": False}, indent=2))
