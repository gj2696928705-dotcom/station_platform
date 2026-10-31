"""Finalize evidence after the full implementation and evaluation checks pass."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import re
import sys

out = Path(__file__).resolve().parent
pv = out.parents[1]
parent = pv/"运行结果"/"20260929_000009_045346"
candidate = pv/"运行结果"/"20261009_point_correction_01"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


raw = (out/"unit_tests.log").read_bytes()
log = raw.decode("utf-16" if raw.startswith(b"\xff\xfe") else "utf-8-sig")
match = re.search(r"Ran (\d+) tests in ([\d.]+)s\s+OK\s*$", log)
assert match and int(match[1]) == 53, "all 53 current regression tests must pass"
tests = {"total_passed": 53, "failed": 0, "seconds": float(match[2]),
    "command": "python -X utf8 -m unittest discover -s tests -v",
    "log_ref": "开发验证/20261009_point_correction/unit_tests.log", "log_sha256": sha(out/"unit_tests.log")}
audit = read(out/"evidence_audit.json")
assert audit["status"] == "passed" and read(out/"candidate_cli_verification.json")["status"] == "passed"
assert read(out/"development_completion.json")["status"] == "completed"
frozen = out/"frozen_2026_seed20261010"
assert read(frozen/"completion.json")["status"] == "completed"
protocol = read(frozen/"protocol.json")
assert all(sha(pv/name) == value for name, value in protocol["source_sha256"].items())
assert all(values[metric]["improved"] for values in audit["new_path_accuracy_changes"].values() for metric in ("rmse", "mae"))
assert audit["months_with_daylight_improvement"] == 12
verification = {"status": "passed", "verified_on": "2026-10-09",
    "scope": "PV standalone algorithm development and simulated candidate validation",
    "unit_tests": tests, "source_file_sha256": {p.relative_to(pv).as_posix(): sha(p)
        for p in [*sorted((pv/"pv_forecast").glob("*.py")), *sorted((pv/"tests").glob("*.py"))]},
    "original_training_audit_ref": "运行结果/20260929_000009_045346/verification/artifact_audit.json",
    "full_correction_report_ref": "运行结果/20261009_point_correction_01/point_correction_report.json",
    "candidate_cli_ref": "candidate_cli_verification.json", "evidence_audit_ref": "evidence_audit.json",
    "new_frozen_evaluation_ref": "frozen_2026_seed20261010/backtest.json",
    "visual_review": {"status": "reviewed", "png_count": 4, "contact_sheet": "candidate_png_review.png",
                      "metric_charts_added": False},
    "reused_parent_GRU_artifacts": 9, "final_correction_training_windows": 24145,
    "retrospective_test_windows": 5161, "frozen_new_path_windows": 34657,
    "GRU_weights_retrained": False, "new_ridge_parameters_fitted": True,
    "model_acceptance": {"status": "candidate_only", "production_accepted": False,
        "known_limits": ["Clear and partly-cloudy daylight RMSE worsened on the new path despite lower aggregate error.",
                        "Clear-weather coverage remains 49.36%; August, September and October daylight coverage is below 80%.",
                        "Night point and zero-width interval assume zero PV power; not validated against real measurements.",
                        "Simulator paths and overlapping windows cannot establish real-site accuracy or future coverage."],
        "future_site_application_requires": ["real power data and issued forecast archive", "site coordinates and power semantics",
                                              "real-site accuracy/reliability review and production publication process"]}}
assert (out/verification["visual_review"]["contact_sheet"]).exists()
write(out/"verification.json", verification)
write(candidate/"verification"/"verification.json", {"status": "passed", "publication_status": "candidate",
    "development_verification_ref": "../../../开发验证/20261009_point_correction/verification.json",
    "unit_tests": tests, "source_file_sha256": verification["source_file_sha256"],
    "model_acceptance": verification["model_acceptance"]})
versions = {}
for name in ("numpy", "pandas", "scikit-learn", "tensorflow", "keras", "matplotlib", "joblib", "jsonschema", "pillow"):
    for distribution in ((name, "tensorflow-cpu", "tensorflow_cpu") if name == "tensorflow" else (name,)):
        try:
            versions[name] = {"distribution": distribution, "version": importlib.metadata.version(distribution)}
            break
        except importlib.metadata.PackageNotFoundError:
            continue
write(out/"runtime_environment.json", {"python": sys.version, "executable": sys.executable,
                                     "platform": platform.platform(), "versions": versions})
implementation_path = parent/"implementation_verification.json"
backup = parent/"implementation_verification_pre_20261009_point_correction.json"
if not backup.exists():
    backup.write_bytes(implementation_path.read_bytes())
implementation = read(implementation_path)
implementation.setdefault("verification_history", []).append({"unit_tests": implementation["unit_tests"],
    "current_source_verification_ref": implementation.get("current_source_verification_ref"), "backup_ref": backup.name})
implementation.update(unit_tests=tests, full_run_status="completed", verified_on="2026-10-09",
    current_source_verification_ref="../../开发验证/20261009_point_correction/verification.json",
    scope=verification["scope"], model_acceptance=verification["model_acceptance"])
implementation.setdefault("source_changes_after_training_started", []).append(
    "Optional causal point correction using all training windows; original GRU weights, scalers and selected configuration retained")
write(implementation_path, implementation)
print(json.dumps({"status": "passed", "unit_tests": tests, "candidate_only": True}, indent=2))
