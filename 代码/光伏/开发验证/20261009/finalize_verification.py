"""Close stale implementation evidence only after current checks pass."""
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
child = pv / "运行结果" / "20261009_daylight_recalibration_01"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def installed_version(name):
    names = (name, "tensorflow-cpu", "tensorflow_cpu") if name == "tensorflow" else (name,)
    for distribution in names:
        try:
            return {"distribution": distribution, "version": importlib.metadata.version(distribution)}
        except importlib.metadata.PackageNotFoundError:
            continue
    return {"distribution": None, "version": None, "status": "package_metadata_unavailable"}


raw_log = (out / "unit_tests.log").read_bytes()
log = raw_log.decode("utf-16" if raw_log.startswith(b"\xff\xfe") else "utf-8-sig")
match = re.search(r"Ran (\d+) tests in ([\d.]+)s\s+OK\s*$", log)
assert match and int(match[1]) == 40, "current full regression has not passed all 40 tests"
tests = {"total_passed": int(match[1]), "seconds": float(match[2]), "failed": 0,
         "command": "python -X utf8 -m unittest discover -s tests -v",
         "log_ref": "开发验证/20261009/unit_tests.log"}
artifact = read(parent / "verification" / "artifact_audit.json")
old_cli = read(parent / "verification" / "final_cli" / "verification.json")
new_cli = read(child / "verification" / "verification.json")
assert artifact["status"] == old_cli["status"] == new_cli["status"] == "passed"
assert len(artifact["training_artifacts"]) == 10
png = read(out / "png_review_sources.json")
assert len(png["rolling_loss"]["source_pngs"]) == 9
assert len(png["recalibrated_inference"]["source_pngs"]) == 4
for group in png.values():
    assert (out / group["sheet"]).is_file()
    assert all((pv / path).is_file() for path in group["source_pngs"])
source_files = [*sorted((pv / "pv_forecast").glob("*.py")), *sorted((pv / "tests").glob("*.py"))]
source_hashes = {str(p.relative_to(pv)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
verification = {"status": "passed", "verified_on": "2026-10-09", "scope": "PV implementation and simulated candidate artifacts",
    "unit_tests": tests, "source_file_sha256": source_hashes,
    "original_training_artifacts": "运行结果/20260929_000009_045346/verification/artifact_audit.json",
    "original_cli_reload_and_fallback": "运行结果/20260929_000009_045346/verification/final_cli/verification.json",
    "derived_cli_reload_hashes_and_fallback": "运行结果/20261009_daylight_recalibration_01/verification/verification.json",
    "visual_review": {"status": "reviewed", "sources_ref": "png_review_sources.json",
                      "notes": "Reviewed nine rolling loss charts, final loss/comparison, old CLI charts, and four derived charts."},
    "regression_evidence": {"before_fix": "prediction_cutoff_before_fix.log", "after_fix": "prediction_cutoff_after_fix.log",
                            "bug": "future rows could fill trailing historical gap and affect input validation/quality"},
    "model_acceptance": {"status": "candidate_only", "production_accepted": False,
        "remaining": ["independent real/unused-data acceptance", "nighttime interval undercoverage", "daylight point forecast accuracy"]}}
write(out / "verification.json", verification)
write(out / "runtime_environment.json", {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
    "versions": {name: installed_version(name) for name in
                 ("numpy", "pandas", "scikit-learn", "tensorflow", "keras", "matplotlib", "joblib", "jsonschema", "pillow")}})
implementation_path = parent / "implementation_verification.json"
old = read(implementation_path)
backup = parent / "implementation_verification_pre_20261009.json"
if not backup.exists():
    backup.write_bytes(implementation_path.read_bytes())
old.setdefault("historical_unit_tests", old["unit_tests"])
old.update(unit_tests=tests, full_run_status="completed", completed_candidate_artifacts_verified=9,
    completed_final_artifacts_verified=1, remaining=[], verified_on="2026-10-09",
    scope=verification["scope"], model_acceptance=verification["model_acceptance"],
    artifact_audit_ref="verification/artifact_audit.json", artifact_audit_status="passed",
    final_cli_verification_ref="verification/final_cli/verification.json", final_cli_status="passed",
    visual_review_ref="../../开发验证/20261009/verification.json",
    current_source_verification_ref="../../开发验证/20261009/verification.json")
old.setdefault("source_changes_after_training_started", []).extend([
    "Optional daylight-conditioned intervals and frozen-weight recalibration; no weight fitting or primary selection change",
    "Forecast-origin truncation before numeric validation, filling and quality summary; clean full-year point predictions unchanged"])
write(implementation_path, old)
print(json.dumps({"status": "passed", "unit_tests": tests, "original_run_status": old["full_run_status"]}, indent=2))
