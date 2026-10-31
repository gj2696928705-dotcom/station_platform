"""Run full-window point correction after freezing the development protocol."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

out = Path(__file__).resolve().parent
pv = out.parents[1]
parent = pv/"运行结果"/"20260929_000009_045346"
candidate = pv/"运行结果"/"20261009_point_correction_01"
data = pv.parent/"数据"/"光伏数据"/"黄花站_2025年_15分钟光伏模拟数据.csv"
protocol_path = out/"development_protocol.json"
assert not candidate.exists() and not protocol_path.exists(), "inspect any previous/live experiment; new paths required"
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
protocol = {"status": "frozen_before_correction_fit", "operation": "frozen_GRU_causal_point_correction",
    "input_sha256": sha(data), "parent_sha256": {name: sha(parent/name) for name in
        ("model_metadata.json", "gru.keras", "scalers.joblib", "rolling_validation.json")},
    "source_sha256": {p.relative_to(pv).as_posix(): sha(p) for p in sorted((pv/"pv_forecast").glob("*.py"))},
    "alphas": [0.001, 0.01, 0.1], "test_used_for_selection": False,
    "GRU_weights_retrained": False, "fit_point_correction_on": "all parent training windows, daytime targets",
    "GRU_configuration_frozen": "gru32_scheduled", "parent_GRU_evidence_required": "all nine full training artifacts",
    "selection": "pooled rolling all-day RMSE with daylight and worst-fold improvement guards",
    "test_scope": "2025 retrospective; a separate unseen simulated path is required after freezing the final candidate"}
protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
env = os.environ.copy()
env.update(TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
command = [sys.executable, "-X", "utf8", "-m", "pv_forecast.cli", "correct-point", "--input", str(data),
           "--bundle-dir", str(parent), "--output-dir", str(candidate), "--batch-windows", "1024"]
with (out/"full_correction_cli.json").open("w", encoding="utf-8") as stdout, (out/"full_correction_stderr.log").open("w", encoding="utf-8") as stderr:
    result = subprocess.run(command, cwd=pv, env=env, stdout=stdout, stderr=stderr)
assert result.returncode == 0, f"CLI returned {result.returncode}; inspect full_correction_stderr.log"
report = json.loads((candidate/"point_correction_report.json").read_text(encoding="utf-8"))
assert report["status"] == "completed" and report["parent_full_gru_artifact_count"] == 9
assert report["final_correction_training_windows"] == 24145
assert all(sha(pv/name) == expected for name, expected in protocol["source_sha256"].items())
assert sha(data) == protocol["input_sha256"]
assert all(sha(parent/name) == expected for name, expected in protocol["parent_sha256"].items())
protocol.update(status="completed", selected_correction=report["selected_correction"],
    report_ref=str((candidate/"point_correction_report.json").relative_to(pv)),
    test_before=report["test_scores"]["before"], test_after=report["test_scores"]["after"])
(out/"development_completion.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"status": "completed", "selected_correction": report["selected_correction"],
    "aggregate": report["aggregate"], "test_before": report["test_scores"]["before"],
    "test_after": report["test_scores"]["after"]}, indent=2))
