"""Freeze the full experiment, execute CLI, preserve logs and parent hashes."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

out = Path(__file__).resolve().parent
pv = out.parents[1]
data_dir = pv.parent/"数据"/"光伏数据"
source = data_dir/"黄花站_2025年_15分钟光伏模拟数据.csv"
control = pv/"运行结果"/"20261009_point_correction_01"
destination = pv/"运行结果"/"20261014_physics_comparison_01"
protocol_path = out/"development_protocol.json"
assert not destination.exists() and not protocol_path.exists(), "inspect any existing/live full experiment; never overwrite or restart blindly"
scratch = (out/"runtime_tmp").resolve()
assert scratch.is_relative_to(pv.resolve())
scratch.mkdir(exist_ok=True)
env = os.environ.copy()
env.update(TEMP=str(scratch), TMP=str(scratch), TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1",
           OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
protocol = {"status": "frozen_before_full_training", "simulated_data_only": True,
    "source_sha256": sha(source), "physics_config_sha256": sha(data_dir/"simulation_config.json"),
    "generator_sha256": sha(data_dir/"generate_huanghua_pv.py"),
    "implementation_sha256": {p.relative_to(pv).as_posix(): sha(p) for p in sorted((pv/"pv_forecast").glob("*.py"))},
    "control_sha256": {name: sha(control/name) for name in ("model_metadata.json", "gru.keras", "scalers.joblib", "point_correction.joblib")},
    "physics_GRU_training": "3 chronological folds x 3 complete GRU configurations, then best configuration final complete fit",
    "training_sample_cap": None, "tree": {"iterations": 200, "learning_rate": .05, "max_leaf_nodes": 31, "l2_regularization": 10},
    "oracle_future_weather_used_for_selection": False, "selection": "pooled rolling daylight RMSE, then MAE; report worst fold/weather",
    "calibration": "final validation second half; excluded from final early stopping",
    "unseen_evaluation_year": 2026, "unseen_evaluation_seeds": [2026101401, 2026101402, 2026101403],
    "final_acceptance": "compare frozen candidates with current control on every complete origin; point accuracy separate from interval coverage"}
protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
command = [sys.executable, "-X", "utf8", "-m", "pv_forecast.cli", "compare-physics", "--input", str(source),
           "--control-bundle", str(control), "--physics-config", str(data_dir/"simulation_config.json"),
           "--output-dir", str(destination), "--batch-windows", "1024"]
with (out/"comparison_cli_stdout.json").open("w", encoding="utf-8") as stdout, (out/"comparison_cli_stderr.log").open("w", encoding="utf-8") as stderr:
    result = subprocess.run(command, cwd=pv, env=env, stdout=stdout, stderr=stderr)
assert result.returncode == 0, f"comparison failed with exit {result.returncode}; inspect logs and existing artifacts"
report = json.loads((destination/"comparison_report.json").read_text(encoding="utf-8"))
assert report["status"] == "completed" and len(report["folds"]) == 3
assert report["final_training_windows"] == 24145 and report["retrospective_test_windows"] == 5161
assert sha(source) == protocol["source_sha256"]
assert all(sha(control/name) == value for name, value in protocol["control_sha256"].items())
protocol.update(status="completed", report_ref=str((destination/"comparison_report.json").relative_to(pv)),
    selected_candidate=report["selected_candidate"], best_enhanced_gru=report["best_enhanced_gru"])
(out/"development_completion.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"status": "completed", "selected": report["selected_candidate"], "aggregate": report["aggregate"]}, indent=2))
