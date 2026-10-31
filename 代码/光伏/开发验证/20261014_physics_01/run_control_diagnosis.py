"""Re-score the already-inspected control path with fine error and oracle diagnostics."""
from pathlib import Path
import json
import os
import subprocess
import sys

out = Path(__file__).resolve().parent
pv = out.parents[1]
scratch = (out/"runtime_tmp").resolve()
assert scratch.is_relative_to(pv.resolve())
scratch.mkdir(exist_ok=True)
env = os.environ.copy()
env.update(TEMP=str(scratch), TMP=str(scratch), TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1",
           OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
source = pv/"开发验证"/"20261009_point_correction"/"frozen_2026_seed20261010"/"黄花站_2026年_15分钟_点预测修正冻结模拟数据.csv"
report_path = out/"control_error_diagnosis.json"
assert not report_path.exists(), "preserve existing diagnosis"
cmd = [sys.executable, "-X", "utf8", "-m", "pv_forecast.cli", "diagnose", "--input", str(source),
    "--bundle-dir", str(pv/"运行结果"/"20261009_point_correction_01"), "--output", str(report_path),
    "--oracle-physics-config", str(pv.parent/"数据"/"光伏数据"/"simulation_config.json")]
with (out/"control_diagnosis_stdout.json").open("w", encoding="utf-8") as stdout, (out/"control_diagnosis_stderr.log").open("w", encoding="utf-8") as stderr:
    result = subprocess.run(cmd, cwd=pv, env=env, stdout=stdout, stderr=stderr)
assert result.returncode == 0
report = json.loads(report_path.read_text(encoding="utf-8"))
assert report["evaluated_windows"] == 34657 and report["skipped_origins"] == 0
assert not report["ideal_future_weather_diagnostic"]["model_selection_eligible"]
print(json.dumps({"status": "completed", "windows": report["evaluated_windows"],
    "diagnostics": report["models"]["20261009_point_correction_01"]["error_diagnostics"],
    "oracle": report["ideal_future_weather_diagnostic"]}, indent=2))
