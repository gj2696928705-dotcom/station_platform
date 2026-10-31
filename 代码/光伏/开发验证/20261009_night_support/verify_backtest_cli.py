"""Verify real CLI dispatch against the frozen backtest API on a small slice."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

out = Path(__file__).resolve().parent
pv = out.parents[1]
sys.path.insert(0, str(pv))
from pv_forecast.backtesting import backtest_bundles

env = os.environ.copy()
env.update(TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1")
os.environ.update(TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1")
source = out / "backtest_cli_input.csv"
full_source = out / "frozen_2026_01" / "黄花站_2026年_15分钟_冻结评估模拟数据.csv"
frame = pd.read_csv(full_source).iloc[:8*96]
frame.to_csv(source, index=False, encoding="utf-8-sig")
bundles = [pv / "运行结果" / name for name in ("20260929_000009_045346", "20261009_night_zero_lower_01")]
report_path = out / "backtest_cli_report.json"
command = [sys.executable, "-X", "utf8", "-m", "pv_forecast.cli", "backtest", "--input", str(source),
           "--bundle-dir", *map(str, bundles), "--output", str(report_path), "--batch-windows", "512"]
result = subprocess.run(command, cwd=pv, env=env, capture_output=True, encoding="utf-8")
(out / "backtest_cli_stderr.log").write_text(result.stderr, encoding="utf-8")
assert result.returncode == 0, result.stderr
stdout = json.loads(result.stdout)
report = json.loads(report_path.read_text(encoding="utf-8"))
assert stdout == report
expected = backtest_bundles(bundles, source, batch_windows=73)
assert report["evaluated_windows"] == len(frame)-288-96+1 == 385
assert report["skipped_origins"] == 0
for name, model in report["models"].items():
    for period, actual in model["overall"].items():
        for metric, value in actual.items():
            reference = expected["models"][name]["overall"][period][metric]
            if value is None:
                assert reference is None
            else:
                np.testing.assert_allclose(value, reference, rtol=1e-5, atol=1e-5)
verification = {"status": "passed", "exit_code": result.returncode,
    "scope": "8-day simulated slice verifies CLI dispatch; full-year evidence is separate",
    "evaluated_windows": report["evaluated_windows"], "stdout_matches_saved_report": True,
    "matches_API_different_batch_size": True,
    "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "report_ref": report_path.name, "stderr_ref": "backtest_cli_stderr.log"}
(out / "backtest_cli_verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(verification, indent=2))
