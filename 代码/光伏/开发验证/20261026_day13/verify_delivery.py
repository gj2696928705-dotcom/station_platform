"""Inspect actual full training, CLI IO, physical units and preserved inputs."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile

out = Path(__file__).resolve().parent
pv = out.parents[1]
workspace = pv.parents[1]
scratch = out/"runtime_tmp"
scratch.mkdir(exist_ok=True)
os.environ.update(TEMP=str(scratch), TMP=str(scratch), TF_NUM_INTRAOP_THREADS="1",
    TF_NUM_INTEROP_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", TF_CPP_MIN_LOG_LEVEL="2")
tempfile.tempdir = str(scratch)
sys.path.insert(0, str(pv))
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pv_forecast.day13_data import read_day_frame, build_day_samples
from pv_forecast.models import GRUForecaster
from pv_forecast.pipeline import predict_record

read = lambda p: json.loads(p.read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
bundle = pv/"运行结果"/"20261026_day13_01"
data_dir = pv.parent/"数据"/"光伏数据"
source = data_dir/"黄花站_2025年_07-19_小时数据.csv"
original = data_dir/"黄花站_2025年_15分钟光伏模拟数据.csv"
assert sha(original) == "28a0842c8b7c28ac5e90ae209bba2a5a215ef1f48ae3f25d616d083515bed1c8"
previous = read(pv/"开发验证"/"20261014_physics_01"/"development_protocol.json")
for name, value in previous["control_sha256"].items():
    assert sha(pv/"运行结果"/"20261009_point_correction_01"/name) == value
assert sha(data_dir/"generate_huanghua_pv.py") == previous["generator_sha256"]
assert sha(data_dir/"simulation_config.json") == previous["physics_config_sha256"]
meta, metrics, summary = [read(bundle/name) for name in ("model_metadata.json", "metrics.json", "training_summary.json")]
before = {name: sha(bundle/name) for name in ("gru.keras", "scalers.joblib", "model_metadata.json", "interval_calibration.json")}
assert (meta["interval_minutes"], meta["history_steps"], meta["horizon_steps"], meta["points_per_day"]) == (60, 39, 13, 13)
assert meta["history_feature_names"] == ["pv_power_kw"] and meta["feature_names"] == ["SWGDN", "DHI", "DNI", "rain"]
assert summary["training_windows_used"] == 252 and 1 <= summary["epochs_completed"] <= 50
history = read(bundle/"training_history.json")
assert summary["best_epoch"] == min(history, key=lambda row: row["val_loss"])["epoch"]
assert summary["epochs_requested"] == 50
scalers = joblib.load(bundle/"scalers.joblib")
for name, expected in (("target_scaler", 252*13), ("history_scaler", 252*39), ("feature_scaler", 252*13)):
    assert scalers[name].n_samples_seen_ == expected
model = GRUForecaster.load(str(bundle/"gru.keras"))
assert model.model.input_shape == [(None, 39, 1), (None, 13, 4)] and model.model.output_shape == (None, 13, 1)
windows = build_day_samples(read_day_frame(source))
assert windows.size == 362 and windows.history_power.shape == (362, 39, 1)
frame = pd.read_csv(source)
assert len(frame) == 4745
test_predictions = pd.read_csv(bundle/"test_predictions.csv")
assert len(test_predictions) == 55*13
independent = {"rmse": float(np.sqrt(mean_squared_error(test_predictions.actual_kw, test_predictions.forecast_kw))),
    "mae": mean_absolute_error(test_predictions.actual_kw, test_predictions.forecast_kw),
    "r2": r2_score(test_predictions.actual_kw, test_predictions.forecast_kw)}
for key, value in independent.items():
    np.testing.assert_allclose(value, metrics["test"][key], atol=1e-9, rtol=0)

cli_dir = out/"cli"
cli_dir.mkdir(exist_ok=True)
commands = {
    "predict": ["predict", "--input", str(source), "--bundle-dir", str(bundle), "--forecast-start", "2025-12-31", "--output", str(cli_dir/"forecast.json")],
    "evaluate": ["evaluate", "--input", str(source), "--bundle-dir", str(bundle), "--output", str(cli_dir/"evaluation.json")],
    "backtest": ["backtest", "--input", str(source), "--bundle-dir", str(bundle), "--output", str(cli_dir/"backtest.json")],
}
for name, args in commands.items():
    with (out/(name+"_cli_stdout.log")).open("w", encoding="utf-8") as stdout, (out/(name+"_cli_stderr.log")).open("w", encoding="utf-8") as stderr:
        proc = subprocess.run([sys.executable, "-X", "utf8", "-m", "pv_forecast.cli", *args], cwd=pv,
                              stdout=stdout, stderr=stderr)
    assert proc.returncode == 0, name
record = read(cli_dir/"forecast.json")
assert record["status"] == "available" and len(record["forecast_values"]) == 13
assert [pd.Timestamp(v["timestamp"]).hour for v in record["forecast_values"]] == list(range(7, 20))
assert record["forecast_start_at"] == "2025-12-31T07:00:00+08:00"
assert record["forecast_end_at"] == "2025-12-31T19:00:00+08:00"
assert all(v["unit"] == "kW" for v in record["forecast_values"])
direct = predict_record(bundle, source, forecast_start="2025-12-31")
assert record["forecast_values"] == direct["forecast_values"]
bad_future = frame.copy()
stamp = pd.to_datetime(bad_future.timestamp)
bad_future.loc[stamp.dt.dayofyear >= 365, "pv_power_kw"] = -123456789.
changed = predict_record(bundle, bad_future, forecast_start="2025-12-31")
assert record["forecast_values"] == changed["forecast_values"]
assert record["prediction_interval"] == changed["prediction_interval"]
evaluation = read(cli_dir/"evaluation.json")
assert evaluation["evaluated_days"] == 362 and evaluation["evaluated_points"] == 4706
assert read(cli_dir/"backtest.json")["evaluated_windows"] == 362
assert {name: sha(bundle/name) for name in before} == before
assert not record["visualization_errors"] and not evaluation["visualization_errors"] and not metrics["visualization_errors"]
pngs = sorted((cli_dir/"visualizations").glob("*.png")) + [bundle/"visualizations"/"training_loss.png", bundle/"visualizations"/"gru_test_comparison.png"]
assert len(pngs) == 6 and any("forecast_day13" in p.name for p in pngs)
canvas = Image.new("RGB", (1600, 1260), "white")
draw = ImageDraw.Draw(canvas)
for i, png in enumerate(pngs):
    with Image.open(png) as picture:
        picture.verify()
    with Image.open(png) as picture:
        picture.thumbnail((780, 380), Image.Resampling.LANCZOS)
        x, y = i%2*800, i//2*420
        canvas.paste(picture, (x+10, y+30))
        draw.text((x+10, y+8), png.stem[-45:], fill="black")
canvas.save(out/"visual_review.png")
result = {"status": "passed", "reference_sha256": sha(workspace/"参考代码文档"/"光伏"/"主代码.py"),
    "original_data_sha256": sha(original), "hourly_data_sha256": sha(source), "hourly_rows": 4745,
    "calendar_days": 365, "daily_samples": 362, "training_samples": 252, "selection_samples": 27,
    "calibration_samples": 28, "test_samples": 55, "history_shape": [39, 1], "future_shape": [13, 4],
    "output_shape": [13], "forecast_hours": list(range(7, 20)), "interval_minutes": 60,
    "epochs_requested": 50, "epochs_completed": summary["epochs_completed"], "best_epoch": summary["best_epoch"],
    "independent_holdout_metrics": independent, "metrics_scope": "known simulated target-day weather; 55 retrospective test days",
    "current_day_default_verified_in_tests": True, "target_power_mutation_ignored": True,
    "real_CLI_predict_evaluate_backtest": True, "saved_reload_matches": True,
    "original_data_generator_config_and_control_unchanged": True, "model_hashes": before,
    "implementation_sha256": {str(p.relative_to(pv)): sha(p) for p in sorted((pv/"pv_forecast").rglob("*.py"))},
    "PNG_count": len(pngs), "PNG_refs": [str(p.relative_to(pv)) for p in pngs]}
(out/"delivery_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({"status": "passed", "metrics": independent, "PNG_count": len(pngs)}, indent=2))
