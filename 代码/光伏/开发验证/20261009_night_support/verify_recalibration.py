"""Verify the derived candidate with real CLI subprocesses and frozen hashes."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
import numpy as np
import pandas as pd
from PIL import Image

pv = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pv))
from pv_forecast.pipeline import predict_record
from pv_forecast.records import validate_forecast_record
from pv_forecast.weather import solar_elevation


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--bundle-dir", required=True)
args = parser.parse_args()
bundle = Path(args.bundle_dir).resolve()
meta, report = read(bundle / "model_metadata.json"), read(bundle / "recalibration_report.json")
parent = Path(report["parent_bundle"])
source = pv.parent / "数据" / "光伏数据" / "黄花站_2025年_15分钟光伏模拟数据.csv"
out = bundle / "verification"
out.mkdir(exist_ok=True)
assert meta["selected_model"] == read(parent / "model_metadata.json")["selected_model"]
assert meta["publication_status"] == "candidate" and meta["model_version"] != report["parent_model_version"]
assert not report["weights_retrained"] and not report["test_used_for_calibration"]
assert not report["test_used_for_model_selection"]
assert report["source_data_sha256"] == digest(source)
for name, expected in report["parent_bundle_sha256"].items():
    assert digest(parent / name) == expected
    if name in ("gru.keras", "scalers.joblib"):
        assert digest(bundle / name) == expected
old_metrics = read(parent / "metrics.json")
for partition in ("validation", "test"):
    for period, key in (("all_day", partition), ("daylight", partition + "_daylight")):
        before, after = [report["metrics"][partition][stage][period] for stage in ("before", "after")]
        for metric in ("rmse", "mae", "r2", "count"):
            np.testing.assert_allclose(before[metric], old_metrics[key][metric], rtol=1e-5, atol=.01)
            np.testing.assert_allclose(before[metric], after[metric], rtol=0, atol=0)
        if old_metrics[key]["coverage"] is not None:
            np.testing.assert_allclose(before["coverage"], old_metrics[key]["coverage"], rtol=0, atol=1e-5)


def check(record):
    validate_forecast_record(record)
    assert record["model_version"] == meta["model_version"]
    pred = np.array([v["value"] for v in record["forecast_values"]])
    lo = np.array([v["lower"] for v in record["prediction_interval"]["values"]])
    hi = np.array([v["upper"] for v in record["prediction_interval"]["values"]])
    assert pred.shape == lo.shape == hi.shape == (96,)
    assert np.isfinite([pred, lo, hi]).all() and ((0 <= lo) & (lo <= pred) & (pred <= hi)).all()
    timestamps = pd.DatetimeIndex([v["timestamp"] for v in record["forecast_values"]])
    assert np.all(np.diff(timestamps.asi8) == pd.Timedelta(minutes=15).value)
    daylight = solar_elevation(timestamps, meta["latitude"], meta["longitude"]) > 0
    calibration = meta["baseline_fallback"]["calibration"] if record["status"] == "degraded" else meta["calibration"]
    groups = calibration["groups"]
    widths = np.where(daylight, groups["daylight"]["quantiles_kw"], groups["night"]["quantiles_kw"])
    np.testing.assert_allclose(hi, pred + widths)
    expected_lower = np.maximum(pred - widths, 0)
    if calibration.get("night_lower_bound") == "zero":
        expected_lower = np.where(daylight, expected_lower, 0.0)
    np.testing.assert_allclose(lo, expected_lower)
    assert record["confidence"]["level"] == calibration["nominal_coverage"]
    return pred


commands = {}
for action in ("predict", "evaluate"):
    output = bundle / ("forecast.json" if action == "predict" else "evaluation.json")
    cmd = [sys.executable, "-X", "utf8", "-m", "pv_forecast.cli", action,
           "--input", str(source), "--bundle-dir", str(bundle), "--output", str(output)]
    result = subprocess.run(cmd, cwd=pv, capture_output=True, text=True, encoding="utf-8", check=True)
    (out / (action + "_stderr.log")).write_text(result.stderr, encoding="utf-8")
    record = read(output)
    assert record["status"] == "available" and not record["visualization_errors"]
    assert len(record["visualizations"]) == (3 if action == "predict" else 1)
    for ref in record["visualizations"]:
        path = (bundle / ref["path"]).resolve()
        assert path.is_relative_to(bundle) and path.stat().st_size > 0
        with Image.open(path) as png:
            assert png.format == "PNG"
            png.verify()
    if action == "predict":
        prediction = check(record)
        original = np.array([v["value"] for v in read(parent / "forecast.json")["forecast_values"]])
        np.testing.assert_allclose(prediction, original, rtol=1e-5, atol=.01)
    else:
        for metric in ("rmse", "mae", "r2"):
            np.testing.assert_allclose(record["metrics"][metric], read(parent / "evaluation.json")["metrics"][metric],
                                       rtol=1e-5, atol=.01)
    commands[action] = {"exit_code": result.returncode, "png_count": len(record["visualizations"])}

raw = pd.read_csv(source)
with tempfile.TemporaryDirectory(dir=out) as temp:
    bad = Path(temp)
    (bad / "model_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    shutil.copy2(bundle / "scalers.joblib", bad / "scalers.joblib")
    missing = predict_record(bad, raw)
    assert missing["status"] == "degraded"
    check(missing)
    shutil.copy2(bundle / "gru.keras", bad / "gru.keras")
    missing_weather = predict_record(bad, raw.drop(columns=meta["history_weather_columns"]))
    assert missing_weather["status"] == "degraded"
    check(missing_weather)
    for record in (missing, missing_weather):
        assert record["fallback_forecast_ref"] == "baseline:" + meta["baseline_fallback"]["strategy"]
    assert predict_record(bad, raw.tail(10))["status"] == "unavailable"
    assert predict_record(bad, raw, forecast_start="2026-01-02")["status"] == "unavailable"
result = {"status": "passed", "commands": commands, "data_kind": "simulated",
          "parent_hashes_unchanged": True, "weights_and_scalers_identical": True,
          "point_predictions_and_metrics_unchanged": True, "full_window_counts_checked": True,
          "conditioned_primary_and_fallback_intervals_checked": True,
          "missing_weight_and_weather_fallback_checked": True,
          "insufficient_and_stale_history_checked": True,
          "test_scope": "retrospective; not independent production acceptance"}
(out / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
