"""Verify final saved model, CLI, image references and degradation paths."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
import joblib
import numpy as np
import pandas as pd
from PIL import Image

run = Path(__file__).resolve().parents[1]
pv = run.parent.parent
sys.path.insert(0, str(pv))
from pv_forecast.data import InputSpec, load_and_regularize, build_windows
from pv_forecast.models import GRUForecaster
from pv_forecast.pipeline import _scale_windows, _inverse_target, predict_record
from pv_forecast.records import validate_forecast_record

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def values(record):
    return np.array([item["value"] for item in record["forecast_values"]])

def check_record(record):
    validate_forecast_record(record)
    y = values(record)
    assert y.shape == (96,) and np.isfinite(y).all() and (y >= 0).all()
    points = pd.DatetimeIndex([item["timestamp"] for item in record["forecast_values"]])
    assert np.all(np.diff(points.asi8) == pd.Timedelta(minutes=15).value)
    lo = np.array([item["lower"] for item in record["prediction_interval"]["values"]])
    hi = np.array([item["upper"] for item in record["prediction_interval"]["values"]])
    assert lo.shape == hi.shape == y.shape
    assert np.isfinite(lo).all() and np.isfinite(hi).all()
    assert ((lo >= 0) & (lo <= y) & (y <= hi)).all()
    assert record["model_version"] == "0.2.0-candidate"
    assert record["model_published_at"] is None
    assert record["training_data_window"] and record["input_data_quality_summary"]

assert read(run / "rolling_validation.json")["status"] == "completed"
meta = read(run / "model_metadata.json")
source = pv.parent / "\u6570\u636e" / "\u5149\u4f0f\u6570\u636e" / "\u9ec4\u82b1\u7ad9_2025\u5e74_15\u5206\u949f\u5149\u4f0f\u6a21\u62df\u6570\u636e.csv"
out = Path(__file__).parent / "final_cli"
out.mkdir(exist_ok=True)
results = {"data_kind": "simulated", "commands": {}, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
for action in ("predict", "evaluate"):
    filename = "forecast.json" if action == "predict" else "evaluation.json"
    result = subprocess.run([sys.executable, "-X", "utf8", "-m", "pv_forecast.cli", action,
        "--input", str(source), "--bundle-dir", str(run), "--output", str(out / filename)],
        cwd=pv, capture_output=True, text=True, encoding="utf-8", check=True)
    (out / (action + "_stderr.log")).write_text(result.stderr, encoding="utf-8")
    record = read(out / filename)
    assert record["status"] == "available"
    assert not record["visualization_errors"]
    for ref in record["visualizations"]:
        path = (out / ref["path"]).resolve()
        assert path.is_relative_to(out.resolve()) and path.stat().st_size > 0
        with Image.open(path) as png:
            assert png.format == "PNG"
            png.verify()
    if action == "predict":
        check_record(record)
        original = read(run / filename)
        np.testing.assert_allclose(values(record), values(original), rtol=1e-5, atol=.01)
        results["reload_max_difference_kw"] = float(np.max(abs(values(record)-values(original))))
        assert len(record["visualizations"]) == 3
    else:
        original = read(run / filename)
        for key in ("rmse", "mae", "r2"):
            np.testing.assert_allclose(record["metrics"][key], original["metrics"][key], rtol=1e-5, atol=.01)
        assert len(record["visualizations"]) == 1
    results["commands"][action] = {"exit_code": 0, "status": record["status"],
        "png_ids": [ref["id"] for ref in record["visualizations"]]}

# Independently load the retained GRU, even if the selected primary is baseline.
spec = InputSpec(**meta["input_spec"])
regular = load_and_regularize(source, spec)
windows = build_windows(regular, len(regular.frame)-96, len(regular.frame),
    history_weather_columns=spec.history_weather_columns, latitude=spec.latitude, longitude=spec.longitude)
assert windows.size == 1 and windows.gru_history.shape == (1, 288, 4)
scalers = joblib.load(run / "scalers.joblib")
scaled = _scale_windows(windows, **scalers)
model = GRUForecaster.load(str(run / "gru.keras"))
pred = _inverse_target(model.predict(scaled.gru_history, scaled.future_features), scalers["target_scaler"])
assert pred.shape == (1, 96) and np.isfinite(pred).all() and (pred >= 0).all()
assert model.model.inputs[0].shape[1:] == (288, 4)
assert model.model.inputs[1].shape[1:] == (96, 6)
results["retained_gru_reload"] = {"status": "passed", "history_shape": [288, 4], "future_shape": [96, 6]}

raw = pd.read_csv(source)
# An isolated QA copy forces the retained GRU to exercise its documented fallback.
with tempfile.TemporaryDirectory(dir=out) as temp:
    bundle = Path(temp)
    candidate_meta = dict(meta, selected_model="gru")
    (bundle / "model_metadata.json").write_text(json.dumps(candidate_meta), encoding="utf-8")
    shutil.copy2(run / "scalers.joblib", bundle / "scalers.joblib")
    missing_model = predict_record(bundle, raw)
    assert missing_model["status"] == "degraded"
    check_record(missing_model)
    shutil.copy2(run / "gru.keras", bundle / "gru.keras")
    missing_weather = predict_record(bundle, raw.drop(columns=spec.history_weather_columns))
    assert missing_weather["status"] == "degraded"
    check_record(missing_weather)
    assert missing_weather["input_data_quality_summary"]["missing_weather_rows"] == len(raw)
    assert missing_weather["input_data_quality_summary"]["missing_history_weather_columns"] == list(spec.history_weather_columns)
    for record in (missing_model, missing_weather):
        assert record["fallback_forecast_ref"] == "baseline:" + meta["baseline_fallback"]["strategy"]
        expected = values(record) + np.array(meta["baseline_fallback"]["calibration"]["quantiles_kw"])
        np.testing.assert_allclose([p["upper"] for p in record["prediction_interval"]["values"]], expected)
    insufficient = predict_record(bundle, raw.tail(10))
    assert insufficient["status"] == "unavailable" and not insufficient["forecast_values"]
    stale = predict_record(bundle, raw, forecast_start="2026-01-02")
    assert stale["status"] == "unavailable" and not stale["forecast_values"]
results["fallback"] = {"missing_model": "degraded", "missing_weather": "degraded",
                       "insufficient_history": "unavailable", "stale_history": "unavailable",
                       "intervals_use_baseline_calibration": True}
results["status"] = "passed"
(out / "verification.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
