"""Fresh saved-model holdout inference and record-boundary checks, no refit."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile

out = Path(__file__).resolve().parent
pv = out.parents[1]
scratch = out/"runtime_tmp"
scratch.mkdir(exist_ok=True)
os.environ.update(TEMP=str(scratch), TMP=str(scratch), TF_NUM_INTRAOP_THREADS="1",
    TF_NUM_INTEROP_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", TF_CPP_MIN_LOG_LEVEL="2")
tempfile.tempdir = str(scratch)
sys.path.insert(0, str(pv))
import numpy as np
import pandas as pd
from pv_forecast.pipeline import predict_record, evaluate_bundle
from pv_forecast.records import validate_forecast_record
from pv_forecast.day13_data import prepare_hourly

read = lambda p: json.loads(p.read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
bundle = pv/"运行结果"/"20261026_day13_01"
artifacts = {p.name: sha(p) for p in bundle.iterdir() if p.suffix in (".keras", ".joblib", ".json")}
data_dir = pv.parent/"数据"/"光伏数据"
source = data_dir/"黄花站_2025年_07-19_小时数据.csv"
original = data_dir/"黄花站_2025年_15分钟光伏模拟数据.csv"
frame = pd.read_csv(source)
recomputed = prepare_hourly(original)
for name in ("pv_power_kw", "SWGDN", "DHI", "DNI", "rain"):
    np.testing.assert_allclose(frame[name], recomputed[name], atol=1e-10, rtol=0)
assert pd.to_datetime(frame.timestamp, utc=True).equals(pd.to_datetime(recomputed.timestamp, utc=True))
frame["timestamp"] = pd.to_datetime(frame.timestamp)
# Only test target days plus their three-day history are supplied to this call.
holdout = frame.loc[frame.timestamp >= pd.Timestamp("2025-11-04T07:00+08:00")]
evaluation = evaluate_bundle(bundle, holdout)
assert evaluation["evaluated_days"] == 55 and evaluation["metrics"]["count"] == 715
expected = read(bundle/"metrics.json")["test"]
for key in ("rmse", "mae", "r2", "coverage"):
    np.testing.assert_allclose(evaluation["metrics"][key], expected[key], rtol=0, atol=1e-8)
record = predict_record(bundle, frame, out/"forecast.json", forecast_start="2025-12-31")
validate_forecast_record(record)
assert record["status"] == "available" and record["interval_minutes"] == 60
assert [pd.Timestamp(p["timestamp"]).hour for p in record["forecast_values"]] == list(range(7, 20))
saved = read(bundle/"forecast.json")
assert record["forecast_values"] == saved["forecast_values"]
assert record["prediction_interval"] == saved["prediction_interval"]
rejects = {}
wrong = deepcopy(record)
wrong["forecast_end_at"] = "2025-12-31T23:00+08:00"
missing = deepcopy(record)
missing["prediction_interval"]["values"].pop()
wrong_date = deepcopy(record)
wrong_date["prediction_interval"]["values"][0]["timestamp"] = "2025-12-30T07:00+08:00"
for name, broken in (("wrong_end_time", wrong), ("missing_interval_step", missing), ("interval_wrong_date", wrong_date)):
    try:
        validate_forecast_record(broken)
    except ValueError as exc:
        rejects[name] = str(exc)
    else:
        raise AssertionError(name+" was accepted")
assert {p.name: sha(p) for p in bundle.iterdir() if p.name in artifacts} == artifacts
result = {"status": "passed", "fresh_holdout_samples": 55, "fresh_holdout_points": 715,
    "fresh_metrics": evaluation["metrics"], "saved_metrics_reproduced": True,
    "known_target_weather_is_conditional_input": True, "retrained": False,
    "full_year_hourly_aggregation_recomputed": True, "original_sha256": sha(original),
    "forecast_points": 13, "interval_minutes": 60, "forecast_hours": list(range(7, 20)),
    "saved_forecast_reproduced": True, "record_boundary_rejections": rejects,
    "bundle_hashes_unchanged": artifacts,
    "source_sha256": {str(p.relative_to(pv)): sha(p) for p in sorted((pv/"pv_forecast").rglob("*.py"))}}
(out/"inference_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"status": "passed", "metrics": evaluation["metrics"], "rejections": rejects}, ensure_ascii=False, indent=2))
