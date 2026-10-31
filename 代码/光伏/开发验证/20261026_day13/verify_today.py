"""Verify the public predictor defaults to today using a synthetic date shift."""
from pathlib import Path
import json
import os
import sys
import tempfile

out = Path(__file__).resolve().parent
pv = out.parents[1]
scratch = out/"runtime_tmp"
os.environ.update(TEMP=str(scratch), TMP=str(scratch), TF_NUM_INTRAOP_THREADS="1",
    TF_NUM_INTEROP_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", TF_CPP_MIN_LOG_LEVEL="2")
tempfile.tempdir = str(scratch)
sys.path.insert(0, str(pv))
import pandas as pd
from pv_forecast.pipeline import predict_record

frame = pd.read_csv(pv.parent/"数据"/"光伏数据"/"黄花站_2025年_07-19_小时数据.csv").tail(52).copy()
index = pd.to_datetime(frame.timestamp)
today = pd.Timestamp.now(tz="Asia/Shanghai").normalize()
frame["timestamp"] = index + (today-index.iloc[-1].normalize())
bundle = pv/"运行结果"/"20261026_day13_01"
default = predict_record(bundle, frame, asset_scope="synthetic_contract_check_only")
explicit = predict_record(bundle, frame, forecast_start=today.date().isoformat())
assert default["status"] == "available"
assert pd.Timestamp(default["forecast_start_at"]) == today + pd.Timedelta(hours=7)
assert pd.Timestamp(default["forecast_end_at"]) == today + pd.Timedelta(hours=19)
assert default["forecast_values"] == explicit["forecast_values"]
assert [pd.Timestamp(v["timestamp"]).hour for v in default["forecast_values"]] == list(range(7, 20))
result = {"status": "passed", "verified_date": today.date().isoformat(),
    "public_predict_record_default_is_current_day": True, "explicit_date_matches": True,
    "hours": list(range(7, 20)), "interval_minutes": 60,
    "input_kind": "synthetic date-shifted 4-day block; contract check, no actual current-day forecast claimed"}
(out/"current_day_default_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False))
