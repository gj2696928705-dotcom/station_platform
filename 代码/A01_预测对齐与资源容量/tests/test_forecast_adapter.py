from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import sys

import pytest

from capacity_alignment.forecast_adapter import normalize, run_candidate, sha, verify_deployment
from capacity_alignment.forecast_capacity import BASE, freeze
from capacity_alignment.common import CapacityError as PlatformError, digest, parse_time
from .test_forecast_capacity import sample


def raw_record(kind="load"):
    first = parse_time("2025-12-31T07:00:00+08:00")
    count, minutes = (96, 15) if kind == "load" else (13, 60)
    points = [{"timestamp": (first+timedelta(minutes=minutes*i)).isoformat(), "unit": "kW", "value": 100} for i in range(count)]
    return {"forecast_id": "candidate-fixture-1", "asset_scope": "HH-Huanghua-CSS",
            "forecast_type": "station_load" if kind == "load" else "pv_power", "model_version": "test-1",
            "status": "available", "target_semantics": BASE, "as_of": first.isoformat(), "interval_minutes": minutes,
            "forecast_start_at": first.isoformat(), "forecast_end_at": (first+timedelta(minutes=minutes*(count if kind == "load" else count-1))).isoformat(),
            "forecast_values": points,
            "prediction_interval": {"values": [{"timestamp": p["timestamp"], "lower": 80, "upper": 120, "unit": "kW"} for p in points]},
            "confidence": {"sampling_layout": "day13_07_19", "weather_forecast_issued_at": "2025-12-31T06:00:00+08:00"}}


def fixture_deployment(tmp_path, behavior="normal"):
    root = tmp_path/"model"
    (root/"source/load_forecast").mkdir(parents=True)
    (root/"bundle").mkdir()
    (root/"source/load_forecast/__init__.py").write_text("", encoding="utf-8")
    body = ""
    if behavior == "sleep": body = "    import time; time.sleep(3)\n"
    elif behavior == "fail": body = "    raise RuntimeError('test failure')\n"
    elif behavior == "noisy": body = "    print('x'*100000)\n"
    write = "Path(output_path).write_text(json.dumps(record), encoding='utf-8')" if behavior != "corrupt" else "Path(output_path).write_text('{', encoding='utf-8')"
    code = "import json\nfrom pathlib import Path\ndef predict_record(bundle, history, output_path, as_of):\n" + body + "    record=json.loads((Path(bundle)/'fixture.json').read_text(encoding='utf-8'))\n    " + write + "\n    return record\n"
    (root/"source/load_forecast/pipeline.py").write_text(code, encoding="utf-8")
    (root/"bundle/fixture.json").write_text(json.dumps(raw_record()), encoding="utf-8")
    manifest = {"schema_version": "candidate-deployment-1", "simulation": True, "kind": "load",
                "station_id": "HH-Huanghua-CSS", "meter_boundary": "AC-BUS", "model_version": "test-1",
                "python": sys.executable, "files": {p.relative_to(root).as_posix(): sha(p) for p in root.rglob("*") if p.is_file()}}
    manifest["manifest_hash"] = digest(manifest)
    path = root/"deployment.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    history = tmp_path/"history.csv"
    history.write_text("station_id,timestamp,value\nHH-Huanghua-CSS,2025-12-30,100\n", encoding="utf-8")
    return path, history


@pytest.mark.parametrize("behavior,code", [("normal", None), ("corrupt", "FORECAST_OUTPUT_INVALID"),
    ("fail", "FORECAST_PROCESS_FAILED"), ("sleep", "FORECAST_TIMEOUT"), ("noisy", "FORECAST_OUTPUT_INVALID")])
def test_independent_process_contract_and_bounded_failures(tmp_path, behavior, code):
    path, history = fixture_deployment(tmp_path, behavior)
    if code:
        with pytest.raises(PlatformError) as error:
            run_candidate(path, history, "2025-12-31T07:00:00+08:00", tmp_path/"runs", timeout=.3 if behavior == "sleep" else 10, max_bytes=10000)
        assert error.value.code == code
    else:
        first = run_candidate(path, history, "2025-12-31T07:00:00+08:00", tmp_path/"runs")
        second = run_candidate(path, history, "2025-12-31T07:00:00+08:00", tmp_path/"runs")
        assert first["status"] == "available"
        assert first["process"]["run_dir"] != second["process"]["run_dir"]
        assert first["raw_record"] == raw_record()


def test_deployment_hash_and_history_station_are_enforced(tmp_path):
    path, history = fixture_deployment(tmp_path)
    history.write_text("station_id\nOTHER-STATION\n", encoding="utf-8")
    with pytest.raises(PlatformError) as error:
        run_candidate(path, history, "2025-12-31T07:00:00+08:00", tmp_path/"runs")
    assert error.value.code == "STATION_MISMATCH"
    (path.parent/"bundle/fixture.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PlatformError) as error: verify_deployment(path)
    assert error.value.code == "MODEL_HASH_MISMATCH"


def test_day13_bounds_and_weather_visibility():
    deployment = {"kind": "pv", "station_id": "HH-Huanghua-CSS", "meter_boundary": "AC-BUS", "model_version": "test-1",
                  "manifest_hash": digest("test"), "files": {}, "value_semantics": "hourly_mean_start_label"}
    raw = raw_record("pv")
    result = normalize(raw, deployment, "2025-12-31T06:30:00+08:00", digest("history"), "test")
    assert result["status"] == "available"
    assert result["expires_at"] == "2025-12-31T20:00:00+08:00"
    assert raw["forecast_end_at"] == "2025-12-31T19:00:00+08:00"
    assert result["raw_record"] == raw
    raw["confidence"]["weather_forecast_issued_at"] = "2025-12-31T06:45:00+08:00"
    assert normalize(raw, deployment, "2025-12-31T06:30:00+08:00", "hash", "test")["status"] == "unavailable"
    raw["confidence"]["weather_forecast_issued_at"] = None
    assert normalize(raw, deployment, "2025-12-31T09:50:00+08:00", "hash", "test")["status"] == "unavailable"


@pytest.mark.parametrize("window", [("2025-12-31T06:50:00+08:00", "2025-12-31T07:50:00+08:00"),
                                     ("2025-12-31T19:07:30+08:00", "2025-12-31T20:07:30+08:00")])
def test_pv_valid_window_never_extrapolates_zero(window):
    inputs = sample(window[0])
    inputs["pv_forecast"]["values"] = [{"start": "2025-12-31T07:00:00+08:00", "end": "2025-12-31T20:00:00+08:00", "value": 100}]
    result = freeze(inputs)
    assert result["status"] == "unavailable"
    assert "COVERAGE_INCOMPLETE" in {i["code"] for i in result["issues"]}
    assert all(s["capacity_kw"] is None for s in result["slots"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1, "1e10000"])
def test_nonfinite_and_negative_values_produce_explainable_failures(value):
    inputs = sample()
    inputs["load_forecast"]["values"][0]["value"] = value
    result = freeze(inputs)
    assert result["status"] == "unavailable"
    assert result["issues"][0]["code"] == "NUMBER_INVALID"
    assert all(s["capacity_kw"] is None for s in result["slots"])


def test_latest_weather_batch_is_not_mixed_or_replaced(tmp_path):
    path, history = fixture_deployment(tmp_path)
    weather = tmp_path/"weather.csv"
    weather.write_text("issued_at,timestamp,SWGDN,DHI,DNI,rain\n2025-12-31T06:15:00+08:00,2025-12-31T07:00:00+08:00,1,1,1,0\n", encoding="utf-8")
    with pytest.raises(PlatformError) as error:
        run_candidate(path, history, "2025-12-31T07:00:00+08:00", tmp_path/"runs", weather=weather)
    assert error.value.code == "WEATHER_BATCH_INCOMPLETE"
