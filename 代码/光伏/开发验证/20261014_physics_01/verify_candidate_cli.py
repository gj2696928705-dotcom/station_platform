"""Actual full-training candidates: CLI, reload, units, causality and fallback."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

out = Path(__file__).resolve().parent
pv = out.parents[1]
sys.path.insert(0, str(pv))
scratch = (out/"runtime_tmp").resolve()
assert scratch.is_relative_to(pv.resolve())
scratch.mkdir(exist_ok=True)
tempfile.tempdir = str(scratch)
os.environ.update(TEMP=str(scratch), TMP=str(scratch), TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1",
    OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from pv_forecast.baselines import BaselineForecaster
from pv_forecast.data import InputSpec, load_and_regularize
from pv_forecast.metrics import apply_prediction_interval
from pv_forecast.pipeline import _prediction_spec, _daylight_mask, predict_record


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


comparison_dir = Path(read(out/"active_comparison.json")["path"]) if (out/"active_comparison.json").exists() else pv/"运行结果"/"20261014_physics_comparison_01"
report = read(comparison_dir/"comparison_report.json")
assert report["status"] == "completed"
source = pv.parent/"数据"/"光伏数据"/"黄花站_2025年_15分钟光伏模拟数据.csv"
raw = pd.read_csv(source)
origin = pd.Timestamp("2025-06-01T00:00:00+08:00")
after = pd.to_datetime(raw.timestamp) >= origin
changed = raw.copy()
for column in ("pv_power_kw", "SWGDN", "temperature_c", "cloud_cover"):
    changed[column] = changed[column].astype(object)
    changed.loc[after, column] = "unavailable-future-observation"
changed.loc[after, "weather_type"] = "unavailable-scoring-label"
evidence = out/"candidate_cli"
assert not evidence.exists(), "preserve completed/live CLI evidence"
evidence.mkdir()
results, pngs = {}, []
for name, folder_path in report["bundle_dirs"].items():
    bundle = Path(folder_path)
    meta = read(bundle/"model_metadata.json")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir()
        if p.name in ("model_metadata.json", "gru.keras", "scalers.joblib", "physics_tree.joblib", "interval_calibration.json")}
    directory = evidence/name
    directory.mkdir()
    commands = {}
    for command in ("predict", "evaluate"):
        output = directory/("forecast.json" if command == "predict" else "evaluation.json")
        cli = subprocess.run([sys.executable, "-X", "utf8", "-m", "pv_forecast.cli", command,
            "--input", str(source), "--bundle-dir", str(bundle), "--output", str(output)],
            cwd=pv, capture_output=True, encoding="utf-8", env=os.environ.copy())
        (directory/f"{command}_stderr.log").write_text(cli.stderr, encoding="utf-8")
        assert cli.returncode == 0, cli.stderr
        record = read(output)
        assert record == json.loads(cli.stdout)
        assert record["status"] == "available" and not record["visualization_errors"]
        for ref in record["visualizations"]:
            path = (directory/ref["path"]).resolve()
            assert path.is_relative_to(directory.resolve())
            with Image.open(path) as image:
                image.verify()
            pngs.append(path)
        commands[command] = {"exit_code": cli.returncode, "png_count": len(record["visualizations"])}
    saved = read(directory/"forecast.json")
    actual = predict_record(bundle, source)
    point = np.array([p["value"] for p in actual["forecast_values"]])
    np.testing.assert_allclose(point, [p["value"] for p in saved["forecast_values"]], atol=1e-4, rtol=0)
    assert actual["confidence"]["physics_features"] and all(p["unit"] == "kW" for p in actual["forecast_values"])
    assert actual["interval_minutes"] == 15 and len(point) == 96
    assert np.isfinite(point).all() and point.min() >= 0 and point.max() <= 5000
    spec = _prediction_spec(meta, InputSpec())
    day = _daylight_mask([pd.Timestamp(actual["forecast_start_at"])], 96, 15, spec.latitude, spec.longitude)[0]
    np.testing.assert_array_equal(point[~day], 0)
    causal = predict_record(bundle, raw, forecast_start=origin)
    altered = predict_record(bundle, changed, forecast_start=origin)
    assert causal["status"] == altered["status"] == "available"
    assert causal["forecast_values"] == altered["forecast_values"]
    assert causal["prediction_interval"] == altered["prediction_interval"]
    assert causal["input_data_quality_summary"] == altered["input_data_quality_summary"]
    regular = load_and_regularize(source, spec)
    missing = "gru.keras" if meta["selected_model"] == "gru" else "physics_tree.joblib"
    fallback_source = source
    with tempfile.TemporaryDirectory() as temp:
        copy_dir = Path(temp)
        for filename in hashes:
            if filename != missing:
                shutil.copy2(bundle/filename, copy_dir/filename)
        if meta["selected_model"] == "physics_persistence":
            fallback_source = raw[["timestamp", "pv_power_kw"]]
        degraded = predict_record(copy_dir, fallback_source)
        assert degraded["status"] == "degraded" and not degraded["confidence"]["physics_features"]
        strategy = meta["baseline_fallback"]["strategy"]
        expected = BaselineForecaster(strategy).predict(regular.frame.pv_power_kw.iloc[-288:].to_numpy()[None], 96)
        np.testing.assert_allclose(expected[0], [p["value"] for p in degraded["forecast_values"]], atol=1e-6, rtol=0)
        lower, upper = apply_prediction_interval(expected, meta["baseline_fallback"]["calibration"], day[None])
        np.testing.assert_allclose(lower[0], [p["lower"] for p in degraded["prediction_interval"]["values"]], atol=1e-6, rtol=0)
        np.testing.assert_allclose(upper[0], [p["upper"] for p in degraded["prediction_interval"]["values"]], atol=1e-6, rtol=0)
    assert all(hashlib.sha256((bundle/filename).read_bytes()).hexdigest() == digest for filename, digest in hashes.items())
    results[name] = {"status": "passed", "commands": commands, "reload_points_match": True,
        "physical_units_and_bounds": "96 kW points, 15 minutes, [0,5000], night zero",
        "future_numeric_observations_and_weather_labels_isolated": True,
        "degraded_uses_own_baseline_and_intervals": True, "bundle_hashes_preserved": hashes}
assert len(pngs) == 12
canvas = Image.new("RGB", (1600, 2520), "white")
draw = ImageDraw.Draw(canvas)
for i, path in enumerate(pngs):
    with Image.open(path) as image:
        image.thumbnail((780, 380), Image.Resampling.LANCZOS)
        x, y = i%2*800, i//2*420
        canvas.paste(image, (x+10, y+30))
    draw.text((x+10, y+8), path.parent.parent.name+"/"+path.name, fill="black")
canvas.save(out/"candidate_png_review.png")
write(out/"candidate_cli_verification.json", {"status": "passed", "candidates": results,
    "png_count": 12, "png_refs": [str(p.relative_to(pv)) for p in pngs], "publication_status": "candidate"})
print(json.dumps(results, ensure_ascii=False, indent=2))
