"""Check real correction bundle CLI, power values and artifact failure behavior."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw

out = Path(__file__).resolve().parent
pv = out.parents[1]
sys.path.insert(0, str(pv))
os.environ.update(TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
from pv_forecast.baselines import BaselineForecaster
from pv_forecast.data import InputSpec, load_and_regularize
from pv_forecast.pipeline import _load_bundle, _prediction_spec, predict_record


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


candidate = pv/"运行结果"/"20261009_point_correction_01"
parent = pv/"运行结果"/"20260929_000009_045346"
source = pv.parent/"数据"/"光伏数据"/"黄花站_2025年_15分钟光伏模拟数据.csv"
meta = read(candidate/"model_metadata.json")
commands, pngs = {}, []
for command in ("predict", "evaluate"):
    output = candidate/("forecast.json" if command == "predict" else "evaluation.json")
    result = subprocess.run([sys.executable, "-X", "utf8", "-m", "pv_forecast.cli", command,
        "--input", str(source), "--bundle-dir", str(candidate), "--output", str(output)],
        cwd=pv, capture_output=True, encoding="utf-8")
    (out/f"{command}_cli_stderr.log").write_text(result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    record = read(output)
    assert record == json.loads(result.stdout)
    assert record["status"] == "available" and not record["visualization_errors"]
    for ref in record["visualizations"]:
        path = (candidate/ref["path"]).resolve()
        assert path.is_relative_to(candidate.resolve())
        with Image.open(path) as im:
            im.verify()
        pngs.append(path)
    commands[command] = {"exit_code": result.returncode, "png_count": len(record["visualizations"])}
saved = read(candidate/"forecast.json")
actual = predict_record(candidate, source)
np.testing.assert_allclose([p["value"] for p in saved["forecast_values"]],
                           [p["value"] for p in actual["forecast_values"]], rtol=0, atol=1e-4)
assert actual["confidence"]["point_correction"] == ("solar_weather_ridge" if meta.get("point_correction") else None)
if meta.get("point_correction"):
    assert hashlib.sha256((candidate/"point_correction.joblib").read_bytes()).hexdigest() == meta["point_correction"]["sha256"]
    # Verify the point transformation independently from public prediction.
    from pv_forecast.data import feature_frame
    import pandas as pd
    _, loaded, _ = _load_bundle(candidate, allow_fallback=False)
    spec = _prediction_spec(meta, InputSpec())
    regular = load_and_regularize(source, spec)
    start = pd.Timestamp(saved["forecast_start_at"])
    future = feature_frame(pd.DataFrame(index=pd.date_range(start, periods=96, freq="15min")),
                           latitude=spec.latitude, longitude=spec.longitude)[0]
    raw = predict_record(parent, source)
    expected = loaded.point_correction.predict(regular.frame[meta["history_feature_names"]].iloc[-288:].to_numpy()[None],
        future[meta["feature_names"]].to_numpy()[None], np.array([[p["value"] for p in raw["forecast_values"]]]),
        meta["history_feature_names"], meta["feature_names"])[0]
    np.testing.assert_allclose(expected, [p["value"] for p in actual["forecast_values"]], atol=1e-4, rtol=0)
    for missing in ("point_correction.joblib", "gru.keras"):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for name in ("model_metadata.json", "gru.keras", "scalers.joblib", "point_correction.joblib"):
                if name != missing:
                    shutil.copy2(candidate/name, folder/name)
            fallback = predict_record(folder, source)
            assert fallback["status"] == "degraded" and fallback["confidence"]["point_correction"] is None
            baseline = BaselineForecaster(meta["baseline_fallback"]["strategy"]).predict(
                regular.frame[meta["target_column"]].iloc[-288:].to_numpy()[None], 96)[0]
            np.testing.assert_allclose(baseline, [p["value"] for p in fallback["forecast_values"]])
            assert fallback["prediction_interval"]["method"] == meta["baseline_fallback"]["calibration"]["method"]
assert len(pngs) == 4
canvas = Image.new("RGB", (1600, 840), "white")
draw = ImageDraw.Draw(canvas)
for i, path in enumerate(pngs):
    with Image.open(path) as im:
        im.thumbnail((780, 380), Image.Resampling.LANCZOS)
        x, y = (i%2)*800, (i//2)*420
        canvas.paste(im, (x+10, y+30))
    draw.text((x+10, y+8), path.name, fill="black")
canvas.save(out/"candidate_png_review.png")
result = {"status": "passed", "commands": commands, "reload_points_match_saved": True,
    "point_transformation_independently_checked": bool(meta.get("point_correction")),
    "missing_correction_and_GRU_use_own_baseline": bool(meta.get("point_correction")),
    "png_refs": [p.relative_to(pv).as_posix() for p in pngs], "candidate_only": True}
(out/"candidate_cli_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
