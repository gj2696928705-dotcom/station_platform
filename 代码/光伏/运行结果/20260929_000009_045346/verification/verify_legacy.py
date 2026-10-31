"""Reproducible CLI compatibility audit; old bundle stays read-only."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

pv = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(pv))
from pv_forecast.records import validate_forecast_record

run = Path(__file__).resolve().parents[1]
old = run.parent / "20260928_224634_531144"
output = Path(__file__).parent / "legacy_cli"
source = pv.parent / "\u6570\u636e" / "\u5149\u4f0f\u6570\u636e" / "\u9ec4\u82b1\u7ad9_2025\u5e74_15\u5206\u949f\u5149\u4f0f\u6a21\u62df\u6570\u636e.csv"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

before = {p.name: digest(p) for p in old.iterdir() if p.is_file()}
env = dict(os.environ, TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1")
results = {}
for action in ("predict", "evaluate"):
    result_file = output / ("forecast.json" if action == "predict" else "evaluation_new.json")
    result = subprocess.run([sys.executable, "-X", "utf8", "-m", "pv_forecast.cli", action,
        "--input", str(source), "--bundle-dir", str(old), "--output", str(result_file)],
        cwd=pv, env=env, capture_output=True, text=True, encoding="utf-8", check=True)
    (output / (action + "_stderr.log")).write_text(result.stderr, encoding="utf-8")
    record = json.loads(result_file.read_text(encoding="utf-8"))
    assert record["status"] == "available"
    images = []
    for ref in record["visualizations"]:
        path = (output / ref["path"]).resolve()
        assert path.is_relative_to(output.resolve()) and path.stat().st_size > 0
        with Image.open(path) as png:
            assert png.format == "PNG"
            png.verify()
        images.append(ref["id"])
    assert not record["visualization_errors"]
    results[action] = {"exit_code": result.returncode, "status": record["status"], "images": images}
    if action == "predict":
        validate_forecast_record(record)
        baseline = json.loads((old / "forecast.json").read_text(encoding="utf-8"))
        values = np.array([p["value"] for p in record["forecast_values"]])
        original = np.array([p["value"] for p in baseline["forecast_values"]])
        assert values.shape == (96,)
        assert np.isfinite(values).all() and (values >= 0).all()
        np.testing.assert_allclose(values, original, rtol=1e-5, atol=.01)
        results[action].update(points=96, max_absolute_difference_kw=float(np.max(abs(values-original))))

assert before == {p.name: digest(p) for p in old.iterdir() if p.is_file()}
results["original_bundle_unchanged"] = True
results["status"] = "passed"
results["limits"] = "One legacy bundle checked across these runtimes; not a general guarantee for arbitrary Keras/sklearn versions."
(output / "verification.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
