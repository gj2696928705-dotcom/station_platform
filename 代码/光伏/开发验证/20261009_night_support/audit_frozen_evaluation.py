"""Audit protocol, all-window counts and interval support comparisons."""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
from PIL import Image, ImageDraw

out = Path(__file__).resolve().parent
pv = out.parents[1]
frozen = out / "frozen_2026_01"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


report, protocol, completion = [read(frozen / name) for name in ("backtest.json", "protocol.json", "completion.json")]
assert protocol["status"] == "frozen_before_generation" and completion["status"] == "completed"
assert completion["evaluated_windows"] == report["evaluated_windows"] == 34657
assert report["eligible_origins"] == 34657 and report["skipped_origins"] == 0
assert report["data_kind"] == "simulated"
assert not report["weights_retrained"] and not report["calibration_fitted"] and not report["model_selected"]
assert all(digest(pv / path) == expected for path, expected in protocol["source_code_sha256"].items())
csv = frozen / protocol["generator_config"]["output"]["data_filename"]
assert digest(csv) == completion["source_sha256"] == report["source_sha256"]
for bundle, values in protocol["bundle_sha256"].items():
    assert all(digest(pv / "运行结果" / bundle / name) == expected for name, expected in values.items())
models = report["models"]
original, grouped, supported = [models[name] for name in
    ("20260929_000009_045346", "20261009_daylight_recalibration_01", "20261009_night_zero_lower_01")]
for model in models.values():
    assert model["overall"]["all_day"]["count"] == 34657*96
    assert sum(m["all_day"]["count"] for m in model["months"].values()) == 34657*96
    assert len(model["months"]) == 12
    assert model["overall"]["daylight"]["count"] + model["overall"]["night"]["count"] == 34657*96
    assert sum(m["count"] for m in model["daylight_weather"].values()) == model["overall"]["daylight"]["count"]
    assert sum(m["count"] for m in model["daylight_lead_groups"].values()) == model["overall"]["daylight"]["count"]
    for period in ("all_day", "daylight", "night"):
        for metric in ("rmse", "mae", "r2", "count"):
            a, b = model["overall"][period][metric], original["overall"][period][metric]
            if a is None:
                assert b is None
            else:
                np.testing.assert_allclose(a, b, rtol=0, atol=0)
    assert model["after_parent_input_window"]
for month in supported["months"]:
    assert supported["months"][month]["night"]["coverage"] == 1
    assert supported["months"][month]["daylight"] == grouped["months"][month]["daylight"]
assert supported["overall"]["night"]["coverage"] == 1
assert supported["overall"]["night"]["mean_interval_width_kw"] > grouped["overall"]["night"]["mean_interval_width_kw"]
assert supported["overall"]["night"]["mean_interval_width_kw"] < original["overall"]["night"]["mean_interval_width_kw"]
candidate = pv / "运行结果" / "20261009_night_zero_lower_01"
cli = read(candidate / "verification" / "verification.json")
assert cli["status"] == "passed"
assert read(candidate / "interval_calibration.json")["night_lower_bound"] == "zero"
# Validate all actual CLI PNG references and build a contact sheet for review.
paths = []
for name in ("forecast.json", "evaluation.json"):
    record = read(candidate / name)
    assert not record["visualization_errors"]
    for reference in record["visualizations"]:
        path = (candidate / reference["path"]).resolve()
        assert path.is_relative_to(candidate.resolve()) and path.stat().st_size > 0
        with Image.open(path) as image:
            image.verify()
        paths.append(path)
assert len(paths) == 4
canvas = Image.new("RGB", (1600, 840), "white")
draw = ImageDraw.Draw(canvas)
for i, path in enumerate(paths):
    with Image.open(path) as image:
        image.thumbnail((780, 380), Image.Resampling.LANCZOS)
        x, y = (i % 2)*800, (i // 2)*420
        canvas.paste(image, (x+10, y+30))
    draw.text((x+10, y+8), path.name, fill="black")
canvas.save(out / "night_support_png_review.png")
result = {"status": "passed", "protocol_frozen_before_generation": True,
    "no_fitting_or_model_selection": True, "evaluated_windows": 34657, "scored_labels": 34657*96,
    "twelve_months_checked": True, "shared_point_predictions_unchanged": True,
    "daylight_metrics_unchanged_from_grouped_candidate": True, "night_zero_support_all_months_checked": True,
    "bundle_and_source_hashes_unchanged": True, "cli_ref": str((candidate/"verification"/"verification.json").relative_to(pv)),
    "png_refs_checked": [str(path.relative_to(pv)) for path in paths],
    "production_accepted": False, "scope": "new random path of a known simulator; not real-site accuracy"}
write(out / "frozen_evaluation_audit.json", result)
print(json.dumps(result, indent=2))
