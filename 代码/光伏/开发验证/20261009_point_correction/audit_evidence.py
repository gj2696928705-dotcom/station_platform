"""Audit artifacts and report accuracy changes without turning them into gates."""
from pathlib import Path
import hashlib
import json

import numpy as np

out = Path(__file__).resolve().parent
pv = out.parents[1]
parent = pv/"运行结果"/"20260929_000009_045346"
candidate = pv/"运行结果"/"20261009_point_correction_01"
frozen = out/"frozen_2026_seed20261010"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


development = read(out/"development_protocol.json")
completion = read(out/"development_completion.json")
report = read(candidate/"point_correction_report.json")
assert development["status"] == "frozen_before_correction_fit" and completion["status"] == report["status"] == "completed"
assert report["parent_full_gru_artifact_count"] == 9
assert not report["test_used_for_selection"] and not report["test_used_for_correction_fit"]
assert not report["gru_weights_retrained"] and report["point_correction_fitted"]
assert report["selected_correction"] == "ridge_0.1" and report["final_correction_training_windows"] == 24145
old = read(parent/"rolling_validation.json")
for fold, historical in zip(report["folds"], old["folds"]):
    assert fold["name"] == historical["name"]
    assert fold["correction_training_windows"] == historical["training_windows_available"]
    for period in ("all_day", "daylight"):
        for metric in ("rmse", "mae", "r2"):
            np.testing.assert_allclose(fold["results"]["raw_gru"][period][metric],
                historical["results"][report["frozen_gru_candidate"]][period][metric], rtol=1e-6, atol=1e-4)
selection = report["aggregate"]
raw, corrected = selection["raw_gru"], selection[report["selected_correction"]]
assert corrected["all_day"]["rmse"] < raw["all_day"]["rmse"]
assert corrected["daylight"]["rmse"] < raw["daylight"]["rmse"]
assert corrected["worst_fold_rmse"] <= raw["worst_fold_rmse"]
assert all(sha(parent/name) == expected for name, expected in report["parent_rolling_artifact_sha256"].items())
assert all(sha(parent/name) == expected for name, expected in report["parent_bundle_sha256"].items())
for name in ("gru.keras", "scalers.joblib"):
    assert sha(candidate/name) == sha(parent/name)
head = read(candidate/"model_metadata.json")["point_correction"]
assert sha(candidate/head["path"]) == head["sha256"]
assert read(out/"candidate_cli_verification.json")["status"] == "passed"
snapshot = out/"source_before_correction"
prior_protocol = read(pv/"开发验证"/"20261009_night_support"/"frozen_2026_01"/"protocol.json")
assert all(sha(snapshot/name) == expected for name, expected in prior_protocol["source_code_sha256"].items())
protocol, done, evaluation = [read(frozen/name) for name in ("protocol.json", "completion.json", "backtest.json")]
assert protocol["status"] == "frozen_before_generation" and done["status"] == "completed"
assert evaluation["evaluated_windows"] == 34657 and evaluation["skipped_origins"] == 0
assert not evaluation["weights_retrained"] and not evaluation["calibration_fitted"] and not evaluation["model_selected"]
assert all(sha(pv/name) == expected for name, expected in protocol["source_sha256"].items())
assert all(sha(pv/"运行结果"/bundle/name) == expected for bundle, values in protocol["bundle_sha256"].items() for name, expected in values.items())
assert sha(frozen/protocol["generator_config"]["output"]["data_filename"]) == evaluation["source_sha256"] == done["source_sha256"]
data = pv.parent/"数据"/"光伏数据"
assert sha(data/"generate_huanghua_pv.py") == protocol["generator_sha256"]
assert sha(data/"黄花站_2025年_15分钟光伏模拟数据.csv") == protocol["original_data_sha256"]
assert read(frozen/"validation_report.json")["repeat_generation_byte_identical"]
models = evaluation["models"]
before = models[parent.name]
after = models[candidate.name]
for model in models.values():
    total = model["overall"]["all_day"]["count"]
    assert total == 34657*96
    assert model["overall"]["daylight"]["count"]+model["overall"]["night"]["count"] == total
    assert len(model["months"]) == 12
    assert sum(m["all_day"]["count"] for m in model["months"].values()) == total
    assert sum(m["count"] for m in model["daylight_weather"].values()) == model["overall"]["daylight"]["count"]
    assert sum(m["count"] for m in model["daylight_lead_groups"].values()) == model["overall"]["daylight"]["count"]
    assert model["after_parent_input_window"]
assert after["overall"]["night"]["rmse"] == 0 and after["overall"]["night"]["coverage"] == 1
changes = {period: {metric: {"before": before["overall"][period][metric], "after": after["overall"][period][metric],
    "improved": after["overall"][period][metric] < before["overall"][period][metric]}
    for metric in ("rmse", "mae")} for period in ("all_day", "daylight")}
months = {month: {"before": before["months"][month]["daylight"]["rmse"],
    "after": after["months"][month]["daylight"]["rmse"],
    "improved": after["months"][month]["daylight"]["rmse"] < before["months"][month]["daylight"]["rmse"]}
    for month in after["months"]}
result = {"status": "passed", "scope": "implementation and one known-simulator new path, not production accuracy",
    "prior_frozen_source_snapshot_verified": True, "parent_all_nine_GRU_evidence_and_hashes_checked": True,
    "selected_GRU_rolling_predictions_match_original": True, "original_weights_scalers_data_generator_unchanged": True,
    "test_excluded_from_head_selection_and_fit": True, "frozen_before_new_evaluation_generation": True,
    "full_evaluation_windows": 34657, "full_evaluation_labels": 3327072,
    "new_path_accuracy_changes": changes, "monthly_daylight_rmse": months,
    "months_with_daylight_improvement": sum(m["improved"] for m in months.values()),
    "weather_daylight_rmse": {state: {"before": before["daylight_weather"][state]["rmse"],
        "after": values["rmse"], "improved": values["rmse"] < before["daylight_weather"][state]["rmse"],
        "coverage_after": values["coverage"], "mean_error_after_kw": values["mean_error_kw"]}
        for state, values in after["daylight_weather"].items()},
    "daylight_interval_coverage": after["overall"]["daylight"]["coverage"],
    "nominal_interval_coverage": after["calibration"]["nominal_coverage"],
    "production_accepted": False}
(out/"evidence_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
