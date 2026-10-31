"""Audit complete fits, temporal boundaries, old artifacts and final cohorts."""
from pathlib import Path
import hashlib
import json
import os
import re

os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

out = Path(__file__).resolve().parent
pv = out.parents[1]
comparison_dir = (Path(json.loads((out/"active_comparison.json").read_text(encoding="utf-8"))["path"])
    if (out/"active_comparison.json").exists() else pv/"运行结果"/"20261014_physics_comparison_01")
control = pv/"运行结果"/"20261009_point_correction_01"
data = pv.parent/"数据"/"光伏数据"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


protocol = read(out/"development_protocol.json")
report = read(comparison_dir/"comparison_report.json")
assert report["status"] == "completed" and len(report["folds"]) == 3
if (out/"recovery_completion.json").exists():
    recovered = read(out/"recovery_completion.json")
    assert recovered["status"] == "completed" and recovered["original_artifacts_unchanged"]
    assert len(recovered["reused_gru_fits"]) == 8 and len(recovered["reused_tree_fits"]) == 3
    assert read(out/"recovery_cache_verification.json")["status"] == "passed"
    assert all(sha(Path(recovered["source_run"])/name) == value for name, value in recovered["original_artifact_sha256"].items())
assert report["test_used_for_selection"] is False and report["oracle_used_for_selection"] is False
assert sha(data/"黄花站_2025年_15分钟光伏模拟数据.csv") == protocol["source_sha256"]
assert sha(data/"simulation_config.json") == protocol["physics_config_sha256"]
assert sha(data/"generate_huanghua_pv.py") == protocol["generator_sha256"]
assert all(sha(control/name) == value for name, value in protocol["control_sha256"].items())
assert all(sha(path) == value for path, value in report["control_hashes"].items())
assert all(sha(pv/name) == value for name, value in protocol["implementation_sha256"].items())
previous = read(pv/"开发验证"/"20261009_point_correction"/"frozen_2026_seed20261010"/"protocol.json")
assert all(sha(out/"source_before_physics"/name) == value for name, value in previous["source_sha256"].items())
assert read(out/"physical_boundary_verification.json")["status"] == "passed"
assert read(out/"aggregation_verification.json")["status"] == "passed"
old_report = read(control/"point_correction_report.json")
fits, pngs = [], []


def check_gru(directory, windows, final=False):
    summary = read(directory/"training_summary.json")
    history = read(directory/"training_history.json")
    config = read(directory/"training_config.json")
    scalers = joblib.load(directory/"scalers.joblib")
    assert summary["status"] == "completed" and summary["training_windows_used"] == windows
    assert summary["shuffle"] is False and summary["history_feature_count"] == 13
    assert summary["epochs_completed"] == len(history)
    assert len(history) <= config["gru_epochs"]
    assert summary["best_epoch"] == min(history, key=lambda row: row["val_loss"])["epoch"]
    assert summary["best_monitor_value"] == min(row["val_loss"] for row in history)
    assert not summary["visualization_errors"] and (directory/"gru.keras").is_file()
    for key, features, points in (("target_scaler", 1, 96), ("feature_scaler", 8, 96), ("history_scaler", 13, 288)):
        assert scalers[key].n_features_in_ == features
        assert scalers[key].n_samples_seen_ == windows*points
    image_path = directory/"visualizations"/"training_loss.png"
    with Image.open(image_path) as image:
        image.verify()
    pngs.append(image_path)
    fits.append({"path": directory.relative_to(pv).as_posix(), "training_windows": windows,
        "epochs_completed": len(history), "best_epoch": summary["best_epoch"],
        "stopped_early": summary["stopped_early"], "full_windows": True,
        "train_only_scalers_count_verified": True, "history_features": 13, "future_features": 8})
    if not final:
        split = read(directory/"data_split.json")
        assert split["training_windows_used"] == split["training_windows_available"] == windows


for fold in report["folds"]:
    old_fold = next(f for f in old_report["folds"] if f["name"] == fold["name"])
    for metric in ("rmse", "mae", "r2"):
        np.testing.assert_allclose(fold["results"]["current_gru_ridge"]["diagnostics"]["daylight"][metric],
            old_fold["results"]["ridge_0.1"]["daylight"][metric], rtol=1e-6, atol=1e-4)
    for candidate in report["gru_candidates"]:
        check_gru(comparison_dir/"experiments"/fold["name"]/("physics_"+candidate), fold["training_windows"])
    tree = read(comparison_dir/"experiments"/fold["name"]/"physics_tree"/"training_summary.json")
    assert tree["training_windows_used"] == fold["training_windows"] and tree["all_available_windows_used"]
    assert tree["iterations"] == 200 and tree["internal_random_validation"] is False
check_gru(Path(report["bundle_dirs"]["enhanced_gru"]), 24145, final=True)
assert len(fits) == len(pngs) == 10
splits = report["data_splits"]
assert report["final_training_windows"] == 24145 and report["retrospective_test_windows"] == 5161
assert report["final_selection_windows"] == report["final_calibration_windows"] == 2533
assert pd.Timestamp(splits["train"]["end"]) < pd.Timestamp(splits["selection_validation"]["start"])
assert pd.Timestamp(splits["selection_validation"]["end"]) < pd.Timestamp(splits["calibration"]["start"])
assert pd.Timestamp(splits["calibration"]["end"]) < pd.Timestamp(splits["test"]["start"])
for name, directory in report["bundle_dirs"].items():
    meta = read(Path(directory)/"model_metadata.json")
    assert meta["calibration_independent_of_final_early_stopping"]
    assert meta["calibration"]["calibration_samples"] == 2533
    assert meta["baseline_fallback"]["calibration"]["calibration_samples"] == 2533
    assert not meta.get("point_correction") and not meta.get("future_weather_columns")
    assert meta["publication_status"] == "candidate"
    assert len(meta["history_feature_names"]) == 13 and len(meta["feature_names"]) == 8
    if name == "physics_tree":
        assert sha(Path(directory)/"physics_tree.joblib") == meta["physics_tree_sha256"]
assert read(out/"candidate_cli_verification.json")["status"] == "passed"
test_log = (out/"unit_tests.log").read_text(encoding="utf-8")
matched = re.search(r"Ran (\d+) tests in ([\d.]+)s\s+OK", test_log)
assert matched and int(matched.group(1)) == 61
cohort = out/"frozen_three_paths"
frozen_protocol, frozen_done, aggregate = [read(cohort/name) for name in ("protocol.json", "completion.json", "aggregate.json")]
assert frozen_protocol["status"] == "frozen_before_generation" and frozen_done["status"] == "completed"
assert pd.Timestamp(read(out/"adoption_guard_amendment.json")["registered_at_utc"]) < pd.Timestamp(frozen_protocol["frozen_at_utc"])
assert "short_leads" in frozen_protocol["adoption_rule"]
assert all(sha(pv/name) == value for name, value in frozen_protocol["source_code_sha256"].items())
assert all(sha(out/name) == value for name, value in frozen_protocol["experiment_script_sha256"].items())
correction = read(out/"aggregation_float64_verification.json")
assert correction["status"] == "passed" and correction["per_group_R2_checks"] == 276
assert correction["max_absolute_R2_difference"] < 3e-8
assert sha(out/"aggregate_float64_correction.py") == correction["correction_script_sha256"]
assert sha(out/"aggregate_frozen.py") == correction["frozen_script_preserved_sha256"]
assert all(sha(cohort/seed/"backtest.json") == value
    for seed, value in correction["backtest_reports_preserved_sha256"].items())
bundle_paths = {"20261009_point_correction_01": control,
    **{name: Path(path) for name, path in report["bundle_dirs"].items()}}
assert all(sha(bundle_paths[name]/filename) == value for name, files in frozen_protocol["bundle_sha256"].items()
    for filename, value in files.items())
for seed in frozen_protocol["seeds"]:
    folder = cohort/str(seed)
    seed_report = read(folder/"backtest.json")
    seed_config = read(folder/"simulation_config.json")
    assert seed_report["evaluated_windows"] == 34657 and seed_report["skipped_origins"] == 0
    assert read(folder/"validation_report.json")["repeat_generation_byte_identical"]
    assert sha(folder/seed_config["output"]["data_filename"]) == seed_report["source_sha256"]
    assert not seed_report["weights_retrained"] and not seed_report["calibration_fitted"] and not seed_report["model_selected"]
    oracle = seed_report["ideal_future_weather_diagnostic"]
    assert oracle["diagnostic_only"] and not oracle["production_eligible"] and not oracle["model_selection_eligible"]
    for model in seed_report["models"].values():
        total = model["overall"]["all_day"]["count"]
        day = model["overall"]["daylight"]["count"]
        assert total == 34657*96 and total == day+model["overall"]["night"]["count"]
        assert sum(month["all_day"]["count"] for month in model["months"].values()) == total
        assert sum(weather["count"] for weather in model["daylight_weather"].values()) == day
        assert sum(lead["count"] for lead in model["error_diagnostics"]["daylight_by_lead"].values()) == day
        assert model["after_parent_input_window"] and model["overall"]["night"]["rmse"] == 0
canvas = Image.new("RGB", (1600, 2100), "white")
draw = ImageDraw.Draw(canvas)
for i, path in enumerate(pngs):
    with Image.open(path) as image:
        image.thumbnail((780, 380), Image.Resampling.LANCZOS)
        x, y = i%2*800, i//2*420
        canvas.paste(image, (x+10, y+30))
    draw.text((x+10, y+8), path.parent.parent.parent.name+"/"+path.parent.parent.name, fill="black")
canvas.save(out/"training_loss_review.png")
result = {"status": "passed", "full_GRU_fits": fits, "nine_rolling_plus_one_final": True,
    "all_four_tree_fits_full_windows": True, "control_predictions_match_original_rolling_results": True,
    "test_and_future_weather_excluded_from_selection": True, "calibration_excluded_from_final_early_stopping": True,
    "three_frozen_full_year_paths": frozen_protocol["seeds"], "evaluated_windows": 103971,
    "scored_labels_per_model": 9981216, "four_comparable_frozen_bundles": True,
    "code_and_bundle_frozen_hashes_preserved": True, "original_data_generator_config_and_control_preserved": True,
    "previous_frozen_source_snapshot_verified": True, "hand_computed_physical_units_and_ratio_floors_verified": True,
    "independent_float64_truth_moments_verified": correction,
    "full_regression": {"tests": int(matched.group(1)), "seconds": float(matched.group(2)), "status": "passed"},
    "CLI_reload_causality_units_fallback_PNG_verified": True,
    "stable_point_gain": aggregate["stable_point_accuracy_gain"], "recommendation": aggregate["recommended_bundle"],
    "production_accepted": False, "oracle_selection_eligible": False}
write(out/"evidence_audit.json", result)
print(json.dumps(result, ensure_ascii=False, indent=2))
