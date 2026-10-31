"""Audit completed output files independently of the training process."""
from pathlib import Path
import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--partial", action="store_true")
args = parser.parse_args()
run = Path(__file__).resolve().parents[1]
pv = run.parent.parent
sys.path.insert(0, str(pv))
from pv_forecast.data import feature_frame

source = pv.parent / "\u6570\u636e" / "\u5149\u4f0f\u6570\u636e" / "\u9ec4\u82b1\u7ad9_2025\u5e74_15\u5206\u949f\u5149\u4f0f\u6a21\u62df\u6570\u636e.csv"
raw = pd.read_csv(source)
raw.index = pd.DatetimeIndex(pd.to_datetime(raw.pop("timestamp")))

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def png_refs(record, base):
    assert not record.get("visualization_errors", [])
    refs = record.get("visualizations", [])
    for ref in refs:
        path = (base / ref["path"]).resolve()
        assert path.is_relative_to(base.resolve()), path
        assert path.stat().st_size > 0
        with Image.open(path) as png:
            assert png.format == "PNG"
            png.verify()
    return len(refs)

def scaler_audit(folder, train_end, input_spec, windows):
    # Check saved extrema against just the eligible observed training range.
    # A 288-point history precedes each complete 96-point target window.
    train = raw.loc[:pd.Timestamp(train_end)]
    future, _ = feature_frame(train, latitude=input_spec["latitude"], longitude=input_spec["longitude"])
    arrays = {
        "target_scaler": train[["pv_power_kw"]].iloc[288:].to_numpy(),
        "feature_scaler": future.iloc[288:].to_numpy(),
        "history_scaler": train[["pv_power_kw", "SWGDN", "temperature_c", "cloud_cover"]].iloc[:-96].to_numpy(),
    }
    scalers = joblib.load(folder / "scalers.joblib")
    for name, array in arrays.items():
        np.testing.assert_allclose(scalers[name].data_min_, array.min(axis=0), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(scalers[name].data_max_, array.max(axis=0), rtol=1e-12, atol=1e-12)
        assert scalers[name].n_samples_seen_ == windows * (288 if name == "history_scaler" else 96)

def training_audit(folder, expected_windows):
    summary = read(folder / "training_summary.json")
    history = read(folder / "training_history.json")
    config = read(folder / "training_config.json")
    csv = pd.read_csv(folder / "training_history.csv")
    assert summary["status"] == "completed"
    assert summary["training_windows_used"] == expected_windows
    assert summary["shuffle"] is False and config["max_gru_samples"] in (0, None)
    assert len(history) == len(csv) == summary["epochs_completed"]
    assert [h["epoch"] for h in history] == list(range(1, len(history) + 1))
    values = np.array([h["val_loss"] for h in history])
    np.testing.assert_allclose(values, csv.val_loss, rtol=1e-10, atol=1e-12)
    assert summary["best_epoch"] == int(values.argmin()) + 1
    assert summary["best_monitor_value"] == float(values.min())
    assert summary["history_feature_count"] == 4
    rates = [(h["learning_rate"], h["next_learning_rate"]) for h in history]
    assert all(rate > 0 and next_rate > 0 for rate, next_rate in rates)
    assert summary["learning_rate_schedule"]["reductions"] == sum(b < a for a, b in rates)
    assert len(history) <= config["gru_epochs"]
    if summary["stopped_early"]:
        # The absolute best checkpoint may improve by less than min_delta;
        # independently replay the significant-improvement patience counter.
        significant_best, wait = float("inf"), 0
        for value in values:
            if value < significant_best - config["min_delta"]:
                significant_best, wait = value, 0
            else:
                wait += 1
        assert wait >= config["gru_patience"]
    text = (folder / "training.log").read_text(encoding="utf-8")
    assert all(label in text for label in ("RMSE=", "MAE=", "R\u00b2=", "LR="))
    assert (folder / "gru.keras").stat().st_size > 0
    assert (folder / "scalers.joblib").stat().st_size > 0
    assert png_refs(summary, folder) == 1
    assert not list(folder.rglob("*metrics*.png"))
    return {"epochs": len(history), "best_epoch": summary["best_epoch"],
            "lr_reductions": summary["learning_rate_schedule"]["reductions"],
            "stop_reason": summary["stop_reason"]}

rolling = read(run / "rolling_validation.json")
assert rolling["test_used_for_selection"] is False
assert rolling["scope"] == "full_windows"
assert set(rolling["candidates"]) == {"gru32_control", "gru32_scheduled", "gru16_scheduled"}
audits = {}
for fold in rolling["folds"]:
    train_end, val_start, val_end = [pd.Timestamp(fold[key]) for key in
                                   ("train_end", "validation_start", "validation_end")]
    assert train_end + pd.Timedelta(minutes=15) == val_start
    assert val_end < pd.Timestamp(rolling["reserved_test_start"])
    assert fold["training_windows_used"] == fold["training_windows_available"]
    assert fold["training_windows_used"] == int(fold["training_days"] * 96) - 383
    val_rows = int((val_end-val_start)/pd.Timedelta(minutes=15)) + 1
    assert fold["validation_windows"] == val_rows - 95
    for candidate in rolling["candidates"]:
        folder = run / "experiments" / fold["name"] / candidate
        if args.partial and not (folder / "metrics.json").exists():
            continue
        features = read(folder / "features.json")
        assert features["history"] == ["pv_power_kw", "SWGDN", "temperature_c", "cloud_cover"]
        assert features["future"][-1] == "solar_elevation_deg" and len(features["future"]) == 6
        assert not features["training_weather_issue_times"]
        audited = training_audit(folder, fold["training_windows_used"])
        scaler_audit(folder, fold["train_end"], rolling["input_spec"], fold["training_windows_used"])
        audited["scalers_match_training_range"] = True
        metrics = read(folder / "metrics.json")
        assert metrics["all_day"]["count"] == fold["validation_windows"] * 96
        np.testing.assert_allclose(metrics["all_day"]["rmse"],
                                   fold["results"][candidate]["all_day"]["rmse"])
        audits[fold["name"] + "/" + candidate] = audited

if not args.partial:
    assert rolling["status"] == "completed"
    assert len(rolling["folds"]) == 3 and len(audits) == 9
    for name, aggregate in rolling["aggregate"].items():
        for period in ("all_day", "daylight"):
            folds = [f["results"][name][period] for f in rolling["folds"]]
            count = sum(f["count"] for f in folds)
            assert aggregate[period]["count"] == count
            rmse = np.sqrt(sum(f["count"] * f["rmse"] ** 2 for f in folds) / count)
            mae = sum(f["count"] * f["mae"] for f in folds) / count
            np.testing.assert_allclose([aggregate[period]["rmse"], aggregate[period]["mae"]],
                                      [rmse, mae], rtol=1e-10)
    best = min(rolling["aggregate"], key=lambda key: rolling["aggregate"][key]["all_day"]["rmse"])
    assert rolling["selected_candidate"] == best
    metadata = read(run / "model_metadata.json")
    metrics = read(run / "metrics.json")
    assert metadata["selected_model"] == ("gru" if best in rolling["candidates"] else best)
    assert metadata["selected_model"] == metrics["selected_model"]
    assert metadata["selection_source"] == "rolling_validation"
    assert metadata["selected_gru_candidate"] == rolling["best_gru_candidate"]
    assert metadata["config"] == rolling["selected_config"]
    assert metadata["publication_status"] == "candidate"
    assert "retrospective_report_only" in metadata["test_usage"]
    assert metadata["trained_gru_available"]
    assert metadata["input_quality"]["missing_target_rows"] == 0
    assert metadata["input_quality"]["missing_weather_rows"] == 0
    assert metadata["baseline_fallback"]["calibration"]["calibration_samples"] == 5161
    audits["final"] = training_audit(run, 24145)
    scaler_audit(run, metadata["training_data_window"]["end"], metadata["input_spec"], 24145)
    audits["final"]["scalers_match_training_range"] = True
    assert metrics["test"]["count"] == 5161 * 96
    assert metrics["gru"]["test"]["count"] == 5161 * 96
    assert png_refs(metrics, run) == 1
    assert png_refs(read(run / "forecast.json"), run) == 3
    assert png_refs(read(run / "evaluation.json"), run) == 1
    result = {"status": "passed", "training_artifacts": audits, "selected_candidate": best,
              "test_metrics": metrics["test"], "data_kind": "simulated"}
    (Path(__file__).parent / "artifact_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
else:
    result = {"status": "in_progress", "completed_training_artifacts": audits}
    (Path(__file__).parent / "artifact_audit_partial.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
