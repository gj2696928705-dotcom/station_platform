"""Reproduce the frozen candidate, without tuning against the already viewed test set."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys

LOAD_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LOAD_ROOT))

import numpy as np
import pandas as pd
from load_forecast.data import InputSpec, baseline_predictions, features, labels, load_data, valid_origins
from load_forecast.models import calibrate, interval, metrics, predict_xgboost, xgb_module
from load_forecast.pipeline import (TrainingConfig, _load_bundle, dump, evaluate_bundle,
                                    predict_record, sha256, train_model)


def score_bundle(bundle, data, stride):
    meta, model = _load_bundle(bundle)
    values, index = data.frame.base_load_kw.to_numpy(), data.frame.index
    start = index.get_loc(pd.Timestamp(meta["calibration_data_window"]["end_exclusive"]))
    origins = valid_origins(values, start, len(values), stride)
    x, names = features(values, index, origins)
    pred = predict_xgboost(model, x, names, 4)
    truth = labels(values, origins)
    lo, hi = interval(pred, meta["calibration"])
    baselines = baseline_predictions(values, origins)
    summary = {"origin_count": len(origins), "stride": stride, "all": metrics(truth, pred, lo, hi),
               "baselines": {k: metrics(truth, v) for k, v in baselines.items()},
               "first_hour": metrics(truth[:, :4], pred[:, :4], lo[:, :4], hi[:, :4]),
               "leads": [metrics(truth[:, i], pred[:, i], lo[:, i], hi[:, i]) for i in range(96)]}
    summary["by_origin_minute"] = {
        str(minute): metrics(truth[index[origins].minute == minute], pred[index[origins].minute == minute],
                             lo[index[origins].minute == minute], hi[index[origins].minute == minute])
        for minute in sorted(set(index[origins].minute))}
    summary["by_origin_month"] = {
        str(month): metrics(truth[index[origins].month == month], pred[index[origins].month == month],
                            lo[index[origins].month == month], hi[index[origins].month == month])
        for month in sorted(set(index[origins].month))}
    independent_targets = (origins - origins[0]) % 96 == 0
    summary["nonoverlapping_daily_origins"] = metrics(truth[independent_targets], pred[independent_targets],
                                                      lo[independent_targets], hi[independent_targets])
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = LOAD_ROOT.parent / "数据/负荷数据/黄花站_2025年_15分钟负荷模拟数据.csv"
    original = LOAD_ROOT / "运行结果/20260928_232203_101506"
    meta, model = _load_bundle(original)
    data = load_data(source, InputSpec(**meta["input_spec"]))
    report = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
              "xgboost": xgb_module().__version__, "source": str(source), "input_quality": data.quality,
              "source_sha256": sha256(source), "original_model_sha256": sha256(original / "xgboost.json"),
              "evidence_scope": "2025 simulated data; previously viewed holdout; no hyperparameter tuning"}
    assert report["source_sha256"] == meta["source_sha256"]
    report["original_stride4"] = score_bundle(original, data, 4)
    stored = json.loads((original / "metrics.json").read_text(encoding="utf-8"))
    for key in ("rmse_kw", "mae_kw", "r2", "coverage"):
        assert abs(report["original_stride4"]["all"][key] - stored["test"][key]) < 1e-9, key
    report["stored_metrics_reproduced"] = True
    # Recompute the independent calibration from its declared time window.
    values, index = data.frame.base_load_kw.to_numpy(), data.frame.index
    window = meta["calibration_data_window"]
    origins = valid_origins(values, index.get_loc(pd.Timestamp(window["start"])),
                            index.get_loc(pd.Timestamp(window["end_exclusive"])), 4)
    x, names = features(values, index, origins)
    cal = calibrate(labels(values, origins), predict_xgboost(model, x, names), .8)
    np.testing.assert_allclose(cal["quantiles_kw"], meta["calibration"]["quantiles_kw"], rtol=0, atol=0)
    report["calibration_exactly_reproduced"] = True
    report["all_quarter_hour_origins"] = score_bundle(original, data, 1)
    record = predict_record(original, source, out / "replay_20260101.json", as_of="2026-01-01T00:07:00+08:00")
    assert record["status"] == "available", record["reason"]
    report["last_day"] = evaluate_bundle(original, source, out / "last_day.json")
    report["current_clock"] = predict_record(original, source, as_of="2026-10-27T12:00:00+08:00")["status"]
    assert report["current_clock"] == "unavailable"  # 2025 data cannot masquerade as current history.
    if args.retrain:
        fresh = out / "full_retrain"
        result = train_model(source, fresh, InputSpec(**meta["input_spec"]),
                             TrainingConfig(**meta["config"]), progress=print)
        report["full_retrain"] = {"config": asdict(TrainingConfig(**meta["config"])),
            "model_sha256": sha256(fresh / "xgboost.json"),
            "weights_identical_to_original": sha256(fresh / "xgboost.json") == meta["xgboost_sha256"],
            "selected_model": result["metadata"]["selected_model"],
            "training": result["metrics"]["xgboost"]["training"], "test": result["metrics"]["test"]}
        report["full_retrain"]["replay_status"] = predict_record(fresh, source, fresh / "forecast.json")["status"]
        report["full_retrain"]["last_day_status"] = evaluate_bundle(fresh, source, fresh / "evaluation.json")["status"]
        for key in ("rmse_kw", "mae_kw", "r2", "coverage"):
            assert abs(result["metrics"]["test"][key] - stored["test"][key]) < 1e-9, key
    dump(out / "algorithm_verification.json", report)
    print(json.dumps({"report": str(out / "algorithm_verification.json"),
                      "test": report["original_stride4"]["all"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
