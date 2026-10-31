"""Offline training/selection/calibration/test, and read-only inference."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from .data import (DataValidationError, HORIZON, HISTORY, INTERVAL, TARGETS, InputSpec,
                   baseline_predictions, features, labels, load_data, timestamp, valid_origins)
from .models import calibrate, interval, metrics, predict_xgboost, train_xgboost, xgb_module
from .visuals import forecast_plot, training_plots


@dataclass(frozen=True)
class TrainingConfig:
    rounds: int = 500
    patience: int = 30
    max_depth: int = 6
    learning_rate: float = .05
    stride: int = 4
    threads: int = 4
    seed: int = 42
    nominal_coverage: float = .8
    train_fraction: float = .70
    validation_fraction: float = .15
    model_id: str = "huanghua-load-xgboost"

    def __post_init__(self):
        if any(not isinstance(x, int) or x <= 0 for x in
               (self.rounds, self.patience, self.max_depth, self.stride, self.threads)):
            raise ValueError("rounds, patience, max_depth, stride and threads must be positive integers")
        if not 0 < self.learning_rate <= 1 or not 0 < self.nominal_coverage < 1:
            raise ValueError("invalid learning_rate or nominal_coverage")
        if not 0 < self.train_fraction < 1 or not 0 < self.validation_fraction < 1:
            raise ValueError("invalid split fractions")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("test partition must be nonempty")


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_line(name, value):
    r2 = "N/A" if value["r2"] is None else f"{value['r2']:.4f}"
    return f"{name}: RMSE={value['rmse_kw']:.4f} kW | MAE={value['mae_kw']:.4f} kW | R²={r2}"


def train_model(source, output_dir, input_spec=InputSpec(), config=TrainingConfig(), progress=None):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty; previous evidence is never overwritten")
    output.mkdir(parents=True, exist_ok=True)

    def emit(message):
        with (output / "training.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
        if progress:
            progress(message)

    dump(output / "training_config.json", {"config": asdict(config), "input_spec": asdict(input_spec)})
    try:
        return _train(source, output, input_spec, config, emit)
    except Exception as exc:
        dump(output / "failure.json", {"status": "failed", "error_type": type(exc).__name__, "reason": str(exc)})
        emit(f"Training failed: {exc}")
        raise


def _train(source, output, spec, config, emit):
    data = load_data(source, spec)
    values, index = data.frame[spec.target_col].to_numpy(float), data.frame.index
    n = len(values)
    train_end = int(n * config.train_fraction)
    val_end = int(n * (config.train_fraction + config.validation_fraction))
    selection_end = train_end + (val_end - train_end) // 2
    if train_end < 28 * HORIZON:
        raise DataValidationError("at least 28 days are required in the training partition")
    boundaries = {"train": (0, train_end), "selection": (train_end, selection_end),
                  "calibration": (selection_end, val_end), "test": (val_end, n)}
    origins, split = {}, {}
    for part, (start, stop) in boundaries.items():
        origins[part] = valid_origins(values, start, stop, config.stride)
        if len(origins[part]) < 20:
            raise DataValidationError(f"{part}: fewer than 20 complete history/target windows")
        split[part] = {"start": index[start].isoformat(), "end_exclusive": (index[stop - 1] + INTERVAL).isoformat(),
                       "rows": stop - start, "days": (stop - start) / HORIZON,
                       "origin_count": len(origins[part]), "supervised_rows": len(origins[part]) * HORIZON,
                       "first_origin": index[origins[part][0]].isoformat(),
                       "last_origin": index[origins[part][-1]].isoformat()}
        emit(f"{part}: {split[part]['days']:g} days, {len(origins[part])} origins, "
             f"{split[part]['supervised_rows']} (origin, lead) rows")
    dump(output / "data_split.json", split)
    emit(f"Target={spec.target_col}, source={data.quality['source_kind']}; all hourly origins at stride={config.stride} quarter-hours")
    xtrain, names = features(values, index, origins["train"])
    xval, _ = features(values, index, origins["selection"])
    ytrain, yval = labels(values, origins["train"]), labels(values, origins["selection"])
    model, summary = train_xgboost(xtrain, ytrain, xval, yval, names, config, output, emit)
    del xtrain, ytrain
    val_predictions = {f"baseline:{name}": pred for name, pred in baseline_predictions(values, origins["selection"]).items()}
    val_predictions["xgboost"] = predict_xgboost(model, xval, names, config.threads)
    del xval
    val_metrics = {name: metrics(yval, pred) for name, pred in val_predictions.items()}
    selected = min(val_metrics, key=lambda key: val_metrics[key]["rmse_kw"])
    best_baseline = min((key for key in val_metrics if key.startswith("baseline:")), key=lambda key: val_metrics[key]["rmse_kw"])
    for name, value in val_metrics.items():
        emit(_metric_line(f"Selection {name}", value))
    emit(f"Selected by validation RMSE: {selected}; XGBoost weights retained independently")
    xcal, _ = features(values, index, origins["calibration"])
    xtest, _ = features(values, index, origins["test"])
    cal_predictions = {f"baseline:{k}": v for k, v in baseline_predictions(values, origins["calibration"]).items()}
    test_predictions = {f"baseline:{k}": v for k, v in baseline_predictions(values, origins["test"]).items()}
    cal_predictions["xgboost"] = predict_xgboost(model, xcal, names, config.threads)
    test_predictions["xgboost"] = predict_xgboost(model, xtest, names, config.threads)
    ycal, ytest = labels(values, origins["calibration"]), labels(values, origins["test"])
    calibrations, test_metrics = {}, {}
    for name in test_predictions:
        calibrations[name] = calibrate(ycal, cal_predictions[name], config.nominal_coverage)
        lo, hi = interval(test_predictions[name], calibrations[name])
        test_metrics[name] = metrics(ytest, test_predictions[name], lo, hi)
    # Calibration labels are never used in early stopping or model selection.
    calibration = {**calibrations[selected], "model": selected, "data_window": split["calibration"]}
    model_version = output.name + "-" + uuid.uuid4().hex[:8]
    metadata = {"schema_version": "load-model-bundle-1", "model_id": config.model_id,
                "model_version": model_version, "selected_model": selected, "input_spec": asdict(spec),
                "target_semantics": TARGETS[spec.target_col], "feature_names": names,
                "interval_minutes": 15, "history_steps": HISTORY, "horizon_steps": HORIZON,
                "publication_status": "candidate", "model_published_at": None,
                "created_at": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
                "training_data_window": split["train"], "selection_data_window": split["selection"],
                "calibration_data_window": split["calibration"], "input_quality": data.quality,
                "source_sha256": sha256(source) if not isinstance(source, pd.DataFrame) else None,
                "xgboost_sha256": sha256(output / "xgboost.json"), "calibration": calibration}
    result = {"selected_model": selected, "selection": val_metrics, "test_all_models": test_metrics,
              "test": test_metrics[selected], "xgboost": {"training": summary, "test": test_metrics["xgboost"]},
              "evaluation_scope": "all complete stride-spaced rolling origins; overlapping target timestamps count per origin",
              "calibration_scope": "second half of validation, separate from early stopping and selection",
              "source_kind": data.quality["source_kind"], "visualizations": [], "visualization_errors": []}
    try:
        history = json.loads((output / "training_history.json").read_text(encoding="utf-8"))
        result["visualizations"] = training_plots(output, history, index, ytest, test_predictions["xgboost"],
                                                 test_predictions[best_baseline], origins["test"])
    except Exception as exc:
        result["visualization_errors"].append(str(exc))
        emit(f"Visualization failed: {exc}")
    dump(output / "model_metadata.json", metadata)
    dump(output / "training_summary.json", summary)
    dump(output / "interval_calibration.json", calibration)
    dump(output / "metrics.json", result)
    emit(_metric_line("XGBoost held-out test", test_metrics["xgboost"]))
    emit(_metric_line("Selected held-out test", test_metrics[selected]))
    emit(f"Nominal coverage={config.nominal_coverage:.0%}; observed test coverage={test_metrics[selected]['coverage']:.2%}")
    return {"metadata": metadata, "metrics": result, "output_dir": str(output.resolve())}


def _load_bundle(bundle):
    bundle = Path(bundle)
    meta = json.loads((bundle / "model_metadata.json").read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or meta["schema_version"] != "load-model-bundle-1" or (
            meta["horizon_steps"] != HORIZON or meta["history_steps"] != HISTORY or meta["interval_minutes"] != 15):
        raise DataValidationError("unsupported model bundle")
    spec = InputSpec(**meta["input_spec"])
    if meta["target_semantics"] != TARGETS[spec.target_col]:
        raise DataValidationError("bundle target semantics differ from input specification")
    if meta["selected_model"] != meta["calibration"]["model"]:
        raise DataValidationError("calibration/model identity mismatch")
    q = np.asarray(meta["calibration"]["quantiles_kw"], float)
    coverage = meta["calibration"]["nominal_coverage"]
    if q.shape != (HORIZON,) or not np.isfinite(q).all() or (q < 0).any() or (
            isinstance(coverage, bool) or not isinstance(coverage, (int, float)) or not 0 < coverage < 1):
        raise DataValidationError("invalid interval calibration")
    model = None
    if meta["selected_model"] == "xgboost":
        if sha256(bundle / "xgboost.json") != meta["xgboost_sha256"]:
            raise DataValidationError("model checksum mismatch")
        model = xgb_module().Booster()
        # Python handles Unicode paths; avoid passing Windows ANSI paths to the C API.
        model.load_model(bytearray((bundle / "xgboost.json").read_bytes()))
    elif meta["selected_model"] not in {f"baseline:{k}" for k in ("persistence", "previous_day", "previous_week", "same_slot_mean")}:
        raise DataValidationError("unknown selected model")
    return meta, model


def validate_record(record):
    required = {"forecast_id", "forecast_type", "asset_scope", "generated_at", "status", "forecast_values",
                "prediction_interval", "target_semantics", "decision_eligible", "audit_refs", "schema_version",
                "interval_minutes", "forecast_start_at", "forecast_end_at", "reason", "publication_status"}
    if not isinstance(record, dict) or required - set(record):
        raise ValueError("forecast record missing required fields")
    if record["schema_version"] != "station-load-forecast-1" or record["forecast_type"] != "station_load":
        raise ValueError("unsupported forecast schema/type")
    if type(record["interval_minutes"]) is not int or record["interval_minutes"] != 15:
        raise ValueError("load forecast interval must be 15 minutes")
    if record["decision_eligible"] is not False or record["publication_status"] != "candidate":
        raise ValueError("candidate forecasts cannot be decision eligible")
    def aware_time(value):
        result = pd.Timestamp(value)
        if pd.isna(result) or result.tzinfo is None:
            raise ValueError("forecast timestamps must include a timezone")
        return result
    aware_time(record["generated_at"])
    if not isinstance(record["forecast_id"], str) or not record["forecast_id"] or not isinstance(record["audit_refs"], list):
        raise ValueError("invalid forecast identity/audit references")
    pi = record["prediction_interval"]
    if not isinstance(record["forecast_values"], list) or not isinstance(pi, dict) or not isinstance(pi.get("values"), list):
        raise ValueError("invalid forecast array/interval structure")
    if record["status"] == "unavailable":
        if record["forecast_values"] or pi["values"] or pi.get("available") is not False:
            raise ValueError("unavailable record cannot contain invented predictions")
        if not isinstance(record["reason"], str) or not record["reason"]:
            raise ValueError("unavailable record must include a reason")
        return
    values = record["forecast_values"]
    bounds = pi["values"]
    if record["status"] != "available" or len(values) != HORIZON or len(bounds) != HORIZON or pi.get("available") is not True:
        raise ValueError("invalid forecast length/status")
    if record["target_semantics"] not in TARGETS.values() or not isinstance(record["asset_scope"], str) or not record["asset_scope"]:
        raise ValueError("invalid forecast target/asset scope")
    coverage = pi.get("nominal_coverage")
    if isinstance(coverage, bool) or not isinstance(coverage, (int, float)) or not 0 < coverage < 1:
        raise ValueError("invalid nominal coverage")
    start = aware_time(record["forecast_start_at"])
    if start != start.floor("15min"):
        raise ValueError("forecast start must align to 15 minutes")
    for i, (point, bound) in enumerate(zip(values, bounds)):
        if not isinstance(point, dict) or not isinstance(bound, dict) or (
                {"value", "timestamp", "unit"} - set(point) or {"lower", "upper", "timestamp", "unit"} - set(bound)):
            raise ValueError("invalid forecast point structure")
        nums = [point["value"], bound["lower"], bound["upper"]]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in nums) or (
                not np.isfinite(nums).all() or not 0 <= nums[1] <= nums[0] <= nums[2]):
            raise ValueError("invalid forecast values/bounds")
        if point["unit"] != "kW" or bound["unit"] != "kW":
            raise ValueError("forecast point and interval units must be kW")
        if aware_time(point["timestamp"]) != start + i * INTERVAL or aware_time(bound["timestamp"]) != aware_time(point["timestamp"]):
            raise ValueError("forecast timeline mismatch")
    if aware_time(record["forecast_end_at"]) != start + HORIZON * INTERVAL:
        raise ValueError("forecast_end_at is exclusive and must span 24 hours")


def predict_record(bundle_dir, source, output_path=None, input_spec=None, forecast_start=None, as_of=None):
    meta = {}
    record = {"schema_version": "station-load-forecast-1", "forecast_id": str(uuid.uuid4()),
              "forecast_type": "station_load", "asset_scope": None,
              "generated_at": datetime.now(timezone.utc).isoformat(), "interval_minutes": 15,
              "forecast_start_at": None, "forecast_end_at": None, "forecast_values": [],
              "prediction_interval": {"available": False, "values": []},
              "status": "unavailable", "reason": None, "target_semantics": None,
              "model_id": None, "model_version": None, "model_published_at": None,
              "selected_model": None, "training_data_window": None, "fallback_forecast_ref": None,
              "publication_status": "candidate", "decision_eligible": False,
              "execution_mode": "offline_replay", "input_data_quality_summary": {},
              "confidence": {"source": "held_out_empirical_residuals", "limitations": [
                  "Candidate model, no production publication or energy Agent integration",
                  "Forecast is not a device limit; synthetic-data metrics are not field accuracy"]},
              "audit_refs": [str(Path(bundle_dir).resolve() / "model_metadata.json")],
              "visualizations": [], "visualization_errors": []}
    try:
        meta, model = _load_bundle(bundle_dir)
        spec = InputSpec(**meta["input_spec"])
        if input_spec is not None and input_spec != spec:
            raise DataValidationError("prediction InputSpec differs from training bundle; target/station/unit cannot change")
        record.update({"asset_scope": spec.station_id, "target_semantics": meta["target_semantics"],
                       "model_id": meta["model_id"], "model_version": meta["model_version"],
                       "selected_model": meta["selected_model"], "training_data_window": meta["training_data_window"]})
        data = load_data(source, spec)
        if data.quality["source_kind"] != meta["input_quality"]["source_kind"]:
            raise DataValidationError("simulation provenance differs from model training")
        clock = timestamp(as_of, spec.timezone) if as_of is not None else None
        start = timestamp(forecast_start, spec.timezone) if forecast_start is not None else (
            clock.floor("15min") if clock is not None else data.frame.index[-1] + INTERVAL)
        if start != start.floor("15min"):
            raise DataValidationError("forecast start must align to 15 minutes")
        if clock is not None and start != clock.floor("15min"):
            raise DataValidationError("as_of replay must start at the current completed-interval boundary")
        # Labels of intervals [t,t+15min) become available only at t+15min.
        past = data.frame.loc[data.frame.index < start]
        if len(past) < HISTORY or past.index[-1] + INTERVAL != start:
            raise DataValidationError("insufficient or stale history for requested forecast start")
        if start < timestamp(meta["calibration_data_window"]["end_exclusive"], spec.timezone):
            raise DataValidationError("forecast origin precedes training/selection/calibration cutoff")
        history = past.iloc[-HISTORY:]
        values = history[spec.target_col].to_numpy(float)
        if not np.isfinite(values).all():
            raise DataValidationError("history contains missing observations; no fill or silent fallback")
        x, names = features(values, history.index, np.array([len(history)]))
        if names != meta["feature_names"]:
            raise DataValidationError("feature schema differs from training")
        if meta["selected_model"] == "xgboost":
            pred = predict_xgboost(model, x, names, meta["config"]["threads"])[0]
        else:
            pred = baseline_predictions(values, np.array([len(history)]))[meta["selected_model"].split(":")[1]][0]
        lo, hi = interval(pred, meta["calibration"])
        times = pd.date_range(start, periods=HORIZON, freq="15min")
        record.update({"status": "available", "forecast_start_at": start.isoformat(),
                       "forecast_end_at": (start + HORIZON * INTERVAL).isoformat(),
                       "forecast_values": [{"timestamp": t.isoformat(), "value": float(v), "unit": "kW"} for t, v in zip(times, pred)],
                       "prediction_interval": {"available": True, "method": meta["calibration"]["method"],
                           "nominal_coverage": meta["calibration"]["nominal_coverage"],
                           "values": [{"timestamp": t.isoformat(), "lower": float(l), "upper": float(h), "unit": "kW"}
                                      for t, l, h in zip(times, lo, hi)]},
                       "predicted_energy_kwh": float(pred.sum() * .25),
                       "as_of": clock.isoformat() if clock is not None else None,
                       "input_data_quality_summary": {**data.quality, "used_history_start": history.index[0].isoformat(),
                           "used_history_end_exclusive": start.isoformat(),
                           "future_rows_excluded": int((data.frame.index >= start).sum())}})
        if spec.target_col == "total_load_kw":
            record["confidence"]["limitations"].append("Includes charging; do not add charging plans again")
        validate_record(record)
        if output_path:
            try:
                record["visualizations"].append(forecast_plot(Path(output_path).parent, data, record))
            except Exception as exc:
                record["visualization_errors"].append(str(exc))
    except (OSError, ValueError, KeyError, RuntimeError, TypeError, IndexError, AttributeError) as exc:
        record.update({"status": "unavailable", "reason": str(exc), "forecast_values": [],
                       "prediction_interval": {"available": False, "values": []}})
    validate_record(record)
    if output_path:
        dump(output_path, record)
    return record


def evaluate_bundle(bundle_dir, source, output_path=None):
    meta, _ = _load_bundle(bundle_dir)
    spec = InputSpec(**meta["input_spec"])
    data = load_data(source, spec)
    if len(data.frame) < HISTORY + HORIZON:
        raise DataValidationError("evaluation requires 7 days history and 1 day labels")
    start = data.frame.index[-HORIZON]
    # predict_record masks every input at/after start even when source includes labels.
    record = predict_record(bundle_dir, source, forecast_start=start)
    result = {"status": record["status"], "scope": "last 24 hours only; full held-out test is in metrics.json",
              "forecast_start_at": start.isoformat(), "selected_model": record["selected_model"],
              "metrics": None, "reason": record["reason"], "visualizations": [], "visualization_errors": []}
    if record["status"] == "available":
        truth = data.frame[spec.target_col].iloc[-HORIZON:].to_numpy(float)
        if not np.isfinite(truth).all():
            result.update(status="unavailable", reason="evaluation labels contain missing values")
        else:
            pred = np.array([v["value"] for v in record["forecast_values"]])
            bounds = record["prediction_interval"]["values"]
            result["metrics"] = metrics(truth, pred, np.array([v["lower"] for v in bounds]), np.array([v["upper"] for v in bounds]))
            if output_path:
                try:
                    result["visualizations"].append(forecast_plot(Path(output_path).parent, data, record, truth, "last_day_backtest.png"))
                except Exception as exc:
                    result["visualization_errors"].append(str(exc))
    if output_path:
        dump(output_path, result)
    return result
