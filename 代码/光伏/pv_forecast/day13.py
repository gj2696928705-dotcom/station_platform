"""Reference-style daily dual-input GRU training and 07:00–19:00 inference."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import uuid

import joblib
import numpy as np
import pandas as pd

from .baselines import BaselineForecaster
from .data import DataValidationError, InputSpec, _read_source
from .day13_data import (DAY_LEN, FUTURE_COLUMNS, LAYOUT, build_day_samples, complete_days,
                         day_index, forecast_start, numeric_block, read_day_frame, subset)
from .metrics import apply_prediction_interval, calibrate_prediction_interval, regression_metrics
from .models import GRUForecaster
from .pipeline import (TrainingConfig, _fit_scalers, _fit_history_scaler, _scale_windows,
                       _inverse_target, _gru_config, _json_dump)
from .records import build_forecast_record, validate_forecast_record
from .weather import WeatherForecastArchive


@dataclass
class Day13TrainingConfig(TrainingConfig):
    interval_minutes: int = 60
    history_days: int = 3
    horizon_steps: int = 13
    min_gru_days: int = 0
    require_gru: bool = True
    gru_batch_size: int = 32
    model_version: str = "0.3.0-candidate+day13"
    sampling_layout: str = LAYOUT


def validate_spec(spec):
    if spec.history_weather_columns:
        raise DataValidationError("day13 historical input is power only; remove history weather columns")
    if spec.weather_columns and tuple(spec.weather_columns) != FUTURE_COLUMNS:
        raise DataValidationError("day13 future columns must be SWGDN,DHI,DNI,rain, in that order")


def _archive(source, timezone):
    return WeatherForecastArchive(source, FUTURE_COLUMNS, timezone) if source is not None else None


def _weather_policy(archive):
    return "issued_forecast" if archive is not None else "provided_target_day_weather"


def _limitations(policy):
    return (["按参考代码使用提供的当日辐照和降雨；其可提前获得性未经证明，离线精度以这些未来输入已知为条件"]
            if policy == "provided_target_day_weather" else ["仅使用07:00前已发布且完整覆盖13个小时的同一批天气预报"])


def _baseline_candidates(windows):
    choices = []
    for strategy in ("persistence", "previous_day", "same_slot_mean"):
        model = BaselineForecaster(strategy, steps_per_day=13)
        pred = model.predict(windows.history_power, 13)
        choices.append((regression_metrics(windows.target, pred)["rmse"], model))
    return min(choices, key=lambda pair: pair[0])[1]


def _predict(windows, model, scalers):
    scaled = _scale_windows(windows, **scalers)
    return _inverse_target(model.predict(scaled.gru_history, scaled.future_features), scalers["target_scaler"])


def train_day13(source, output_dir, input_spec=InputSpec(), config=None, progress=None, future_weather_source=None):
    config = config or Day13TrainingConfig()
    validate_spec(input_spec)
    if (config.interval_minutes, config.history_days, config.horizon_steps) != (60, 3, 13):
        raise DataValidationError("day13 contract is fixed: 60 minutes, previous 3 days, 13 hourly outputs")
    if config.max_gru_samples:
        raise DataValidationError("day13 training uses all daily samples; sample caps are not supported")
    if config.require_gru and config.skip_gru:
        raise DataValidationError("require_gru and skip_gru conflict")
    if not (0 < config.train_ratio < 1 and 0 < config.val_ratio < 1-config.train_ratio):
        raise DataValidationError("invalid chronological train/validation/test ratios")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise DataValidationError("output directory must be new or empty; existing weights are preserved")
    output.mkdir(parents=True, exist_ok=True)

    def emit(message):
        with (output/"training.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
        if progress:
            progress(message)

    frame = read_day_frame(source, input_spec)
    days = complete_days(frame)
    archive = _archive(future_weather_source, input_spec.timezone)
    windows = build_day_samples(frame, input_spec, archive)
    train_end = int(len(days)*config.train_ratio)
    val_end = int(len(days)*(config.train_ratio+config.val_ratio))
    sample_day = pd.DatetimeIndex(windows.forecast_starts).normalize()
    if train_end < 4 or val_end >= len(days):
        raise DataValidationError("insufficient complete days for chronological partitions")
    train_ids = np.flatnonzero(sample_day < days[train_end])
    val_ids = np.flatnonzero((sample_day >= days[train_end]) & (sample_day < days[val_end]))
    test_ids = np.flatnonzero(sample_day >= days[val_end])
    if not len(train_ids) or len(val_ids) < 2 or not len(test_ids):
        raise DataValidationError("need nonempty train/test and at least two validation days")
    train = subset(windows, train_ids)
    selection = subset(windows, val_ids[:len(val_ids)//2])
    calibration_days = subset(windows, val_ids[len(val_ids)//2:])
    test = subset(windows, test_ids)
    baseline = _baseline_candidates(selection)
    target_scaler, feature_scaler = _fit_scalers(train)
    scalers = {"target_scaler": target_scaler, "feature_scaler": feature_scaler,
               "history_scaler": _fit_history_scaler(train)}
    joblib.dump(scalers, output/"scalers.joblib")
    _json_dump(output/"training_config.json", asdict(config))
    split = {name: {"samples": data.size, "target_start": data.forecast_starts[0].isoformat(),
        "target_end": (data.forecast_starts[-1] + pd.Timedelta(hours=12)).isoformat()}
        for name, data in (("train", train), ("selection_validation", selection),
                           ("interval_calibration", calibration_days), ("test", test))}
    _json_dump(output/"data_split.json", split)
    emit(f"按13行一天组织：历史{train.history_power.shape}，未来{train.future_features.shape}，目标{train.target.shape}。")
    model = None
    if not config.skip_gru:
        if train.size < config.min_gru_days:
            raise DataValidationError("fewer complete training days than min_gru_days")
        model = GRUForecaster(config=_gru_config(config))
        scaled_train = _scale_windows(train, **scalers)
        scaled_val = _scale_windows(selection, **scalers)
        model.fit(scaled_train.gru_history, scaled_train.future_features, scaled_train.target,
                  validation_data=(scaled_val.gru_history, scaled_val.future_features, scaled_val.target),
                  target_scaler=target_scaler, output_dir=output, progress=emit)
        model.save(str(output/"gru.keras"))
    selected = "gru" if model is not None else "baseline:" + baseline.strategy
    predict = (lambda data: _predict(data, model, scalers)) if model is not None else (
        lambda data: baseline.predict(data.history_power, 13))
    cal_pred = predict(calibration_days)
    calibration = calibrate_prediction_interval(calibration_days.target, cal_pred, config.nominal_coverage)
    baseline_cal = calibrate_prediction_interval(calibration_days.target,
        baseline.predict(calibration_days.history_power, 13), config.nominal_coverage)
    test_pred = predict(test)
    lo, hi = apply_prediction_interval(test_pred, calibration)
    metrics = {"status": "completed", "selected_model": selected,
        "validation": regression_metrics(selection.target, predict(selection)),
        "test": regression_metrics(test.target, test_pred, lo, hi),
        "baseline_test": regression_metrics(test.target, baseline.predict(test.history_power, 13)),
        "gru": {"status": "completed", "training": model.training_summary} if model else {"status": "explicitly_skipped"},
        "per_hour_test": {str(hour): regression_metrics(test.target[:, i], test_pred[:, i]) for i, hour in enumerate(range(7, 20))},
        "evaluation_scope": "one 13-point sample per complete calendar day; held-out target days only",
        "weather_input_policy": _weather_policy(archive), "limitations": _limitations(_weather_policy(archive)),
        "visualizations": [], "visualization_errors": []}
    metadata = {"schema_version": "pv-model-bundle-2", "sampling_layout": LAYOUT,
        "model_id": config.model_id, "model_version": config.model_version, "publication_status": "candidate",
        "selected_model": selected, "interval_minutes": 60, "history_days": 3, "history_steps": 39,
        "horizon_steps": 13, "points_per_day": 13, "forecast_hours": list(range(7, 20)),
        "target_column": input_spec.target_col, "history_feature_names": [input_spec.target_col],
        "history_weather_columns": [], "feature_names": list(FUTURE_COLUMNS), "weather_columns": list(FUTURE_COLUMNS),
        "weather_input_policy": _weather_policy(archive), "input_spec": asdict(input_spec),
        "training_data_window": {"start": frame.index[0].isoformat(), "end": split["train"]["target_end"]},
        "data_splits": split, "calibration": calibration,
        "baseline_fallback": {"strategy": baseline.strategy, "calibration": baseline_cal},
        "all_daily_training_samples_used": True, "test_used_for_selection": False,
        "interval_calibration_excluded_from_early_stopping": True,
        "reference": "参考代码文档/光伏/主代码.py",
        "differences_from_reference": ["校验完整自然日而非截断尾行", "历史只取前三个自然日功率", "RMSE按均方误差开方一次", "训练专属缩放与最佳权重", "独立区间校准日"]}
    _json_dump(output/"model_metadata.json", metadata)
    _json_dump(output/"interval_calibration.json", calibration)
    from .visuals import generate_training_evaluation_visualizations
    refs, errors = generate_training_evaluation_visualizations(test, test_pred,
        baseline.predict(test.history_power, 13), output/"visualizations", 60)
    metrics.update(visualizations=refs, visualization_errors=errors)
    _json_dump(output/"metrics.json", metrics)
    rows = [{"timestamp": stamp.isoformat(), "actual_kw": float(actual), "forecast_kw": float(pred)}
        for start, truth, prediction in zip(test.forecast_starts, test.target, test_pred)
        for stamp, actual, pred in zip(day_index(start), truth, prediction)]
    pd.DataFrame(rows).to_csv(output/"test_predictions.csv", index=False, encoding="utf-8-sig")
    emit(f"完成：训练{train.size}天、早停{selection.size}天、独立校准{calibration_days.size}天、测试{test.size}天；每天07:00–19:00。")
    return {"metadata": metadata, "metrics": metrics, "output_dir": str(output)}


def _load(bundle_dir):
    bundle = Path(bundle_dir)
    meta = json.loads((bundle/"model_metadata.json").read_text(encoding="utf-8"))
    if (meta.get("sampling_layout"), meta.get("history_steps"), meta.get("horizon_steps"), meta.get("interval_minutes")) != (LAYOUT, 39, 13, 60):
        raise DataValidationError("incompatible day13 bundle contract")
    if meta["feature_names"] != list(FUTURE_COLUMNS) or meta["history_feature_names"] != [meta["target_column"]]:
        raise DataValidationError("incompatible day13 feature order")
    spec = InputSpec(**meta["input_spec"])
    model = scalers = None
    if meta["selected_model"] == "gru":
        try:
            model = GRUForecaster.load(str(bundle/"gru.keras"))
            scalers = joblib.load(bundle/"scalers.joblib")
            if model.model.input_shape != [(None, 39, 1), (None, 13, 4)]:
                raise DataValidationError("GRU input shape does not match day13 metadata")
        except Exception as exc:
            model = scalers = None
            meta["model_load_error"] = str(exc)
    return meta, spec, model, scalers


def predict_day13(bundle_dir, source, output_path=None, input_spec=InputSpec(), forecast_date=None,
                  asset_scope="configured_pv_assets", future_weather_source=None):
    meta, spec, model, scalers = _load(bundle_dir)
    # Caller column mapping is the only permitted override of a saved input contract.
    if input_spec.column_map:
        spec = replace(spec, column_map=input_spec.column_map)
    start = forecast_start(forecast_date, spec.timezone)
    frame = read_day_frame(source, spec)
    past_index = day_index(start-pd.Timedelta(days=3)).append([
        day_index(start-pd.Timedelta(days=2)), day_index(start-pd.Timedelta(days=1))])
    quality = {"sampling_layout": LAYOUT, "history_points_required": 39, "target_points": 13,
               "source_rows": len(frame), "history_days": 3, "interpolation": "none"}
    policy = meta["weather_input_policy"]
    limitations = _limitations(policy)
    status, used, issue = "available", meta["selected_model"], None
    prediction, history = [], None
    calibration = meta["calibration"]
    try:
        history = numeric_block(frame, past_index, [spec.target_col])
    except DataValidationError as exc:
        status = "unavailable"
        limitations.append(str(exc))
    if history is not None:
        quality["history_points_used"] = len(history)
        weather, reason = None, None
        if future_weather_source is not None:
            supplied_raw = _read_source(future_weather_source)
            if meta["weather_input_policy"] == "issued_forecast" or "issued_at" in supplied_raw:
                archive = _archive(future_weather_source, spec.timezone)
                weather, issue = archive.at(start, day_index(start))
                if weather is not None:
                    weather = numeric_block(weather, day_index(start), FUTURE_COLUMNS)
            else:
                supplied = read_day_frame(future_weather_source, spec)
                try:
                    weather = numeric_block(supplied, day_index(start), FUTURE_COLUMNS)
                except DataValidationError as exc:
                    reason = str(exc)
        elif policy == "provided_target_day_weather":
            try:
                weather = numeric_block(frame, day_index(start), FUTURE_COLUMNS)
            except DataValidationError as exc:
                reason = str(exc)
        if used == "gru" and (model is None or weather is None):
            status = "degraded"
            used = "baseline:"+meta["baseline_fallback"]["strategy"]
            calibration = meta["baseline_fallback"]["calibration"]
            limitations.append(reason or meta.get("model_load_error", "缺少完整的当日未来天气输入"))
            limitations.append("使用13点日基线及其独立校准区间")
        if used == "gru":
            from .data import WindowedData
            windows = WindowedData(history[None], weather[None], np.zeros((1, 13)), [start],
                FUTURE_COLUMNS, history[None], [spec.target_col])
            prediction = _predict(windows, model, scalers)[0]
        else:
            prediction = BaselineForecaster(used.split(":")[-1], 13).predict(history[None], 13)[0]
    lower, upper = (None, None) if not len(prediction) else [v[0] for v in apply_prediction_interval(np.asarray(prediction), calibration)]
    record = build_forecast_record(str(uuid.uuid4()), prediction, start, 60, asset_scope,
        meta["model_id"], meta["model_version"], None, meta["training_data_window"], calibration,
        quality, status=status, limitations=limitations, lower=lower, upper=upper,
        fallback_forecast_ref=used if status == "degraded" else None, audit_refs=["model_metadata.json"])
    record["confidence"].update(model_used=used, sampling_layout=LAYOUT, weather_input_policy=policy,
        weather_forecast_issued_at=issue.isoformat() if issue is not None else None,
        runtime_weather_input_policy="issued_forecast" if issue is not None else policy,
        history_input_columns=[spec.target_col], future_input_columns=list(FUTURE_COLUMNS))
    if output_path and len(prediction):
        from .visuals import generate_forecast_visualizations
        refs, errors = generate_forecast_visualizations(past_index, history[:, 0], record, Path(output_path).parent/"visualizations")
        record.update(visualizations=refs, visualization_errors=errors)
    validate_forecast_record(record)
    if output_path:
        _json_dump(Path(output_path), record)
    return record


def evaluate_day13(bundle_dir, source, input_spec=InputSpec(), visualizations_dir=None, future_weather_source=None):
    meta, spec, model, scalers = _load(bundle_dir)
    if input_spec.column_map:
        from dataclasses import replace
        spec = replace(spec, column_map=input_spec.column_map)
    archive = _archive(future_weather_source, spec.timezone)
    if meta["weather_input_policy"] == "issued_forecast" and archive is None:
        raise DataValidationError("this bundle requires an issued forecast archive for evaluation")
    frame = read_day_frame(source, spec)
    windows = build_day_samples(frame, spec, archive)
    degraded = meta["selected_model"] == "gru" and model is None
    selected = meta["baseline_fallback"] if degraded else None
    if model is not None:
        prediction = _predict(windows, model, scalers)
    else:
        strategy = selected["strategy"] if degraded else meta["selected_model"].split(":")[-1]
        prediction = BaselineForecaster(strategy, 13).predict(windows.history_power, 13)
    calibration = selected["calibration"] if degraded else meta["calibration"]
    lo, hi = apply_prediction_interval(prediction, calibration)
    result = {"status": "degraded" if degraded else "available", "sampling_layout": LAYOUT,
        "evaluated_days": windows.size, "evaluated_points": windows.size*13,
        "metrics": regression_metrics(windows.target, prediction, lo, hi),
        "per_day": {start.date().isoformat(): regression_metrics(truth, pred)
            for start, truth, pred in zip(windows.forecast_starts, windows.target, prediction)},
        "per_hour": {str(hour): regression_metrics(windows.target[:, i], prediction[:, i]) for i, hour in enumerate(range(7, 20))},
        "evaluation_scope": "all complete target days in supplied source; not automatically an unseen holdout",
        "weather_input_policy": meta["weather_input_policy"], "limitations": _limitations(meta["weather_input_policy"]),
        "visualizations": [], "visualization_errors": []}
    if visualizations_dir:
        from .visuals import generate_actual_vs_forecast_visualization
        record = predict_day13(bundle_dir, source, input_spec=input_spec,
            forecast_date=windows.forecast_starts[-1], future_weather_source=future_weather_source)
        ref, errors = generate_actual_vs_forecast_visualization(day_index(windows.forecast_starts[-1]),
            windows.target[-1], record, visualizations_dir)
        result.update(visualizations=[ref] if ref else [], visualization_errors=errors)
    return result
