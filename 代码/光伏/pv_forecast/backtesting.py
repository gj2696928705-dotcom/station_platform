"""Stream every complete labelled window through frozen model bundles."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (DataValidationError, InputSpec, RegularizedData, WindowedData,
                   _apply_column_map, _normalise_timestamp, _read_source, build_windows, load_and_regularize)
from .experiments import MetricAccumulator
from .metrics import apply_prediction_interval
from .pipeline import _daylight_mask, _json_dump, _load_bundle, _prediction_spec
from .recalibration import _predict_windows
from .weather import WeatherForecastArchive
from .diagnostics import ForecastDiagnostics, PointStats


class IntervalAccumulator:
    """Pool sufficient statistics so chunking never averages RMSE or R²."""
    def __init__(self):
        self.point = MetricAccumulator()
        self.covered = 0
        self.width = self.error = 0.0

    def add(self, true, pred, lower, upper, mask):
        a, b, lo, hi = [np.asarray(value)[mask] for value in (true, pred, lower, upper)]
        self.point.add(a, b)
        self.covered += int(np.sum((a >= lo) & (a <= hi)))
        self.width += float(np.sum(hi - lo))
        self.error += float(np.sum(b - a))

    def result(self):
        result = self.point.result()
        n = result["count"]
        result.update(coverage=self.covered / n if n else None,
                      mean_interval_width_kw=self.width / n if n else None,
                      mean_error_kw=self.error / n if n else None)
        return result


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _take(windows, indices):
    return WindowedData(windows.history_power[indices], windows.future_features[indices], windows.target[indices],
        [windows.forecast_starts[i] for i in indices], windows.feature_names,
        windows.gru_history[indices], windows.history_feature_names,
        [windows.weather_issued_at[i] for i in indices] if len(windows.weather_issued_at) else [])


def backtest_bundles(bundle_dirs, source, output_path=None, input_spec=InputSpec(), *,
                     batch_windows=512, future_weather_source=None, progress=None, oracle_diagnostic_config=None):
    """Evaluate all eligible origins; batch size changes memory only.

    This strict primary-model backtest never trains, selects or calibrates.
    Unobserved target points and missing last historical observations are not
    scored as truth. Bundles sharing exactly the same predictor/scalers reuse
    predictions while applying their own saved intervals. Weather-state labels
    are used only for diagnostics after predictions, never as future inputs.
    """
    if not isinstance(batch_windows, int) or isinstance(batch_windows, bool) or batch_windows < 1:
        raise ValueError("batch_windows must be a positive integer")
    bundles = [Path(path).resolve() for path in bundle_dirs]
    if not bundles or len({p.name for p in bundles}) != len(bundles):
        raise ValueError("provide bundles with unique directory names")
    layouts = [json.loads((p/"model_metadata.json").read_text(encoding="utf-8")).get("sampling_layout") for p in bundles]
    if "day13_07_19" in layouts:
        if any(layout != "day13_07_19" for layout in layouts) or oracle_diagnostic_config is not None:
            raise ValueError("day13 backtests require only day13 bundles and do not use legacy oracle diagnostics")
        from .day13 import evaluate_day13
        models = {p.name: evaluate_day13(p, source, input_spec, future_weather_source=future_weather_source) for p in bundles}
        if any(v["status"] != "available" for v in models.values()):
            raise ValueError("strict day13 backtest requires all primary models available")
        result = {"status": "completed", "sampling_layout": "day13_07_19", "models": models,
            "evaluated_windows": next(iter(models.values()))["evaluated_days"],
            "weights_retrained": False, "calibration_fitted": False, "model_selected": False}
        if output_path:
            _json_dump(Path(output_path), result)
        return result
    entries, predictors, input_identity = [], {}, None
    for path in bundles:
        meta = json.loads((path / "model_metadata.json").read_text(encoding="utf-8"))
        spec = _prediction_spec(meta, input_spec)
        identity = json.dumps({"spec": asdict(spec), "interval": meta["interval_minutes"],
            "history_days": meta["history_days"], "horizon": meta["horizon_steps"]}, sort_keys=True)
        if input_identity is not None and input_identity != identity:
            raise ValueError("compared bundles must have the same input and window contract")
        input_identity = identity
        hashes = {name: _digest(path / name) for name in
                  ("model_metadata.json", "gru.keras", "scalers.joblib", "point_correction.joblib", "physics_tree.joblib") if (path / name).is_file()}
        key = (meta["selected_model"], hashes.get("gru.keras"), hashes.get("scalers.joblib"), hashes.get("point_correction.joblib"),
               hashes.get("physics_tree.joblib"), json.dumps(meta.get("physics_config"), sort_keys=True),
               tuple(meta.get("feature_names", [])), tuple(meta.get("history_feature_names", [])))
        if key not in predictors:
            loaded, gru, scalers = _load_bundle(path, allow_fallback=False)
            predictors[key] = (loaded, gru, scalers)
        entries.append({"path": path, "meta": meta, "hashes": hashes, "key": key,
                        "overall": {}, "months": {}, "weather": {}, "leads": {}, "diagnostics": ForecastDiagnostics()})
    raw = _apply_column_map(_read_source(source), spec)
    canonical_spec = replace(spec, column_map={})
    timestamps = _normalise_timestamp(raw[spec.timestamp_col], spec.timezone)
    regular = load_and_regularize(raw, canonical_spec, interval_minutes=meta["interval_minutes"])
    index = regular.frame.index
    observed_target = index.isin(timestamps)
    raw_last_valid = raw[[spec.target_col, *spec.history_weather_columns]].notna().all(axis=1).to_numpy()
    last_history_available = pd.Series(raw_last_valid, index=timestamps).reindex(index, fill_value=False).to_numpy()
    horizon, interval = int(meta["horizon_steps"]), int(meta["interval_minutes"])
    history = int(meta["history_days"] * 1440 // interval)
    first, end = history, len(index) - horizon + 1
    if end <= first:
        raise DataValidationError("complete history and target horizons are required for backtesting")
    observed_horizons = np.lib.stride_tricks.sliding_window_view(observed_target, horizon).all(axis=1)
    diagnostic_weather = (pd.Series(raw.weather_type.to_numpy(), index=timestamps).reindex(index).to_numpy()
                          if "weather_type" in raw else None)
    oracle_prediction = None
    oracle_stats, oracle_diagnostics = PointStats(), ForecastDiagnostics()
    if oracle_diagnostic_config is not None:
        from .physics import irradiance_to_power, load_physics_config
        physics = load_physics_config(oracle_diagnostic_config)
        if not {"SWGDN", "DHI", "DNI", "temperature_c"}.issubset(raw.columns):
            raise DataValidationError("ideal weather diagnostic requires observed SWGDN,DHI,DNI,temperature_c")
        oracle_prediction = pd.Series(irradiance_to_power(timestamps, raw.SWGDN, raw.DHI, raw.DNI,
            raw.temperature_c, physics), index=timestamps).reindex(index).to_numpy()
    archive = (WeatherForecastArchive(future_weather_source, spec.weather_columns, spec.timezone)
               if future_weather_source is not None else None)
    evaluated = 0
    first_scored = last_scored = None
    for lo in range(first, end, batch_windows):
        hi = min(end, lo + batch_windows)
        local = RegularizedData(regular.frame.iloc[lo-history:hi+horizon-1], regular.quality, interval, regular.timezone)
        windows = build_windows(local, history, len(local.frame), meta["history_days"], horizon,
            target_col=spec.target_col, history_weather_columns=spec.history_weather_columns,
            weather_columns=spec.weather_columns, latitude=spec.latitude, longitude=spec.longitude,
            weather_archive=archive)
        if windows.size == 0:
            continue
        positions = index.get_indexer(windows.forecast_starts)
        usable = observed_horizons[positions] & last_history_available[positions-1]
        if not usable.all():
            windows = _take(windows, np.flatnonzero(usable))
            positions = positions[usable]
        if windows.size == 0:
            continue
        for entry in entries:
            if (tuple(windows.feature_names) != tuple(entry["meta"].get("raw_feature_names", entry["meta"]["feature_names"]))
                    or tuple(windows.history_feature_names) != tuple(entry["meta"].get("raw_history_feature_names", entry["meta"]["history_feature_names"]))):
                raise ValueError("constructed feature order does not match the frozen model contract")
        evaluated += windows.size
        first_scored = first_scored or windows.forecast_starts[0].isoformat()
        last_scored = windows.forecast_starts[-1].isoformat()
        target_positions = positions[:, None] + np.arange(horizon)[None, :]
        daylight = _daylight_mask(windows.forecast_starts, horizon, interval, spec.latitude, spec.longitude)
        months = index.month.to_numpy()[target_positions]
        weather = diagnostic_weather[target_positions] if diagnostic_weather is not None else None
        preceding_weather = diagnostic_weather[positions-1] if diagnostic_weather is not None else None
        predictions = {key: _predict_windows(windows, *predictor) for key, predictor in predictors.items()}
        for entry in entries:
            prediction = predictions[entry["key"]]
            entry["diagnostics"].add(windows.target, prediction, daylight, weather, preceding_weather)
            lower, upper = apply_prediction_interval(prediction, entry["meta"]["calibration"], daylight)
            def add(store, name, mask):
                store.setdefault(name, IntervalAccumulator()).add(windows.target, prediction, lower, upper, mask)
            for period, mask in (("all_day", np.ones_like(daylight)), ("daylight", daylight), ("night", ~daylight)):
                add(entry["overall"], period, mask)
                for month in np.unique(months):
                    add(entry["months"].setdefault(f"{month:02d}", {}), period, mask & (months == month))
            if weather is not None:
                for state in pd.unique(weather.ravel()):
                    if not pd.isna(state):
                        add(entry["weather"], str(state), daylight & (weather == state))
            for start in range(0, horizon, max(1, 360 // interval)):
                stop = min(horizon, start + max(1, 360 // interval))
                mask = daylight.copy()
                mask[:, :start] = False
                mask[:, stop:] = False
                add(entry["leads"], f"steps_{start+1:02d}_{stop:02d}", mask)
        if oracle_prediction is not None:
            oracle = oracle_prediction[target_positions]
            oracle_stats.add(windows.target, oracle)
            oracle_diagnostics.add(windows.target, oracle, daylight, weather, preceding_weather)
        if progress:
            progress(f"已评价 {evaluated} 个完整窗口；当前预测起点 {last_scored}")
    if evaluated == 0:
        raise DataValidationError("no complete observed targets and primary-model histories are available")
    simulated = "is_simulated" in raw and bool(raw.is_simulated.astype(str).str.lower().eq("true").all())
    report = {"status": "completed", "operation": "frozen_model_full_window_backtest",
        "data_kind": "simulated" if simulated else "not_established",
        "weights_retrained": False, "calibration_fitted": False, "model_selected": False,
        "eligible_origins": end-first, "evaluated_windows": evaluated, "skipped_origins": end-first-evaluated,
        "first_forecast_start": first_scored, "last_forecast_start": last_scored,
        "horizon_steps": horizon, "interval_minutes": interval, "batch_windows": batch_windows,
        "evaluation_scope": "every eligible 15-minute origin; overlapping targets count per origin",
        "weather_diagnostics": "labelled target weather_type used for scoring only; not future model inputs",
        "input_quality": regular.quality.as_dict(), "models": {}}
    for entry in entries:
        if any(_digest(entry["path"] / name) != value for name, value in entry["hashes"].items()):
            raise RuntimeError("model bundle changed during frozen backtesting")
        report["models"][entry["path"].name] = {
            "bundle_dir": str(entry["path"]), "model_version": entry["meta"]["model_version"],
            "bundle_sha256": entry["hashes"], "calibration": entry["meta"]["calibration"],
            "overall": {name: accumulator.result() for name, accumulator in entry["overall"].items()},
            "months": {month: {name: accumulator.result() for name, accumulator in values.items()}
                       for month, values in entry["months"].items()},
            "daylight_weather": {name: accumulator.result() for name, accumulator in entry["weather"].items()},
            "daylight_lead_groups": {name: accumulator.result() for name, accumulator in entry["leads"].items()},
            "error_diagnostics": entry["diagnostics"].result(),
            "after_parent_input_window": index[0] > pd.Timestamp(entry["meta"]["input_data_window"]["end"])}
    report["source_sha256"] = _digest(source) if isinstance(source, (str, Path)) else hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest()
    if oracle_prediction is not None:
        report["ideal_future_weather_diagnostic"] = {"diagnostic_only": True, "model_selection_eligible": False,
            "production_eligible": False, "uses_actual_future_weather": True,
            "method": "independent physical reconstruction with target-period observed irradiance and temperature",
            "not_a_strict_mathematical_bound": True, "evaluated_windows": evaluated,
            "all_day": oracle_stats.result(), "error_diagnostics": oracle_diagnostics.result()}
    if output_path is not None:
        _json_dump(Path(output_path), report)
    return report
